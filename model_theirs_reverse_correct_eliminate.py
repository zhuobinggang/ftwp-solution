# 2025.5.16 在训练好的模型的基础上改善ucb1逻辑，使用model_danger_command.py来判断危险指令，对于危险指令，我们将其设定为执行过一次
# 2025.6.30 Action elimination
from game_command_generate import Game_command_generate_bert_filter, default_game
from dataset_create_taku import row_to_game_state, get_cv_games
from dataset_create_taku_command_generate import read_csv_dataset
import re
import pandas as pd
from tqdm import tqdm
from game import Game_state, test_game
from common_new import logging, beutiful_print_command_and_probs
from bert_utils import default_tokenizer, special_tokens_dict, EMPTY_RECIPE, EMPTY_INVENTORY
from bert_utils import BertInput, command_indexs_tokenized, init_bert_ours, DEVICE, NextCommandResult
logger = logging.getLogger('model_theirs')
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch import nn, optim
from functools import lru_cache
import numpy as np
from recordclass import recordclass
from pydash.arrays import chunk

CUT_COMMAND_PREFIX = ['slice', 'chop', 'dice']
COOK_COMMAND_PREFIX = ['cook']
TAKE_COMMAND_PREFIX = ['take']
OTHERS = ['prepare', 'eat']
MAY_GET_POINT_COMMAND_PREFIX = CUT_COMMAND_PREFIX + COOK_COMMAND_PREFIX + TAKE_COMMAND_PREFIX + OTHERS

BEST_MODELS = [4, 4, 2]
SAVE_DIR = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_theirs'
TRAIN_SPLIT = 'train'
# PART_VALID_SPLIT = 'partial_valid'
VALID_SPLIT = 'valid'
TEST_SPLIT = 'test'
MAX_TEST_STEP = 100
MAX_TOKEN_SIZE = 342
NEGATIVE_SAMPLE_SIZE = 4

DANGER_FILTER_ON = True

# NOTE: For testing
# TRAIN_SPLIT = 'fake_test'
# VALID_SPLIT = 'fake_test'
# TEST_SPLIT = 'fake_test'
# MAX_TEST_STEP = 10

GAME_INIT_FUNC = Game_command_generate_bert_filter

def get_writer():
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    return writer

def bert_tokenize_prompt_cut_theirs(game_state: Game_state, action: str):
    toker = default_tokenizer()
    CLS, SEP = special_tokens_dict().cls, special_tokens_dict().sep
    text = f'{CLS} '
    # before_history_text += f"Room: {game_state.room} {SEP} "
    inventory_text = game_state.inventory_clean().strip()
    if inventory_text == '':
        inventory_item_count = 0
        inventory_text = EMPTY_INVENTORY
    else:
        inventory_item_count = 1 + inventory_text.count(',')
    text += f'{inventory_item_count} {inventory_text} '
    recip_text = game_state.recipe_clean().strip()
    if recip_text == '':
        recip_text = EMPTY_RECIPE
    text += f"{recip_text} {game_state.description_clean()} " # NOTE: 2025.5.11 space is important!
    text_b = f"{SEP} {action} {SEP}"
    tokens = toker.encode(text, add_special_tokens=False) # list of numbers
    text_b_tokens = toker.encode(text_b, add_special_tokens=False)
    if len(tokens) + len(text_b_tokens) > MAX_TOKEN_SIZE:
        tokens = tokens[:MAX_TOKEN_SIZE - len(text_b_tokens)]
    return tokens, text_b_tokens

# NOTE: 使用CLS token作为解码token
def to_bert_input_theirs(state: Game_state, action: str, positive = True, need_padding = True):
    a_tokens, b_tokens = bert_tokenize_prompt_cut_theirs(state, action) # (length)
    prompt_ids = a_tokens + b_tokens
    attention_mask = [1] * len(prompt_ids)
    pad_size = 0
    if need_padding and len(prompt_ids) < MAX_TOKEN_SIZE:
        pad_size = MAX_TOKEN_SIZE - len(prompt_ids)
        prompt_ids += [default_tokenizer().pad_token_id] * pad_size
        attention_mask += [0] * pad_size
    if need_padding:
        assert len(prompt_ids) == MAX_TOKEN_SIZE, f"prompt_ids length {len(prompt_ids)} != {MAX_TOKEN_SIZE}"
    labels = [-100] * MAX_TOKEN_SIZE if need_padding else [-100] * len(prompt_ids)
    action_idx = 1 if positive else 0
    labels[0] = command_indexs_tokenized()[action_idx]
    # prepare token_type_ids
    token_type_ids = [0] * (len(a_tokens) + 1) + [1] * (len(b_tokens) - 1) # 需要注意的是，b_tokens的第一个token是SEP，所以需要-1
    if need_padding:
        token_type_ids += [0] * pad_size
    return BertInput(
        input_ids = prompt_ids,
        attention_mask = attention_mask,
        labels = labels,
        token_type_ids = token_type_ids
    )

@lru_cache(maxsize=1)
def dataloader_get(split = 'train', batch_size = 8):
    csv = read_csv_dataset(split = split)
    csv = csv.sample(frac=1)
    bert_inputs = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        state = row_to_game_state(row) # NOTE: 2025.5.5 打乱以提高模型的泛化能力
        negative_commands = [command for command in row['admissible_commands'] if command != row['action']]
        negative_commands = negative_commands[:NEGATIVE_SAMPLE_SIZE]
        for command in negative_commands:
            bert_input = to_bert_input_theirs(state, command, positive=False, need_padding=True)
            bert_inputs.append(bert_input)
        bert_input = to_bert_input_theirs(state, row['action'], positive=True, need_padding=True)
        bert_inputs.append(bert_input)
    all_input_ids = torch.tensor([bert_input.input_ids for bert_input in bert_inputs], dtype=torch.long)
    all_attention_mask = torch.tensor([bert_input.attention_mask for bert_input in bert_inputs], dtype=torch.long)
    all_label_ids = torch.tensor([bert_input.labels for bert_input in bert_inputs], dtype=torch.long)
    # NOTE: only for their model
    all_token_type_ids = torch.tensor([bert_input.token_type_ids for bert_input in bert_inputs], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_attention_mask, all_label_ids, all_token_type_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

def get_next_command_do_not_use(bert, game_state: Game_state):
        # 对于每一个action，计算它的概率
        commands = game_state.filtered_available_commands()
        bert_inputs = []
        for command in commands:
            bert_input = to_bert_input_theirs(game_state, command, positive=True, need_padding=True)
            bert_inputs.append(bert_input)
        command_probs = []
        for bert_input in bert_inputs:
            input_ids = torch.tensor([bert_input.input_ids], dtype=torch.long).to(DEVICE)
            attention_mask = torch.tensor([bert_input.attention_mask], dtype=torch.long).to(DEVICE)
            # NOTE: 2025.5.11 RoBERTa don't use token_type_ids! Error happens if use it!
            # token_type_ids = torch.tensor([bert_input.token_type_ids], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                cls_token_index = 0
                logits = logits[0, cls_token_index] # (30522)
                command_length = 2 # 0 or 1
                command_indexs = command_indexs_tokenized()[:command_length]
                command_logits = logits[command_indexs] # (2)
                command_probs.append(command_logits.softmax(dim=0)[1].item()) # 取正例的概率
        command_index = np.argmax(command_probs)
        max_prob_command = commands[command_index]
        # beutiful_print_command_and_probs(commands, command_probs)
        result = NextCommandResult(command_index, max_prob_command, command_probs)
        return result

@torch.no_grad()
def batch_predict(bert, batch_bert_input):
    input_ids = torch.tensor([bert_input.input_ids for bert_input in batch_bert_input], dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor([bert_input.attention_mask for bert_input in batch_bert_input], dtype=torch.long).to(DEVICE)
    # NOTE: 2025.5.11 RoBERTa don't use token_type_ids! Error happens if use it!
    # token_type_ids = torch.tensor([bert_input.token_type_ids for bert_input in batch_bert_input], dtype=torch.long).to(DEVICE)
    outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    cls_token_index = 0
    logits = logits[:, cls_token_index] # (batch_size, 30522)
    command_length = 2 # 0 or 1
    command_indexs = command_indexs_tokenized()[:command_length]
    command_logits = logits[:, command_indexs] # (batch_size, 2)
    return command_logits.softmax(dim=1)[:, 1].tolist() # probabilities of positive class

def get_next_command_batch(bert, game_state: Game_state, commands, batch_size = 8):
    # 对于每一个action，计算它的概率
    bert_inputs = []
    for command in commands:
        bert_input = to_bert_input_theirs(game_state, command, positive=True, need_padding=True)
        bert_inputs.append(bert_input)
    command_probs = []
    for batch_bert_input in chunk(bert_inputs, batch_size):
        command_probs += batch_predict(bert, batch_bert_input)
    command_index = np.argmax(command_probs)
    max_prob_command = commands[command_index]
    # beutiful_print_command_and_probs(commands, command_probs)
    result = NextCommandResult(command_index, max_prob_command, command_probs)
    return result

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.bert = init_bert_ours()
        self.bert = None
        self.prefix = 'roberta_ours'
    def init_bert(self):
        if not self.bert:
            self.bert = init_bert_ours()
    def predict(self, game_state:Game_state): # @return: action
        result = get_next_command_batch(self.bert, game_state, game_state.filtered_available_commands())
        return result.command
    def save_checkpoint(self, base_path = 'log', epoch = -1):
        path = f'{base_path}/{self.prefix}_epoch_{epoch}.pth'
        torch.save({
            'iteration': epoch,
            'state': self.state_dict(),
        }, path)
    def load_checkpoint(self, path):
        self.init_bert() # NOTE: 需要先初始化然后加载
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint['state'])

# ^^^^^^^^^^^^^^^^^^^^^^^^ w/o UCB1 ^^^^^^^^^^^^^^^^^^^^^^^^
# vvvvvvvvvvvvvvvvvvvvvvvv UCB1 vvvvvvvvvvvvvvvvvvvvvvv

Room = recordclass('Room', 'name east west north south')

# 只考虑房间名，库存状况和recipe的检查状况
def game_state_to_ucb1_key(game_state: Game_state):
    recipe_got = True if game_state.recipe_clean() else False
    return f'Room: {game_state.room}\nInventory: {game_state.inventory_clean()}\nRecipe: {recipe_got}'

def ucb1(action_cnt, total_cnt):
    if action_cnt == 0:
        return 5
    else:
        return np.sqrt(2*np.log(total_cnt)/action_cnt) # 如果total_cnt=10, action_cnt=1，值大概为2.15，总之都不会比2.5大的感觉。


def draw_chart():
    import matplotlib.pyplot as plt
    import numpy as np
    # 全体の試行回数 n（log項に使う）
    n_total = 100
    # n_i の値（1〜100）
    n_i_values = np.arange(1, 101)
    # 平均報酬は一定と仮定（例：0.5）
    avg_reward = 0
    # UCB1スコア計算（n_i > 0 の場合）
    ucb_scores = avg_reward + np.sqrt(2 * np.log(n_total) / n_i_values)
    # n_i = 0 のときのスコア
    ucb_zero = avg_reward + 5
    # グラフ描画
    plt.figure(figsize=(8, 5))
    plt.plot(n_i_values, ucb_scores, label="UCB1 (n_i > 0)")
    plt.axhline(y=ucb_zero, color='r', linestyle='--', label="UCB1 (n_i = 0)")
    plt.title("UCB1 Score vs. n_i (n_total = 100)")
    plt.xlabel("n_i (number of times arm i was selected)")
    plt.ylabel("UCB1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def maxmin_norm(p): # 返回0-1之间的值
    return (p - p.min())/(p.max() - p.min())

def choose_action_ubc1(logits, action_visited_count, alpha=1):
    """
    :param logits: vector with logits for actions
    :param sacnt: vector with counts for each visit of the action
    :returns: action number
    """
    total_visits = sum(action_visited_count) # 该状态下所有行动的执行次数
    uscore = [alpha*ucb1(v, total_visits) for v in action_visited_count] # 加权分
    ssc = maxmin_norm(logits) + torch.tensor(uscore).cuda() # 如果所有指令都没有访问过，那么所有指令的分数都是5，很公平。如果指令被访问过，那么logits几乎不影响它的分数 -> 意味着模型会优先选择没有访问过的指令
    return ssc.argmax(), ssc.softmax(dim=0)


class Model_ucb1(Model):
    def __init__(self):
        super(Model_ucb1, self).__init__()
        self.prefix = 'roberta_ours_ucb1'
        self.reset_state_action_count()
    def incresase_state_action_count(self, key, action):
        self.state_action_count[key][action] += 1
    def get_state_action_count(self, key, action):
        if key not in self.state_action_count:
            self.state_action_count[key] = {}
        if action not in self.state_action_count[key]:
            self.state_action_count[key][action] = 0
        return self.state_action_count[key][action]
    def reset_state_action_count(self, room_name = ''):
        logger.debug(f'\n\n\n === \n\n重新开始，清空地图信息, 房间名: {room_name}')
        self.state_action_count = {} # 记录每个状态下的动作选择次数
    def reset_state_action_count_if_need(self, game_state: Game_state):
        action_obs_pairs = game_state.clean_action_obs_pairs()
        if len(action_obs_pairs) == 0:
            self.reset_state_action_count(game_state.room)
        self.current_room = game_state.room # 总是要更新当前房间，但是在更新之前需要先更新世界地图（如果有必要）
    def calculated_state_action_count(self, game_state: Game_state, actions):
        # NOTE: 使用move_action_mask来促进模型探索新的房间
        state_key = game_state_to_ucb1_key(game_state)
        state_action_executed_count = [self.get_state_action_count(state_key, action) for action in actions]
        # 通过mask来屏蔽掉已经知道的房间
        state_action_executed_count_mask = [0] * len(actions)
        all_direction_known = True
        room_object = game_state.worldMap[game_state.room]
        direction_count = 0
        for idx, (action, executed_count) in enumerate(zip(actions, state_action_executed_count)):
            if action.startswith('go '):
                direction_count += 1
                direction = action.replace('go ', '')
                # dbg(f'Room {game_state.room} Dcirection {direction} exist, executed {executed_count} times.')
                if direction not in room_object: # 未知房间
                    all_direction_known = False
                    # dbg(f'Room {game_state.room} Dcirection {direction} unknown.')
                elif executed_count == 0: # 知道房间存在，但是没有真正执行过
                    state_action_executed_count_mask[idx] = 1
                    # dbg(f'Room {game_state.room} Dcirection {direction} known but never executed.')
                else: # 知道房间存在，并且执行过
                    # dbg(f'Room {getattr(room_object, direction)} known and already visited {executed_count} times.')
                    pass
        if all_direction_known: # 清空所有的已知房间的值
            state_action_executed_count_mask = [0] * len(actions)
            if direction_count > 0:
                # logger.debug(f'All {direction_count} directions are known, resetting the mask.')
                pass
        masked_state_action_executed_count = [a + b for a, b in zip(state_action_executed_count, state_action_executed_count_mask)]
        return masked_state_action_executed_count
    def predict(self, game_state:Game_state):
        # NOTE: 更新世界地图(根据上一步动作的结果)，只要发生移动必须对链接进行更新
        all_actions = game_state.filtered_available_commands()
        all_actions = [command for command in all_actions if command.split()[0] not in MAY_GET_POINT_COMMAND_PREFIX] # NOTE: 过滤掉可能得分的指令
        if len(all_actions) == 0:
            return 'inventory'
        self.reset_state_action_count_if_need(game_state)
        masked_state_action_executed_count = self.calculated_state_action_count(game_state, all_actions)
        # NOTE: 获取logits并使用ucb1算法选择动作
        result = get_next_command_batch(self.bert, game_state, all_actions)
        logits = result.logits # (actions_length)
        logits = torch.tensor(logits).to(DEVICE)
        best_action_idx, action_prob = choose_action_ubc1(logits, masked_state_action_executed_count)
        best_action = all_actions[best_action_idx]
        state_key = game_state_to_ucb1_key(game_state)
        self.incresase_state_action_count(state_key, best_action)
        if False:
            logger.debug(f'Recipe: {game_state.recipe_clean()}\n')
            logger.debug(f'Description: {game_state.description_clean()}\n')
            logger.debug(f'Inventory: {game_state.inventory_clean()}\n')
            logger.debug(f'Inventory: {game_state.clean_action_obs_pairs()[-5:]}\n')
            beutiful_print_command_and_probs(actions, action_prob, log_func=logger.debug)
            logger.debug(f'Action: {best_action}\n\n')
        return best_action


def train(model, batch_size = 8, split = 'train', log_name = ''):
    writer = get_writer()
    train_dataloader = dataloader_get(split=split, batch_size=batch_size)
    # training
    from accelerate import Accelerator
    accelerator = Accelerator()
    # model.cuda()
    model.train()
    logger.debug('Model train on.')
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) # 从1e-3到2e-5
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        input_ids, input_mask, label_ids, token_type_ids = batch
        # NOTE: 2025.5.11 RoBERTa don't use token_type_ids! Error happens if use it!
        outputs = model.bert(input_ids=input_ids.to(DEVICE), 
                   attention_mask=input_mask.to(DEVICE), 
                   # token_type_ids=token_type_ids.to(DEVICE),
                   labels=label_ids.to(DEVICE))
        loss = outputs.loss
        accelerator.backward(loss)
        writer.add_scalar(f'Loss/train_{log_name}', loss.item(), batch_idx)
        optimizer.step()
        optimizer.zero_grad()

def train_repeat(repeat = 3, epoch = 8, batch_size = 8):
    global BEST_MODELS
    # TRAIN_SPLIT = 'fake_test'
    # FULL_VALID_SPLIT = 'fake_test'
    INIC_FUNC = Model_ucb1
    ucb1_on = 'with UCB1' if INIC_FUNC == Model_ucb1 else 'w/o UCB1'
    for rp in range(repeat):
        model = get_model(init_func = INIC_FUNC)
        model.prefix = f'roberta_theirs_repeat_{rp}'
        max_score = 0
        for i in range(epoch):
            train(model, batch_size=batch_size, split=TRAIN_SPLIT, log_name=f'{rp}')
            score, avg_step = valid_all(model, split=VALID_SPLIT, game_init_func=Game_command_generate_bert_filter)
            logger.error(f'Full valid score ({rp}) {ucb1_on}: {score}, average step {avg_step}')
            print(f'Full valid score ({rp}) {ucb1_on}: {score}, average step {avg_step}')
            # get_writer().add_scalar(f'Score/valid_rp{rp}', score, i)
            if score > max_score:
                max_score = score
                BEST_MODELS[rp] = i
                logger.error(f'Best model ({rp}) at epoch {i}, score {max_score}, average step {avg_step}. BEST_MODELS: {BEST_MODELS}')
                # get_writer().add_scalar(f'Score/best_valid_rp{rp}', score, i)
                # 补上测试分数
                score, avg_step = valid_all(model, split=TEST_SPLIT, game_init_func=Game_command_generate_bert_filter)
                logger.error(f'Full test score ({rp}) {ucb1_on}: {score}, average step {avg_step}')
                print(f'Full test score ({rp}) {ucb1_on}: {score}, average step {avg_step}')
            model.save_checkpoint(base_path = SAVE_DIR, epoch=i)

def test_script():
    game = default_game()
    model = Model()
    model.init_bert()
    _ = model.cuda()
    _ = game.reset()
    game_state = game.to_game_state()
    commands = game_state.filtered_available_commands()
    bert_input = to_bert_input_theirs(game_state, commands[0], positive=True, need_padding=True)
    input_ids = torch.tensor([bert_input.input_ids], dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor([bert_input.attention_mask], dtype=torch.long).to(DEVICE)
    token_type_ids = torch.tensor([bert_input.token_type_ids], dtype=torch.long).to(DEVICE)
    return model.bert(input_ids=input_ids, attention_mask=attention_mask)

def valid_all(model: Model, split = 'partial_valid', game_init_func = None):
    if game_init_func is None:
        game_init_func = GAME_INIT_FUNC
        assert GAME_INIT_FUNC == Game_command_generate_bert_filter, '默认使用bert来过滤合成的动作'
    game_paths = get_cv_games(split=split)
    score = 0
    max_score = 0
    steps = []
    logger.debug(f'Validating {split} games, total {len(game_paths)}')
    for game_path in tqdm(game_paths, desc=f"Validating {split} games"):
        game = game_init_func(game_path)
        result = test_game(game, model, max_step=MAX_TEST_STEP)
        score += result.score
        max_score += result.max_score
        steps.append(result.step)
        # dbg(f'Valid results,  {result.score} / {result.max_score}, steps {result.step}, game {game_path}')
    average_step = np.mean(steps)
    return score / max_score, average_step

def valid_all_keep_individual_scores(model: Model, split = 'partial_valid', game_init_func = None):
    if game_init_func is None:
        game_init_func = GAME_INIT_FUNC
        assert GAME_INIT_FUNC == Game_command_generate_bert_filter, '默认使用bert来过滤合成的动作'
    game_paths = get_cv_games(split=split)
    scores = []
    steps = []
    logger.debug(f'Validating {split} games, total {len(game_paths)}')
    for game_path in tqdm(game_paths, desc=f"Validating {split} games"):
        game = game_init_func(game_path)
        result = test_game(game, model, max_step=MAX_TEST_STEP)
        scores.append(result.score /  result.max_score)
        steps.append(result.step)
        # dbg(f'Valid results,  {result.score} / {result.max_score}, steps {result.step}, game {game_path}')
    average_step = np.mean(steps)
    return scores, average_step

def get_model(checkpoint_path = None, init_func = Model):
    model = init_func()
    model.prefix = 'roberta_theirs'
    model.init_bert()
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    model.cuda()
    return model

def test_trained(repeat = 1): # default: 3
    INIC_FUNC = Model_ucb1
    ucb1_on = 'with UCB1' if INIC_FUNC == Model_ucb1 else 'w/o UCB1'
    logger.error(f'vvvvv\nTesting trained models {ucb1_on}')
    logger.error(f'Best models: {BEST_MODELS}')
    logger.error(f'ACTION ELIMINATION')
    # FULL_VALID_SPLIT = 'fake_test'
    # TEST_SPLIT = 'fake_test'
    for rp in range(repeat):
        path = f'{SAVE_DIR}/roberta_theirs_repeat_{rp}_epoch_{BEST_MODELS[rp]}.pth'
        model = get_model(path, init_func=INIC_FUNC)
        s1, avg_step = valid_all(model, split=VALID_SPLIT, game_init_func=Game_command_generate_bert_filter)
        print(f'Full valid score ({rp}): {s1} {ucb1_on}, average step {avg_step}')
        logger.error(f'Full valid score ({rp}): {s1} {ucb1_on}, average step {avg_step}')
        s2, avg_step = valid_all(model, split=TEST_SPLIT, game_init_func=Game_command_generate_bert_filter)
        print(f'Full test score ({rp}): {s2} {ucb1_on}, average step {avg_step}')
        logger.error(f'Full test score ({rp}): {s2} {ucb1_on}, average step {avg_step}')

def test_trained_individual_scores(repeat = 3):
    INIC_FUNC = Model_ucb1
    ucb1_on = 'with UCB1' if INIC_FUNC == Model_ucb1 else 'w/o UCB1'
    logger.error(f'vvvvv\nTesting trained models {ucb1_on}')
    logger.error(f'Best models: {BEST_MODELS}')
    # FULL_VALID_SPLIT = 'fake_test'
    # TEST_SPLIT = 'fake_test'
    for rp in range(repeat):
        path = f'{SAVE_DIR}/roberta_theirs_repeat_{rp}_epoch_{BEST_MODELS[rp]}.pth'
        model = get_model(path, init_func=INIC_FUNC)
        scores, avg_step = valid_all_keep_individual_scores(model, split=VALID_SPLIT, game_init_func=Game_command_generate_bert_filter)
        print(f'Ours: Full valid scores ({rp}): {scores} {ucb1_on}, average step {avg_step}')
        logger.error(f'Ours: Full valid scores ({rp}): {scores} {ucb1_on}, average step {avg_step}')
        scores, avg_step = valid_all_keep_individual_scores(model, split=TEST_SPLIT, game_init_func=Game_command_generate_bert_filter)
        print(f'Ours: Full test scores ({rp}): {scores} {ucb1_on}, average step {avg_step}')
        logger.error(f'Ours: Full test scores ({rp}): {scores} {ucb1_on}, average step {avg_step}')


def night_run():
    train_repeat(repeat=3, epoch=5, batch_size=8)
    # test_trained(repeat=3)