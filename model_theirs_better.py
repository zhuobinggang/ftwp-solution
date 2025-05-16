# 2025.5.12 将二分模型改为多选一
# 2025.5.12 重新实现cogniAgent（ftwp竞赛第一名），发现在测试集上的性能达到0.9。
# 之前实现的时候只有0.6的性能。性能提升可能有多个原因，包括引入RoBERTa，还有近期优化的指令合成方法。
# 需要在此之上进一步改善模型的性能才能撰写论文。
from model_theirs import default_tokenizer, special_tokens_dict, EMPTY_INVENTORY, EMPTY_RECIPE, BertInput
from model_theirs import Game_state, Game_command_generate_bert_filter, command_indexs_tokenized, NextCommandResult
from model_theirs import read_csv_dataset, row_to_game_state, lru_cache, tqdm, Model, Model_ucb1
from model_theirs import default_game, TensorDataset, DataLoader, RandomSampler, game_state_to_ucb1_key, DEVICE, choose_action_ubc1
from model_theirs import beutiful_print_command_and_probs, get_writer, logger, optim, get_model, get_cv_games, test_game
import numpy as np
import torch

BEST_MODELS = [0,0,0]
SAVE_DIR = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_theirs_multi_choice'
TRAIN_SPLIT = 'train'
# PART_VALID_SPLIT = 'partial_valid'
VALID_SPLIT = 'valid'
TEST_SPLIT = 'test'
MAX_TEST_STEP = 100
MAX_TOKEN_SIZE = 504

# NOTE: For testing
# TRAIN_SPLIT = 'fake_test'
# VALID_SPLIT = 'fake_test'
# TEST_SPLIT = 'fake_test'
# MAX_TEST_STEP = 10
GAME_INIT_FUNC = Game_command_generate_bert_filter

# vvvvvvvvvvvvvvvv 重写 vvvvvvvvvvvvvvv

def bert_tokenize_prompt_cut_theirs(game_state: Game_state):
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
    # NOTE: 将action改为action list
    text_b = f"{SEP}\n{game_state.available_commands_text()} {SEP}" # NOTE
    tokens = toker.encode(text, add_special_tokens=False) # list of numbers
    text_b_tokens = toker.encode(text_b, add_special_tokens=False)
    if len(tokens) + len(text_b_tokens) > MAX_TOKEN_SIZE:
        tokens = tokens[:MAX_TOKEN_SIZE - len(text_b_tokens)]
    return tokens, text_b_tokens


# For training
# NOTE: 使用CLS token作为解码token
def to_bert_input_theirs(state: Game_state, action = '', need_padding = True):
    a_tokens, b_tokens = bert_tokenize_prompt_cut_theirs(state) # (length)
    prompt_ids = a_tokens + b_tokens
    attention_mask = [1] * len(prompt_ids)
    pad_size = 0
    if need_padding and len(prompt_ids) < MAX_TOKEN_SIZE:
        pad_size = MAX_TOKEN_SIZE - len(prompt_ids)
        prompt_ids += [default_tokenizer().pad_token_id] * pad_size
        attention_mask += [0] * pad_size
    if need_padding:
        assert len(prompt_ids) == MAX_TOKEN_SIZE, f"prompt_ids length {len(prompt_ids)} != {MAX_TOKEN_SIZE}"
    if not action:
        return BertInput(
            input_ids = prompt_ids,
            attention_mask = attention_mask
        )
    else:
        labels = [-100] * MAX_TOKEN_SIZE if need_padding else [-100] * len(prompt_ids)
        action_idx = state.filtered_available_commands().index(action)
        labels[0] = command_indexs_tokenized()[action_idx] # NOTE
        return BertInput(
            input_ids = prompt_ids,
            attention_mask = attention_mask,
            labels = labels
        )

# For testing
@torch.no_grad()
def get_next_command(bert, game_state: Game_state):
        # 对于每一个action，计算它的概率
        commands = game_state.filtered_available_commands()
        bert_input = to_bert_input_theirs(game_state, action = '', need_padding=True)
        input_ids = torch.tensor([bert_input.input_ids], dtype=torch.long).to(DEVICE)
        attention_mask = torch.tensor([bert_input.attention_mask], dtype=torch.long).to(DEVICE)
        # NOTE: 2025.5.11 RoBERTa don't use token_type_ids! Error happens if use it!
        # token_type_ids = torch.tensor([bert_input.token_type_ids], dtype=torch.long).to(DEVICE)
        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        cls_token_index = 0
        logits = logits[0, cls_token_index] # (30522)
        command_length = len(commands)
        command_indexs = command_indexs_tokenized()[:command_length] # NOTE
        command_logits = logits[command_indexs] # (command_length)
        command_probs = command_logits.softmax(dim=0) # (command_length)
        # max_command_prob = command_probs.max().item()
        the_command_index = command_probs.argmax().item()
        max_prob_command = commands[the_command_index]
        # beutiful_print_command_and_probs(commands, command_probs)
        result = NextCommandResult(the_command_index, max_prob_command, command_probs)
        return result

@lru_cache(maxsize=1)
def dataloader_get(split = 'train', batch_size = 8):
    csv = read_csv_dataset(split = split)
    csv = csv.sample(frac=1)
    bert_inputs = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        state = row_to_game_state(row) # NOTE: 2025.5.5 打乱以提高模型的泛化能力
        bert_input = to_bert_input_theirs(state, row['action'], need_padding=True)
        bert_inputs.append(bert_input)
    all_input_ids = torch.tensor([bert_input.input_ids for bert_input in bert_inputs], dtype=torch.long)
    all_attention_mask = torch.tensor([bert_input.attention_mask for bert_input in bert_inputs], dtype=torch.long)
    all_label_ids = torch.tensor([bert_input.labels for bert_input in bert_inputs], dtype=torch.long)
    # NOTE: only for their model
    train_data = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

# vvvvvvvvvvvvvvvv 使用 vvvvvvvvvvvvvvvvv


class Model_ucb1(Model_ucb1):
    def predict(self, game_state:Game_state):
        # NOTE: 更新世界地图(根据上一步动作的结果)，只要发生移动必须对链接进行更新
        self.update_room_link(game_state)
        masked_state_action_executed_count = self.calculated_state_action_count(game_state)
        # NOTE: 获取logits并使用ucb1算法选择动作
        actions = game_state.filtered_available_commands()
        result = get_next_command(self.bert, game_state)
        logits = result.logits # (actions_length)
        logits = logits.clone().detach().to(DEVICE)
        best_action_idx, action_prob = choose_action_ubc1(logits, masked_state_action_executed_count)
        best_action = actions[best_action_idx]
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

class Model(Model):
    def predict(self, game_state:Game_state): # @return: action
        result = get_next_command(self.bert, game_state)
        return result.command


def test_script():
    game = default_game()
    _ = game.reset()
    game.act('go east')
    game_state = game.to_game_state()
    return game_state

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
        input_ids, input_mask, label_ids = batch
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


def test_trained(repeat = 3):
    INIC_FUNC = Model
    ucb1_on = 'with UCB1' if INIC_FUNC == Model_ucb1 else 'w/o UCB1'
    logger.error(f'vvvvv\nTesting trained models {ucb1_on}')
    logger.error(f'Best models: {BEST_MODELS}')
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

def night_run_test():
    train_repeat(repeat=2, epoch=2, batch_size=8) # with UCB1
    test_trained(repeat=2) # without UCB1

def night_run():
    train_repeat(repeat=3, epoch=5, batch_size=8) # with UCB1
    test_trained(repeat=3) # without UCB1