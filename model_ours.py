from transformers import BertForMaskedLM, AutoTokenizer
from functools import lru_cache
import common_new as common
from common_new import draw_line_chart, beutiful_print_command_and_probs
import torch
import torch.optim as optim
from torch import nn
import re
from game import Game_handle_recipe, game_state_from_game, Game_state, default_game, test_game
from game_command_generate import Game_command_generate
from dataset_create_taku import read_csv_dataset, get_cv_games
from bert_utils import default_tokenizer, init_bert_ours, action_select_loss, action_select_loss_batched
from bert_utils import get_next_command, tokenize_game_state, command_indexs_tokenized, to_bert_input, DEVICE
from dataset_create_taku import row_to_game_state
from tqdm import tqdm
from recordclass import recordclass
import numpy as np
from typing import List
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

Statistic = recordclass('Statistic', 'losses')

import logging
logger = logging.getLogger('model_ours')
dbg = logger.debug

# GAME_INIT_FUNC = Game_handle_recipe
TRAIN_SPLIT = 'train_command_generate'
PART_VALID_SPLIT = 'partial_valid'
FULL_VALID_SPLIT = 'valid'
TEST_SPLIT = 'test'
PART_TEST_SPLIT = 'partial_test'
# SAVE_DIR = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours_command_gen'
SAVE_DIR = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours'
# GAME_INIT_FUNC = Game_command_generate
GAME_INIT_FUNC = Game_handle_recipe


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.bert = init_bert_ours()
        self.bert = None
        self.prefix = 'roberta_ours'
    def init_bert(self):
        if not self.bert:
            self.bert = init_bert_ours()
    def loss_from_state(self, state:Game_state, action_idx: int):
        return action_select_loss(self.bert, state, action_idx)
    def loss_from_state_batched(self, states:List[Game_state], action_idxs:List[int]):
        return action_select_loss_batched(self.bert, states, action_idxs)
    def predict(self, game_state:Game_state):
        result = get_next_command(self.bert, game_state)
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


# vvvvvvvvvvvvv UCB1 Model vvvvvvvvvvvvvvv

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
        dbg(f'\n\n\n === \n\n重新开始，清空地图信息, 房间名: {room_name}')
        self.world_map = {}
        self.current_room = room_name # 并不会及时反应当前房间，而是在确定发生了移动之后才会更新
        if room_name != '':
            self.world_map[room_name] = Room(room_name, None, None, None, None)
        self.state_action_count = {} # 记录每个状态下的动作选择次数
    def update_room_link(self, game_state: Game_state):
        action_obs_pairs = game_state.clean_action_obs_pairs()
        if len(action_obs_pairs) == 0:
            self.reset_state_action_count(game_state.room)
        else:
            action, obs = action_obs_pairs[-1]
            if action.startswith('go '): # NOTE: Update current room, and connecting rooms
                prev_room_name = self.current_room
                assert prev_room_name != '', f'先前房间名为空，action: {action}, obs: {obs}'
                current_room_name = game_state.room
                assert current_room_name != '', f'当前房间名为空，action: {action}, obs: {obs}'
                if current_room_name not in self.world_map: # 说明来到一个新的房间
                    self.world_map[current_room_name] = Room(current_room_name, None, None, None, None)
                    # dbg(f'New room: {current_room_name}, prev room: {prev_room_name}')
                # dbg(f'Update room link, action: {action}, prev room: {prev_room_name}, now room: {current_room_name}')
                if prev_room_name not in self.world_map: # 说明为什么之前的房间会不存在？只有一种情况：在开始的第一步进行了移动
                    raise ValueError(f'Previous room {prev_room_name} not in world map, current room: {current_room_name}')
                prev_room_object = self.world_map[prev_room_name]
                # 更新链接 NOTE: 需要更新反向的链接
                current_room_object = self.world_map[current_room_name]
                if action == 'go east':
                    prev_room_object.east = current_room_name
                    current_room_object.west = prev_room_name
                elif action == 'go west':
                    prev_room_object.west = current_room_name
                    current_room_object.east = prev_room_name
                elif action == 'go north':
                    prev_room_object.north = current_room_name
                    current_room_object.south = prev_room_name
                elif action == 'go south':
                    prev_room_object.south = current_room_name
                    current_room_object.north = prev_room_name
                else:
                    logger.error(f'XXXXXXXXXXXXXX错误状况XXXXXXXXXXXXXX')
        self.current_room = game_state.room # 总是要更新当前房间，但是在更新之前需要先更新世界地图（如果有必要）
    def calculated_state_action_count(self, game_state: Game_state):
        # NOTE: 使用move_action_mask来促进模型探索新的房间
        state_key = game_state_to_ucb1_key(game_state)
        actions = game_state.filtered_available_commands()
        state_action_executed_count = [self.get_state_action_count(state_key, action) for action in actions]
        # 通过mask来屏蔽掉已经知道的房间
        state_action_executed_count_mask = [0] * len(actions)
        all_direction_known = True
        room_object = self.world_map[game_state.room]
        direction_count = 0
        for idx, (action, executed_count) in enumerate(zip(actions, state_action_executed_count)):
            if action.startswith('go '):
                direction_count += 1
                direction = action.replace('go ', '')
                # dbg(f'Room {game_state.room} Dcirection {direction} exist, executed {executed_count} times.')
                if getattr(room_object, direction) is None: # 未知房间
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
        self.update_room_link(game_state)
        masked_state_action_executed_count = self.calculated_state_action_count(game_state)
        # NOTE: 获取logits并使用ucb1算法选择动作
        actions = game_state.filtered_available_commands()
        result = get_next_command(self.bert, game_state)
        logits = result.logits # (actions_length)
        best_action_idx, action_prob = choose_action_ubc1(logits, masked_state_action_executed_count)
        best_action = actions[best_action_idx]
        state_key = game_state_to_ucb1_key(game_state)
        self.incresase_state_action_count(state_key, best_action)
        dbg(f'Recipe: {game_state.recipe_clean()}\n')
        dbg(f'Description: {game_state.description_clean()}\n')
        dbg(f'Inventory: {game_state.inventory_clean()}\n')
        beutiful_print_command_and_probs(actions, action_prob, log_func=dbg)
        dbg(f'Action: {best_action}\n\n')
        return best_action

# ^^^^^^^^^^^^^^^^^^^^^^^^^

def test():
    m = Model()
    g = default_game()
    _ = g.reset()
    g.act('go east')
    gs = game_state_from_game(g)
    # loss = m.loss_from_info(gs, 11)
    # print(loss)
    # print(m.predict(gs))
    return m, g, gs

def test2():
    m = Model()
    g = default_game()
    _ = g.reset()
    g.act('go east')
    state = game_state_from_game(g)
    prompt_ids = tokenize_game_state(state)
    action_idx = 11
    label_code_in_tokenizer = command_indexs_tokenized()[action_idx]
    return m, g, state

@lru_cache(maxsize=1)
def dataloader_get(split = TRAIN_SPLIT, batch_size = 8):
    # dataloader
    csv = read_csv_dataset(split=split)
    csv = csv.sample(frac=1) # shuffle to train
    bert_inputs = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        state = row_to_game_state(row)
        action_idx = state.filtered_available_commands().index(row['action'])
        bert_input = to_bert_input(state, action_idx)
        bert_inputs.append(bert_input)
    all_input_ids = torch.tensor([bert_input.input_ids for bert_input in bert_inputs], dtype=torch.long)
    all_attention_mask = torch.tensor([bert_input.attention_mask for bert_input in bert_inputs], dtype=torch.long)
    all_label_ids = torch.tensor([bert_input.labels for bert_input in bert_inputs], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

@lru_cache(maxsize=1)
def get_writer():
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    return writer

def train(model, batch_size = 8, split = TRAIN_SPLIT):
    train_dataloader = dataloader_get(split=split, batch_size=batch_size)
    # training
    from accelerate import Accelerator
    accelerator = Accelerator()
    # model.cuda()
    model.train()
    dbg('Model train on.')
    optimizer = optim.AdamW(model.parameters(), lr=2e-5) # 从1e-3到2e-5
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        input_ids, input_mask, label_ids = batch
        outputs = model.bert(input_ids=input_ids.to(DEVICE), 
                   attention_mask=input_mask.to(DEVICE), 
                   labels=label_ids.to(DEVICE))
        loss = outputs.loss
        accelerator.backward(loss)
        get_writer().add_scalar('Loss/train', loss.item(), batch_idx)
        optimizer.step()
        optimizer.zero_grad()

def valid_all(model: Model, split = 'partial_valid'):
    game_paths = get_cv_games(split=split)
    score = 0
    max_score = 0
    steps = []
    dbg(f'Validating {split} games, total {len(game_paths)}')
    for game_path in tqdm(game_paths, desc=f"Validating {split} games"):
        game = GAME_INIT_FUNC(game_path)
        result = test_game(game, model)
        score += result.score
        max_score += result.max_score
        steps.append(result.step)
        # dbg(f'Valid results,  {result.score} / {result.max_score}, steps {result.step}, game {game_path}')
    average_step = np.mean(steps)
    return score / max_score, average_step

def valid_all_by_model_path(model_path: str):
    model = Model()
    model.load_checkpoint(model_path)
    return valid_all(model)

# ================================

def get_model(checkpoint_path = None, init_func = Model):
    model = init_func()
    model.prefix = 'roberta_ours'
    model.init_bert()
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    return model


def train_reapeat(repeat = 3, epoch = 5, batch_size = 8):
    for rp in range(repeat):
        model = get_model()
        model.prefix = f'roberta_ours_repeat_{rp}'
        for i in range(epoch):
            train(model, batch_size=batch_size, split=TRAIN_SPLIT)
            score, avg_step = valid_all(model, split=FULL_VALID_SPLIT)
            dbg(f'Full valid score ({rp}): {score}, average step {avg_step}')
            # get_writer().add_scalar(f'Score/valid_rp{rp}', score, i)
            model.save_checkpoint(base_path = SAVE_DIR, epoch=i)

def test_trained():
    best_model_index = [4,4,4] # NOTE: 对于使用引擎选项的模型
    # best_model_index = [3,4,4] # NOTE: 对于使用生成选项的模型
    for rp in range(3):
        path = f'{SAVE_DIR}/roberta_ours_repeat_{rp}_epoch_{best_model_index[rp]}.pth'
        model = get_model(path, init_func=Model_ucb1)
        s1, avg_step = valid_all(model, split=FULL_VALID_SPLIT)
        dbg(f'Full valid score ({rp}): {s1}, average step {avg_step}')
        s2, avg_step = valid_all(model, split=TEST_SPLIT)
        dbg(f'Full test score ({rp}): {s2}, average step {avg_step}')

def test_normal_model(model_path = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours/roberta_ours_repeat_0_epoch_4.pth'):
    assert GAME_INIT_FUNC == Game_handle_recipe, f'模型路径 {model_path} 需要使用Game_handle_recipe'
    model = get_model(model_path, init_func=Model_ucb1)
    s1, avg_step = valid_all(model, split=PART_TEST_SPLIT)
    dbg(f'Full valid score: {s1}, average step {avg_step}')


def test_command_generate_model(model_path = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours_command_gen/roberta_ours_repeat_2_epoch_4.pth'):
    assert GAME_INIT_FUNC == Game_command_generate, f'模型路径 {model_path} 需要使用Game_command_generate'
    model = get_model(model_path, init_func=Model_ucb1)
    s1, avg_step = valid_all(model, split=PART_TEST_SPLIT)
    dbg(f'Full valid score: {s1}, average step {avg_step}')