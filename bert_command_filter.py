"""
<Description> [MASK] <open cmd0> [MASK] <open cmd1> [MASK] <go cmd0> [MASK] <go cmd1>

Use RoBERTa to decode [MASK] tokens. Labels: true & false.
"""
from bert_utils import default_tokenizer, init_bert_ours, action_select_loss, action_select_loss_batched
from game import game_state_from_game, Game_state, default_game, test_game
from typing import List
from bert_utils import get_next_command, special_tokens_dict, DEVICE
import torch
from torch import nn
import numpy as np
from bert_utils import default_tokenizer
from recordclass import recordclass
from functools import lru_cache
from common_new import logging, beutiful_print_command_and_probs
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import os
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from dataset_create_taku import get_cv_games

logger = logging.getLogger(__name__)
dbg = logger.debug

TRUE_LABEL = 'true'
FALSE_LABEL = 'false'

BERTInput = recordclass('BERTInput', 'input_ids label_ids label_ids_raw mask_idxs mask_ids extra')
PredictResult = recordclass('PredictResult', 'filtered_cmds filtered_probs cmds probs extra')

"""
# Decoded token_ids:
<s> Description: Kitchen Well I'll be you are in the place we're calling the kitchen You can make out an open fridge The fridge is empty what a horrible day You scan the room seeing an oven You wonder idly who left that here You can see a table On the table you make out a knife Huh weird You see a counter What a coincidence weren't you just thinking about a counter The counter is vast On the counter you make out a green apple a red apple a raw purple potato and a cookbook Hey want to see a stove Look over there a stove The stove is conventional Unfortunately there isn't a thing on it There is an open plain door leading south There is an open sliding patio door leading west There is an exit to the north Don't worry there is no door </s>
<mask> open plain door
<mask> open sliding patio door
<mask> open fridge
<mask> go north
<mask> go south
<mask> go west
</s>
# Decoded token_ids[mask_idxs]:
<mask><mask><mask><mask><mask><mask>
"""
# TODO: 如果选项太长要分批进行处理，不过目前应该不会有这个问题
def prompt_from_game_state(game_state: Game_state):
    desc_clean = game_state.description_clean()
    all_cmds = game_state.filtered_available_commands()
    open_cmds = [cmd for cmd in all_cmds if cmd.startswith('open')]
    go_cmds = [cmd for cmd in all_cmds if cmd.startswith('go')]
    open_go_cmds = open_cmds + go_cmds
    result = bert_input_command_filter_loss(desc_clean, open_go_cmds, open_go_cmds)
    result.extra = {'open_go_cmds': open_go_cmds}
    return result

def test_prompt_from_game_state():
    from game import Game_command_generate
    game = Game_command_generate('/home/taku/Downloads/cog2019_ftwp/games/partial_valid/tw-cooking-recipe1+cut+go9-Ly5OHNjvSvlmi3JK.ulx')
    obs, info = game.reset()
    game_state = game.to_game_state()
    game_state.admissible_commands = game.get_admissible_commands() # BUG: 如果在game_state_from_game中调用这个会导致无限循环
    return prompt_from_game_state(game_state)


def bert_input_command_filter(desc_clean, gen_open_go_cmds):
    SEP, CLS, MASK = special_tokens_dict().sep, special_tokens_dict().cls, special_tokens_dict().mask
    toker = default_tokenizer()
    prompt = f'{CLS} Description: {desc_clean} {SEP}\n'
    token_ids = toker.encode(prompt, add_special_tokens=False) # list of numbers
    mask_idxs = [len(token_ids)]
    for cmd in gen_open_go_cmds:
        prompt = f'{MASK} {cmd}\n'
        token_ids += toker.encode(prompt, add_special_tokens=False)
        mask_idxs.append(len(token_ids))
    token_ids += toker.encode(f'{SEP}', add_special_tokens=False)
    mask_idxs = mask_idxs[:-1] # remove the last one
    return BERTInput(input_ids=token_ids, mask_idxs=mask_idxs)

def bert_input_command_filter_loss(desc_clean, gen_open_go_cmds, cheat_open_go_cmds):
    """
    Filter the generated open/go commands based on the description.
    :param desc_clean: The cleaned description of the game state.
    :param gen_open_go_cmds: The generated open/go commands.
    :param cheat_open_go_cmds: The cheat open/go commands.
    """
    toker = default_tokenizer()
    result = bert_input_command_filter(desc_clean, gen_open_go_cmds)
    # 准备标签
    token_ids = np.array(result.input_ids)
    # label_ids = token_ids.copy()
    label_ids = np.full(token_ids.shape, -100) # -100 means ignore
    TRUE_LABEL_ID = toker.encode(TRUE_LABEL, add_special_tokens=False)[0]
    FALSE_LABEL_ID = toker.encode(FALSE_LABEL, add_special_tokens=False)[0]
    label_ids[result.mask_idxs] = [TRUE_LABEL_ID if cmd in cheat_open_go_cmds else FALSE_LABEL_ID for cmd in gen_open_go_cmds]
    result.label_ids = label_ids
    return result


def test_cal_loss():
    from game import Game_command_generate
    game = Game_command_generate('/home/taku/Downloads/cog2019_ftwp/games/partial_valid/tw-cooking-recipe1+cut+go9-Ly5OHNjvSvlmi3JK.ulx')
    obs, info = game.reset()
    game_state = game.to_game_state()
    game_state.admissible_commands = game.get_admissible_commands() # BUG: 如果在game_state_from_game中调用这个会导致无限循环
    result = prompt_from_game_state(game_state)
    token_ids, mask_idxs, extra = result.input_ids, result.mask_idxs, result.extra
    # prepare labels
    cheat_open_go_cmds = [cmd for cmd in info['admissible_commands'] if cmd.startswith('open') or cmd.startswith('go')]
    gen_open_go_cmds = extra['open_go_cmds']
    extra['cheat_open_go_cmds'] = cheat_open_go_cmds
    token_ids = np.array(token_ids)
    # label_ids = token_ids.copy()
    label_ids = np.full(token_ids.shape, -100) # -100 means ignore
    toker = default_tokenizer()
    TRUE_LABEL_ID = toker.encode(TRUE_LABEL, add_special_tokens=False)[0]
    FALSE_LABEL_ID = toker.encode(FALSE_LABEL, add_special_tokens=False)[0]
    label_ids[mask_idxs] = [TRUE_LABEL_ID if cmd in cheat_open_go_cmds else FALSE_LABEL_ID for cmd in gen_open_go_cmds]
    return token_ids, label_ids, extra


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.bert = init_bert_ours()
        self.bert = None
        self.prefix = 'roberta_command_filter'
    def init_bert(self):
        if not self.bert:
            self.bert = init_bert_ours()
    def loss(self, desc_clean, gen_open_go_cmds, cheat_open_go_cmds):
        bert_input = bert_input_command_filter_loss(desc_clean, gen_open_go_cmds, cheat_open_go_cmds)
        input_ids = torch.LongTensor(bert_input.input_ids).unsqueeze(0)
        label_ids = torch.LongTensor(bert_input.label_ids).unsqueeze(0)
        outputs = self.bert(input_ids=input_ids.to(DEVICE), labels=label_ids.to(DEVICE))
        return outputs.loss
    def loss_from_state_batched(self, gen_open_go_cmds, cheat_open_go_cmds):
        raise NotImplementedError("loss_from_state_batched is not implemented")
    def predict(self, desc_clean, gen_open_go_cmds):
        bert_input = bert_input_command_filter(desc_clean, gen_open_go_cmds)
        input_ids = torch.LongTensor(bert_input.input_ids).unsqueeze(0)
        with torch.no_grad():
            logits = self.bert(input_ids=input_ids.to(DEVICE)).logits # (1, len_of_masks, 30522)
        logits = logits[0, bert_input.mask_idxs] # (len_of_masks, 30522)
        toker = default_tokenizer()
        TRUE_LABEL_ID = toker.encode(TRUE_LABEL, add_special_tokens=False)[0]
        FALSE_LABEL_ID = toker.encode(FALSE_LABEL, add_special_tokens=False)[0]
        logits = logits[:, [TRUE_LABEL_ID, FALSE_LABEL_ID]] # (len_of_masks, 2)
        probs = torch.softmax(logits, dim=1) # (len_of_masks, 2)
        probs = probs[:, 0] # (len_of_masks)
        probs = probs.tolist()
        assert len(probs) == len(bert_input.mask_idxs)
        filtered_cmds = []
        filtered_probs = []
        for cmd, prob in zip(gen_open_go_cmds, probs):
            if prob > 0.5:
                filtered_cmds.append(cmd)
                filtered_probs.append(prob)
        return PredictResult(filtered_cmds, filtered_probs, gen_open_go_cmds, probs, bert_input.extra)
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

@lru_cache(maxsize=1)
def default_trained_model():
    # epoch 1, valid score: 1.0
    path = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours_command_filter/roberta_command_filter_0_epoch_0.pth'
    model = Model()
    model.load_checkpoint(path)
    model.to(DEVICE)
    return model

def use_bert_to_filter_command(desc_clean, gen_open_go_cmds):
    model = default_trained_model()
    result = model.predict(desc_clean, gen_open_go_cmds)
    return result

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvv Train vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# GAME_INIT_FUNC = Game_move_action_augment
TRAIN_SPLIT = 'train'
PART_VALID_SPLIT = 'partial_valid'
FULL_VALID_SPLIT = 'valid'
TEST_SPLIT = 'test'
PART_TEST_SPLIT = 'partial_test'
# SAVE_DIR = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours_command_gen'
SAVE_DIR = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours_command_filter'

def read_csv_dataset(inputpath = 'good_dataset', split = 'fake_test', suffix = '_command_generate'):
    df= pd.read_csv(os.path.join(inputpath,
                        f'open_go_filter_{split}{suffix}.csv'))
    df['cheat_open_go_cmds'] = df['cheat_open_go_cmds'].apply(eval)
    df['generated_open_go_cmds'] = df['generated_open_go_cmds'].apply(eval)
    return df

@lru_cache(maxsize=1)
def dataloader_get(split = TRAIN_SPLIT, batch_size = 8, MAX_SEQ_LEN = 480):
    PAD_TOKEN = special_tokens_dict().pad
    PAD_TOKEN_ID = default_tokenizer().encode(PAD_TOKEN, add_special_tokens=False)[0]
    # dataloader
    csv = read_csv_dataset(split=split)
    csv = csv.sample(frac=1) # shuffle to train
    bert_inputs = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        # bert_input = bert_input_command_filter(desc_clean, gen_open_go_cmds)
        bert_input = bert_input_command_filter_loss(row['description'], row['generated_open_go_cmds'], row['cheat_open_go_cmds']) # BERTInput(input_ids=token_ids, mask_idxs=mask_idxs, label_ids=label_ids)
        # padding
        meaning_full_length = len(bert_input.input_ids)
        if meaning_full_length > MAX_SEQ_LEN:
            logger.warning(f'Meaning full length {meaning_full_length} > MAX_SEQ_LEN {MAX_SEQ_LEN}, skip.')
        else:
            bert_input.input_ids = np.concatenate((bert_input.input_ids, np.full(MAX_SEQ_LEN - meaning_full_length, PAD_TOKEN_ID, dtype=int))) 
            bert_input.label_ids = np.concatenate((bert_input.label_ids, np.full(MAX_SEQ_LEN - meaning_full_length, -100, dtype=int)))
            bert_input.mask_ids = np.concatenate((np.ones(meaning_full_length, dtype=int), np.zeros(MAX_SEQ_LEN - meaning_full_length, dtype=int)))
            bert_inputs.append(bert_input)
    all_input_ids = torch.tensor([bert_input.input_ids for bert_input in bert_inputs], dtype=torch.long)
    # all_mask_idxs = torch.tensor([bert_input.mask_idxs for bert_input in bert_inputs], dtype=torch.long)
    all_label_ids = torch.tensor([bert_input.label_ids for bert_input in bert_inputs], dtype=torch.long)
    all_mask_ids = torch.tensor([bert_input.mask_ids for bert_input in bert_inputs], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_mask_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

@lru_cache(maxsize=1)
def get_writer():
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    return writer

def train(model, batch_size = 8, split = TRAIN_SPLIT, log_name = ''):
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
        input_ids, mask_ids, label_ids = batch
        outputs = model.bert(input_ids=input_ids.to(DEVICE), 
                   attention_mask=mask_ids.to(DEVICE), 
                   labels=label_ids.to(DEVICE))
        loss = outputs.loss
        accelerator.backward(loss)
        get_writer().add_scalar(f'Loss/train_{log_name}', loss.item(), batch_idx)
        optimizer.step()
        optimizer.zero_grad()

def valid_all_by_csv(model: Model, split = 'partial_valid'):
    csv = read_csv_dataset(split=split)
    results = []
    gen_commandss = []
    cheat_commandss = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc=f"Validating {split} games"):
        predict_result = model.predict(row['description'], row['generated_open_go_cmds'])
        filtered_cmds = predict_result.filtered_cmds
        cheat_commands = row['cheat_open_go_cmds']
        result = set(filtered_cmds) == set(cheat_commands) # 注意顺序
        if not result:
            logger.debug(f'{row_idx}: Filtered commands {filtered_cmds} not equal to cheat commands {cheat_commands}.')
            beutiful_print_command_and_probs(predict_result.cmds, predict_result.probs, log_func=logger.debug)
            logger.debug(f'^^^')
        results.append(result)
        gen_commandss.append(filtered_cmds)
        cheat_commandss.append(cheat_commands)
        # dbg(f'Valid results,  {result.score} / {result.max_score}, steps {result.step}, game {game_path}')
    extra = {'gen_commandss': gen_commandss, 'cheat_commandss': cheat_commandss, 'results': results}
    score = sum(results) / len(results)
    return score, extra

def valid_all_by_model_path(model_path: str):
    model = Model()
    model.load_checkpoint(model_path)
    return valid_all(model)


def get_model(checkpoint_path = None, init_func = Model):
    model = init_func()
    model.prefix = 'roberta_command_filter'
    model.init_bert()
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    return model

def train_reapeat(repeat = 1, epoch = 3, batch_size = 8):
    for rp in range(repeat):
        model = get_model()
        model.cuda()
        model.prefix = f'roberta_command_filter_{rp}'
        for i in range(epoch):
            train(model, batch_size=batch_size, split=TRAIN_SPLIT, log_name=f'{rp}')
            score, extra = valid_all_by_csv(model, split=FULL_VALID_SPLIT)
            dbg(f'Full valid score ({rp}): {score}')
            # get_writer().add_scalar(f'Score/valid_rp{rp}', score, i)
            model.save_checkpoint(base_path = SAVE_DIR, epoch=i)

def test_trained():
    model_paths = [
        '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours_command_filter/roberta_command_filter_0_epoch_0.pth',
        '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours_command_filter/roberta_command_filter_0_epoch_1.pth'
    ]
    for model_path in model_paths:
        model = get_model(checkpoint_path=model_path)
        model.cuda()
        score, extra = valid_all_by_csv(model, split=FULL_VALID_SPLIT)
        logger.debug(f'Full valid score: {score}')