# 将token等级的二分换成recipe + command的形式
# 使用危险动作识别器来判断是否危险
"""
<Recipe> [MASK] <cmd0> [MASK] <cmd1> [MASK] <cmd2> [MASK] <cmd3>

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
from common_new import logging, beutiful_print_command_and_probs, COMMAND_LIST_SHUFFLE
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import os
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from dataset_create_taku import get_cv_games
from game_command_generate import Game_command_generate_bert_filter
import random
import re

GAME_INIT_FUNC = Game_command_generate_bert_filter

logger = logging.getLogger(__name__)
dbg = logger.debug

DANGER_LABEL = 'danger'
SAFE_LABEL = 'safe'
CUT_COMMAND_PREFIX = ['slice', 'chop', 'dice']
COOK_COMMAND_PREFIX = ['cook']
MAYBE_DANGER_COMMAND_PREFIX = CUT_COMMAND_PREFIX + COOK_COMMAND_PREFIX

BERTInput = recordclass('BERTInput', 'input_ids label_ids label_ids_raw mask_idxs mask_ids extra')
PredictResult = recordclass('PredictResult', 'command danger_prob safe_prob')

"""
# Use <s> to binary classify the command.
<s> Recipe </s> Command </s>
"""
MAX_TOKEN_LENGTH = 100

def is_vital_action(action):
    prefix = action.split()[0]
    if prefix in CUT_COMMAND_PREFIX or prefix in COOK_COMMAND_PREFIX:
        return True
    return False

# TESTED
def bert_input_command_filter(recipe_clean, command, padding = True, logging = False):
    SEP, CLS, PAD = special_tokens_dict().sep, special_tokens_dict().cls, special_tokens_dict().pad
    toker = default_tokenizer()
    upper_part = f'{CLS} Recipe: {recipe_clean}'
    down_part = f' {SEP} Command: {command} {SEP}'
    prompt = f'{upper_part}{down_part}'
    # prompt = f'{CLS} Recipe: {recipe_clean} {SEP} {command} {SEP}'
    upper_token_ids = toker.encode(upper_part, add_special_tokens=False) # list of numbers
    down_token_ids = toker.encode(down_part, add_special_tokens=False) # list of numbers
    token_ids = upper_token_ids + down_token_ids
    token_ids = np.array(token_ids, dtype=int)
    if logging:
        logger.debug(f'{prompt}')
    meaningful_length = len(token_ids)
    if meaningful_length > MAX_TOKEN_LENGTH:
        raise ValueError(f'The length of the tokenized input is {meaningful_length}, which exceeds the maximum length of {MAX_TOKEN_LENGTH}.')
    if padding:
        pad_id = toker.encode(PAD, add_special_tokens=False)[0] # BUG fixed
        token_ids = np.concatenate((token_ids, np.full(MAX_TOKEN_LENGTH - meaningful_length, pad_id, dtype=int)))
        mask_ids = np.concatenate((np.ones(meaningful_length, dtype=int), np.zeros(MAX_TOKEN_LENGTH - meaningful_length, dtype=int)))
        return BERTInput(input_ids=token_ids, mask_ids=mask_ids)
    else:
        return BERTInput(input_ids=token_ids)

# TESTED
def bert_input_command_filter_loss(recipe_clean, command, danger = True):
    """
    Filter the generated open/go commands based on the description.
    :param desc_clean: The cleaned description of the game state.
    :param gen_open_go_cmds: The generated open/go commands.
    :param cheat_open_go_cmds: The cheat open/go commands.
    """
    toker = default_tokenizer()
    bert_input = bert_input_command_filter(recipe_clean, command, padding=True)
    label_ids = np.full(len(bert_input.input_ids), -100) # -100 means ignore
    DANGER_LABEL_ID = toker.encode(DANGER_LABEL, add_special_tokens=False)[0]
    SAFE_LABEL_ID = toker.encode(SAFE_LABEL, add_special_tokens=False)[0]
    label_ids[0] = DANGER_LABEL_ID if danger else SAFE_LABEL_ID
    bert_input.label_ids = label_ids
    return bert_input


def test_bert_input_command_filter():
    recipe_clean = 'I am in the kitchen. There is a knife on the table.'
    commands = ['go north', 'go south', 'go west']
    target_commands = ['go north', 'go south']
    result = bert_input_command_filter_loss(recipe_clean, target_commands[0], danger=True)
    return result

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.bert = init_bert_ours()
        self.bert = None
        self.prefix = 'roberta_command_filter'
    def init_bert(self):
        if not self.bert:
            self.bert = init_bert_ours()
    def predict(self, recipe_clean, command, logging = False):
        bert_input = bert_input_command_filter(recipe_clean, command, padding=True, logging=logging)
        input_ids = torch.LongTensor([bert_input.input_ids]).to(DEVICE) # (1, 100)
        mask_ids = torch.LongTensor([bert_input.mask_ids]).to(DEVICE) # (1, 100)
        with torch.no_grad():
            logits = self.bert(input_ids=input_ids, attention_mask=mask_ids).logits # (1, 100, 30522)
        logits = logits[0, 0] # (30522)
        toker = default_tokenizer()
        DANGER_LABEL_ID = toker.encode(DANGER_LABEL, add_special_tokens=False)[0]
        SAFE_LABEL_ID = toker.encode(SAFE_LABEL, add_special_tokens=False)[0]
        logits = logits[[DANGER_LABEL_ID, SAFE_LABEL_ID]] # (2)
        assert logits.shape[0] == 2
        probs = torch.softmax(logits, dim=0).tolist() # (2)
        return PredictResult(command, probs[0], probs[1])
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
SAVE_DIR = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_theirs_danger_command_filter'

def read_csv_dataset(inputpath = 'good_dataset', split = 'fake_test', suffix = '_command_generate'):
    df= pd.read_csv(os.path.join(inputpath,
                        f'danger_action_filter_{split}{suffix}.csv'))
    df['danger_actions'] = df['danger_actions'].apply(eval)
    df['safe_actions'] = df['safe_actions'].apply(eval)
    return df

def data_analyse():
    """
    Analyse the dataset.
    """
    df = read_csv_dataset(split = 'train')
    print(f'The dataset has {len(df)} rows.')
    print(f'The dataset has {df["danger_actions"].apply(len).sum()} danger actions.')
    print(f'The dataset has {df["safe_actions"].apply(len).sum()} safe actions.')
    # print(f'The dataset has {df["recipe"].nunique()} unique recipes.')
    # print(f'The dataset has {df["recipe"].apply(lambda x: len(x.split())).mean()} average length of recipe.')

def get_game_name(gamepath):
    return os.path.split(gamepath)[-1]

def extract_walkthrough_dataset(split = 'fake_test', skip_bad_actions = True):
    from game_command_generate import game_state_from_game
    assert GAME_INIT_FUNC == Game_command_generate_bert_filter, 'We need to use Game_command_generate_bert_filter to generate the dataset.'
    datapath = '/home/taku/Downloads/cog2019_ftwp/games'
    train_games = get_cv_games(datapath, split)
    train_lines = []
    for game_path in tqdm(train_games):
        game = GAME_INIT_FUNC(game_path)
        game.reset()
        counter = 0
        safe_set = set()
        danger_set = set()
        recipe = ''
        for cmd in game.clean_walkthrough():
            if game.done:
                break
            game_state = game_state_from_game(game)
            # 过滤掉不好的动作
            admissible_commands = game.filtered_available_commands()
            if cmd.startswith('take'):
                cmd = re.sub(r'\sfrom.*$', '', cmd)
            if skip_bad_actions and cmd not in admissible_commands:
                if cmd == 'eat meal':
                    dbg(f'Eat meal not in admissible_commands, I will add it to make sure the game can done.')
                    admissible_commands.append(cmd)
                else:
                    logger.error(f'Command {cmd} not in {admissible_commands}, I will skip it.')
                    continue
            for action in admissible_commands: # 先全部加进danger_actions
                if is_vital_action(action):
                    danger_set.add(action)
                # prefix = action.split()[0]
                # if prefix in CUT_COMMAND_PREFIX or prefix in COOK_COMMAND_PREFIX:
                #     danger_set.add(action)
            prefix = cmd.split()[0] # 对于需要执行的动作，全部加进safe_actions
            if prefix in CUT_COMMAND_PREFIX or prefix in COOK_COMMAND_PREFIX:
                safe_set.add(cmd)
            recipe = game_state.recipe_clean()
            game.act(cmd)
            counter += 1
        assert game.done
        danger_set = danger_set - safe_set # 只保留危险动作
        if len(danger_set) + len(safe_set) == 0:
            continue
        else:
            train_line = {
                'recipe': recipe, 
                'danger_actions': list(danger_set),
                'safe_actions': list(safe_set)
            }
            train_lines.append(train_line)
    return pd.DataFrame(train_lines)

def create_csv_dataset(outputpath = 'good_dataset', suffix = '_command_generate'):
    for split in ['fake_test', 'train', 'valid', 'test']:
        df = extract_walkthrough_dataset(split)
        df.to_csv(os.path.join(outputpath,
                        f'danger_action_filter_{split}{suffix}.csv'), index=False)

def dataloader_raw(split = TRAIN_SPLIT):
    # dataloader
    csv = read_csv_dataset(split=split)
    csv = csv.sample(frac=1) # shuffle to train
    results = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        recipe = row['recipe']
        danger_actions = row['danger_actions']
        safe_actions = row['safe_actions']
        results.append((recipe, danger_actions, safe_actions))
    return results

@lru_cache(maxsize=1)
def dataloader_get(split = TRAIN_SPLIT, batch_size = 32):
    bert_inputs = []
    for recipe, danger_actions, safe_actions in dataloader_raw(split=split):
        for cmd in safe_actions:
            bert_inputs.append(bert_input_command_filter_loss(recipe, cmd, danger=False))
        for cmd in danger_actions:
            bert_inputs.append(bert_input_command_filter_loss(recipe, cmd, danger=True))
    random.shuffle(bert_inputs)
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

def train(model, batch_size = 32, split = TRAIN_SPLIT, log_name = ''):
    model.train()
    train_dataloader = dataloader_get(split=split, batch_size=batch_size)
    # training
    from accelerate import Accelerator
    accelerator = Accelerator()
    # model.cuda()
    dbg('Model train on.')
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) # 从1e-3到2e-5
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

def valid_all_by_csv(model: Model, split = 'valid'):
    model.eval()
    from sklearn.metrics import f1_score
    dataset = dataloader_raw(split=split)
    results = []
    predicts = []
    danger_accuracy = []
    safe_accuracy = []
    for row_idx, (recipe, danger_actions, safe_actions) in enumerate(tqdm(dataset, desc=f"Validating {split} games")):
        for cmd in danger_actions:
            results.append(1)
            predict_result = model.predict(recipe, cmd)
            predicts.append(1 if predict_result.danger_prob > 0.5 else 0) 
            danger_accuracy.append(1 if predict_result.danger_prob > 0.5 else 0)
            if predict_result.danger_prob <= 0.5:
                logger.debug(f'{cmd} is predicted as safe, but it is dangerous. {recipe} ')
        for cmd in safe_actions:
            results.append(0)
            predict_result = model.predict(recipe, cmd)
            predicts.append(1 if predict_result.danger_prob > 0.5 else 0)
            safe_accuracy.append(1 if predict_result.danger_prob <= 0.5 else 0)
            if predict_result.danger_prob > 0.5:
                logger.debug(f'{cmd} is predicted as dangerous, but it is safe. {recipe} ')
    # score = f1_score(results, predicts, average='binary', pos_label=1)
    return sum(danger_accuracy) / len(danger_accuracy), sum(safe_accuracy) / len(safe_accuracy)

def valid_all_by_csv_standard(model: Model, split = 'valid'):
    model.eval()
    from sklearn.metrics import f1_score, precision_score, recall_score
    dataset = dataloader_raw(split=split)
    results = []
    predicts = []
    for row_idx, (recipe, danger_actions, safe_actions) in enumerate(tqdm(dataset, desc=f"Validating {split} games")):
        for cmd in danger_actions:
            results.append(1)
            predict_result = model.predict(recipe, cmd)
            predicts.append(1 if predict_result.danger_prob > 0.5 else 0) 
        for cmd in safe_actions:
            results.append(0)
            predict_result = model.predict(recipe, cmd)
            predicts.append(1 if predict_result.danger_prob > 0.5 else 0)
    # score = f1_score(results, predicts, average='binary', pos_label=1)
    return f1_score(results, predicts, average='binary', pos_label=1), \
           precision_score(results, predicts, average='binary', pos_label=1), \
           recall_score(results, predicts, average='binary', pos_label=1)

def valid_all_by_model_path(model_path: str):
    # model = Model()
    # model.load_checkpoint(model_path)
    model = default_trained_model()
    # model.cuda()
    # return valid_all_by_csv(model, split=FULL_VALID_SPLIT)
    return valid_all_by_csv_standard(model, split=FULL_VALID_SPLIT)


def train_reapeat(repeat = 1, epoch = 3, batch_size = 32):
    for rp in range(repeat):
        model = get_model()
        model.cuda()
        model.prefix = f'roberta_command_filter_{rp}'
        for i in range(epoch):
            train(model, batch_size=batch_size, split='train', log_name=f'{rp}')
            score = valid_all_by_csv(model, split='valid')
            print(f'Full valid score ({rp}): {score}')
            logger.debug(f'Full valid score ({rp}): {score}')
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



def get_model(checkpoint_path = None, init_func = Model):
    model = init_func()
    model.prefix = 'roberta_command_filter'
    model.init_bert()
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    return model

# vvvvvvvvvvvvvvvvvvvvvvvvvv export vvvvvvvvvvvvvvvvvvvvvvvvvvv

@lru_cache(maxsize=1)
def default_trained_model():
    # epoch 1, valid score: {danger_accuracy: 0.904, safe_accuracy: 1.0}
    path = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_theirs_danger_command_filter/roberta_command_filter_0_epoch_1.pth'
    model = get_model(checkpoint_path=path)
    model.to(DEVICE)
    return model

@lru_cache(maxsize=284)
def use_bert_to_identify_danger_command(recipe, command, logging = False):
    model = default_trained_model()
    result = model.predict(recipe, command, logging=logging)
    return result.danger_prob > 0.5