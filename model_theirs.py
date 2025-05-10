# TODO: dataset generation
from game_command_generate import Game_command_generate, Game_command_generate_bert_filter
from dataset_create_taku import row_to_game_state
from dataset_create_taku_command_generate import read_csv_dataset
import re
import pandas as pd
from tqdm import tqdm
from game import Game_state
from common_new import logging
from bert_utils import default_tokenizer, special_tokens_dict, EMPTY_RECIPE, MAX_TOKEN_SIZE, EMPTY_INVENTORY
from bert_utils import BertInput, command_indexs_tokenized, init_bert_ours, DEVICE
logger = logging.getLogger('model_theirs')
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch import nn, optim
from functools import lru_cache

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
    text += f"{recip_text} {game_state.description_clean()}"
    text_b = f" {SEP} {action} {SEP}"
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
    csv = read_csv_dataset(split)
    csv = csv.sample(frac=1)
    bert_inputs = []
    for row_idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Dataset processing"):
        state = row_to_game_state(row) # NOTE: 2025.5.5 打乱以提高模型的泛化能力
        for command in row['admissible_commands']:
            if command == row['action']: # positive
                bert_input = to_bert_input_theirs(state, command, positive=True, need_padding=True)
            else: # negative
                bert_input = to_bert_input_theirs(state, command, positive=False, need_padding=True)
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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.bert = init_bert_ours()
        self.bert = None
        self.prefix = 'roberta_ours'
    def init_bert(self):
        if not self.bert:
            self.bert = init_bert_ours()
    def predict(self, game_state:Game_state):
        raise NotImplementedError("predict function should be implemented in subclass")
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
        outputs = model.bert(input_ids=input_ids.to(DEVICE), 
                   attention_mask=input_mask.to(DEVICE), 
                   token_type_ids=token_type_ids.to(DEVICE),
                   labels=label_ids.to(DEVICE))
        loss = outputs.loss
        accelerator.backward(loss)
        writer.add_scalar(f'Loss/train_{log_name}', loss.item(), batch_idx)
        optimizer.step()
        optimizer.zero_grad()