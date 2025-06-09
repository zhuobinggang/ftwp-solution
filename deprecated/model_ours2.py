## 将训练方式改为从游戏环境中直接训练，以过渡到强化学习。
from model_ours import Model, Model_ucb1, get_model, optim, tqdm, get_writer, DEVICE, get_cv_games, test_game, np
from model_ours import Game_command_generate_bert_filter, re, TEST_SPLIT
from bert_utils import to_bert_input
import torch
import logging
logger = logging.getLogger('model_ours2')
GAME_INIT_FUNC = Game_command_generate_bert_filter
skip_bad_actions = True
SAVE_DIR = '/home/taku/Downloads/cog2019_ftwp/trained_models/roberta_ours_command_gen_with_filter2'
TRAIN_SPLIT, FULL_VALID_SPLIT, TEST_SPLIT = 'train', 'valid', 'test'
import random

LOSS_AUGMENT = False # 2025.5.4 如果开启，train函数中的loss会根据动作的反馈进行强化

def loss_scalar(command, reward):
    if reward > 0: # 直接有回报的行动
        return 2
    if command.startswith('take'): # 没有回报，但是拿起物品，有意义
        return 1.5
    if command.startswith('open'): # 没有回报，但是打开容器，有意义
        return 1.5
    return 1


def train(model, batch_size = 8, split = TRAIN_SPLIT, log_name = '', train_proportion = 1.0):
    writer = get_writer()
    assert not isinstance(model, Model_ucb1), 'Model should be instance of Model, but not Model_ucb1'
    from game_command_generate import game_state_from_game
    train_games = get_cv_games(split = split)
    train_games = train_games[:int(len(train_games) * train_proportion)]
    random.shuffle(train_games) # 随机打乱训练集
    # training
    from accelerate import Accelerator
    accelerator = Accelerator()
    # model.cuda()
    model.train()
    logger.debug('Model train on.')
    optimizer = optim.AdamW(model.parameters(), lr=2e-5) # 从1e-3到2e-5
    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    counter = 0
    batch_idx = 0
    input_token_lengths = []
    for game_path in tqdm(train_games):
        game = GAME_INIT_FUNC(game_path)
        game.reset()
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
                    logger.debug(f'Eat meal not in admissible_commands, I will add it to make sure the game can done.')
                    admissible_commands.append(cmd)
                else:
                    logger.debug(f'Command {cmd} not in {admissible_commands}, I will skip it.')
                    continue
            obs, reward, done, info = game.act(cmd)
            counter += 1
            # TODO: 计算loss
            action_idx = admissible_commands.index(cmd)
            bert_input = to_bert_input(game_state, action_idx, need_padding=False)
            input_token_lengths.append(len(bert_input.input_ids))
            outputs = model.bert(
                    input_ids=torch.tensor([bert_input.input_ids], dtype=torch.long).to(DEVICE), 
                    attention_mask=torch.tensor([bert_input.attention_mask], dtype=torch.long).to(DEVICE),
                    labels=torch.tensor([bert_input.labels], dtype=torch.long).to(DEVICE)
                )
            loss = outputs.loss
            if LOSS_AUGMENT:
                loss = loss * loss_scalar(cmd, reward)
            accelerator.backward(loss)
            writer.add_scalar(f'Loss/train_{log_name}', loss.item(), batch_idx)
            if counter % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_idx += 1
        assert game.done
    if counter % batch_size != 0:
        optimizer.step()
        optimizer.zero_grad()
    logger.error(f'Trained! Batch size: {batch_size}, average input token length: {np.mean(input_token_lengths)}, max input token length: {np.max(input_token_lengths)}')


def valid_all(model, split = 'partial_valid', valid_proportion = 1.0):
    game_init_func = Game_command_generate_bert_filter
    game_paths = get_cv_games(split=split)
    game_paths = game_paths[:int(len(game_paths) * valid_proportion)]
    score = 0
    max_score = 0
    steps = []
    logger.debug(f'Validating {split} games, total {len(game_paths)}')
    for game_path in tqdm(game_paths, desc=f"Validating {split} games"):
        game = game_init_func(game_path)
        result = test_game(game, model)
        score += result.score
        max_score += result.max_score
        steps.append(result.step)
        # dbg(f'Valid results,  {result.score} / {result.max_score}, steps {result.step}, game {game_path}')
    average_step = np.mean(steps)
    norm_score = score / max_score
    if isinstance(model, Model_ucb1):
        print(f'UCB1: score ({split}), average step {average_step}')
        logger.error(f'UCB1: score ({split}): {norm_score} with UCB1, average step {average_step}')
    else:
        print(f'W/O UCB1: score ({split}): {norm_score}, average step {average_step}')
        logger.error(f'W/O UCB1: score ({split}): {norm_score}, average step {average_step}')
    return norm_score, average_step


BEST_MODELS = [0, 0, 0]

def train_repeat(repeat = 3, epoch = 8, batch_size = 8):
    global BEST_MODELS
    MODEL_INIT_FUNC = Model
    assert MODEL_INIT_FUNC == Model # 这里是为了避免在训练时使用ucb1
    # TRAIN_SPLIT = 'fake_test'
    # FULL_VALID_SPLIT = 'fake_test'
    for rp in range(repeat):
        model = get_model(init_func=MODEL_INIT_FUNC)
        model.prefix = f'roberta_ours_repeat_{rp}'
        max_score = 0
        for i in range(epoch):
            train(model, batch_size=batch_size, split=TRAIN_SPLIT, log_name=f'{rp}') # 不会使用predict
            score, avg_step = valid_all(model, split=FULL_VALID_SPLIT) # 会使用predict
            # get_writer().add_scalar(f'Score/valid_rp{rp}', score, i)
            if score > max_score:
                max_score = score
                BEST_MODELS[rp] = i
                logger.error(f'BEST MODEL RECORDED at E {i}. BEST MODELS: {BEST_MODELS}')
                # get_writer().add_scalar(f'Score/best_valid_rp{rp}', score, i)
            model.save_checkpoint(base_path = SAVE_DIR, epoch=i)

def test_trained(repeat = 3):
    MODEL_INIT_FUNC = Model_ucb1
    assert MODEL_INIT_FUNC == Model_ucb1
    logger.error('vvvvv\nTesting trained models')
    logger.error(f'Best models: {BEST_MODELS}')
    # FULL_VALID_SPLIT = 'fake_test'
    # TEST_SPLIT = 'fake_test'
    for rp in range(repeat):
        path = f'{SAVE_DIR}/roberta_ours_repeat_{rp}_epoch_{BEST_MODELS[rp]}.pth'
        model = get_model(path, init_func=MODEL_INIT_FUNC)
        s1, avg_step = valid_all(model, split=FULL_VALID_SPLIT, game_init_func=Game_command_generate_bert_filter)
        print(f'Full valid score ({rp}): {s1} with UCB1, average step {avg_step}')
        s2, avg_step = valid_all(model, split=TEST_SPLIT, game_init_func=Game_command_generate_bert_filter)
        print(f'Full test score ({rp}): {s2} with UCB1, average step {avg_step}')


def night_run():
    train_repeat(repeat=3, epoch=8, batch_size=8)
    test_trained(repeat=3)