import os
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from game import Game_move_action_augment, game_state_from_game, Game_state
import common_new as common
from common_new import COMMAND_LIST_SHUFFLE
import logging
logger = logging.getLogger('dataset_create_taku')
dbg = logger.debug
import random

BASE_PATH = '/home/taku/Downloads/cog2019_ftwp/games'
TRAIN_PATH = '/home/taku/Downloads/cog2019_ftwp/games/train'
TEST_PATH = '/home/taku/Downloads/cog2019_ftwp/games/test'
VALID_PATH = '/home/taku/Downloads/cog2019_ftwp/games/valid'




@lru_cache(maxsize=None)
def all_game_paths(test_path = TEST_PATH):
    import os
    results = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(test_path):
        if filename.endswith('.z8'):  # 只处理 .z8 文件
            file_path = os.path.join(test_path, filename)
            results.append(file_path)
    return results

def get_cv_games(path = BASE_PATH, split = 'fake_test'):
    return all_game_paths(f'{path}/{split}')


def get_game_name(gamepath):
    return os.path.split(gamepath)[-1]


def extract_walkthrough_dataset(split = 'fake_test', skip_bad_actions = True):
    datapath = BASE_PATH
    train_games = get_cv_games(datapath, split)
    gamesteps = []
    for game_path in tqdm(train_games):
        game = Game_move_action_augment(game_path)
        game.reset()
        counter = 0
        for cmd in game.clean_walkthrough():
            if game.done:
                break
            game_state = game_state_from_game(game)
            # 过滤掉不好的动作
            admissible_commands = game_state.filtered_available_commands()
            if skip_bad_actions and cmd not in admissible_commands:
                if cmd == 'eat meal':
                    dbg(f'Eat meal not in admissible_commands, I will add it to make sure the game can done.')
                    admissible_commands.append(cmd)
                else:
                    dbg(f'Command {cmd} not in admissible_commands, I will skip it.')
                    continue
            game_step = {
                'game_path': get_game_name(game_path),
                'room': game_state.room,
                'step': counter,
                'action': cmd,
                'action_obs_pairs': game_state.clean_action_obs_pairs(),
                'recipe': game_state.recipe_clean(),
                'inventory': game_state.inventory_clean(),
                'admissible_commands': admissible_commands,
                'description': game_state.description_clean(),
                'won': game.info['won'],
                'lost': game.info['lost'],
                'score': game.info['score'],
                # 'entities': game.info['entities'],
                'max_score': game.info['max_score'],
            }
            gamesteps.append(game_step)
            game.act(cmd)
            counter += 1
        assert game.done
    return pd.DataFrame(gamesteps)

def create_csv_dataset(outputpath = 'good_dataset', suffix = ''):
    for split in ['train', 'valid', 'test', 'fake_test']:
        df = extract_walkthrough_dataset(split)
        df.to_csv(os.path.join(outputpath,
                        f'walkthrough_{split}{suffix}.csv'), index=False)
        
def read_csv_dataset(inputpath = 'good_dataset', split = 'fake_test', suffix = ''):
    df= pd.read_csv(os.path.join(inputpath,
                        f'walkthrough_{split}{suffix}.csv'))
    df['action_obs_pairs'] = df['action_obs_pairs'].apply(eval)
    df['admissible_commands'] = df['admissible_commands'].apply(eval)
    df['recipe'] = df['recipe'].fillna('')
    df['inventory'] = df['inventory'].fillna('')
    return df


class Game_state_clean(Game_state):
    def __init__(self):
        super().__init__()
        self.recipe_good = ''
        self.inventory_good = ''
        self.description_good = ''
        self.action_obs_pairs_good = []
        self.available_commands_good = []
    def recipe_clean(self):
        return self.recipe_good
    def inventory_clean(self):
        return self.inventory_good
    def description_clean(self):
        return self.description_good
    def clean_action_obs_pairs(self):
        return self.action_obs_pairs_good
    def filtered_available_commands(self):
        return self.available_commands_good


def row_to_game_state(row):
    game_state = Game_state_clean()
    game_state.room = row['room']
    game_state.action_obs_pairs_good = row['action_obs_pairs']
    game_state.recipe_good = row['recipe']
    game_state.inventory_good = row['inventory']
    game_state.available_commands_good = row['admissible_commands']
    if COMMAND_LIST_SHUFFLE:
        random.shuffle(game_state.available_commands_good)
    game_state.description_good = row['description']
    return game_state

# DONE: 2025.4.10
def check_forward_backward_eual():
    csv = read_csv_dataset(split='fake_test')
    game_paths = get_cv_games(split='fake_test')
    idx = 0
    for group_name, df_group in csv.groupby('game_path', sort=False):
        print(f'{group_name} \n {get_game_name(game_paths[idx])}')
        game = Game_move_action_augment(game_paths[idx])
        game.reset()
        real_state = game.to_game_state()
        for row_index, row in df_group.iterrows():
            row_state = row_to_game_state(row)
            assert row_state.room == real_state.room, f"{row_index} {row_state.room} != {real_state.room}"
            assert row_state.recipe_clean() == real_state.recipe_clean(), f"{row_index} {row_state.recipe_clean()} != {real_state.recipe_clean()}"
            # NOTE: 因为inventory_clean实际上将set转换成string，所以顺序可能会改变，需要重新转回set再比较
            row_state_inventory_good = set([item.strip() for item in row_state.inventory_clean().split(',')])
            real_state_inventory_good = set([item.strip() for item in real_state.inventory_clean().split(',')])
            assert row_state_inventory_good == real_state_inventory_good, f"{row_index} {row_state_inventory_good} != {real_state_inventory_good}"
            # assert row_state.inventory_clean() == real_state.inventory_clean(), f"{row_index} {row_state.inventory_clean()} != {real_state.inventory_clean()}"
            assert row_state.action_history() == real_state.action_history(), f"{row_index} {row_state.action_history()} != {real_state.action_history()}"
            assert row_state.available_commands_text() == real_state.available_commands_text(), f"{row_index} {row_state.available_commands_text()} != {real_state.available_commands_text()}"
            game.act(row['action'])
            real_state = game.to_game_state()
        idx += 1