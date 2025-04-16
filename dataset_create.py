import os
import pandas as pd
from tqdm import tqdm
import textworld.gym
from textworld import EnvInfos, gym
import common_new as common
from functools import lru_cache

TRAIN_PATH = '/home/taku/Downloads/cog2019_ftwp/games/train'
TEST_PATH = '/home/taku/Downloads/cog2019_ftwp/games/test'
VALID_PATH = '/home/taku/Downloads/cog2019_ftwp/games/valid'


DEBUG = False


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

requested_infos = EnvInfos(description=True, inventory=True,
                            admissible_commands=True, objective=False,
                            verbs=False, command_templates=False,
                            entities=True, max_score=True, won=True,
                            lost=True, extras=["recipe", "walkthrough"])


def get_cv_games(path, split = 'train'):
    return all_game_paths(f'{path}/{split}')


def get_game_name(gamepath):
    return os.path.split(gamepath)[-1]

def extract_walkthrough_dataset(games):
    """
    runs a sequence of games and collects all information into a dataframe
    """
    gamesteps = []
    for game in tqdm(games):
        gamesteps += run_game(game)
    return pd.DataFrame(gamesteps)

def extra_info_by_game_path(game_path):
    import json
    json_path = game_path.replace('.z8', '.json')
    f = open(json_path)
    data = json.load(f)
    f.close()
    return data['extras']

def run_game(gamefile):
    """ runs a game following the walkthrough and extracts the information """
    if DEBUG:
        print('Game name:', gamefile)
    env_id = textworld.gym.register_games([gamefile], requested_infos)
    env = gym.make(env_id)
    obs, infos = env.reset()
    walkthrough = common.filter_commands_default(infos['extra.walkthrough'])
    seen_cookbook = False
    gamename = get_game_name(gamefile)
    gamesteps = []
    idx = 0
    # taku
    taku_extra = extra_info_by_game_path(gamefile)
    taku_recipe = taku_extra['recipe']
    for cmd in walkthrough:
        if DEBUG:
            print('cmd:', cmd)
        step = {
            'gamename': gamename,
            'step': idx,
            'description': infos['description'],
            'inventory': infos['inventory'],
            'recipe': taku_recipe if seen_cookbook else '', # NOTE: 不知道为什么现在info取不到recipe，所以用taku_recipe来代替
            'admissible_commands': common.filter_commands_default(infos['admissible_commands']), # ?
            'entities': infos['entities'],
            # 'verbs': infos['verbs'],
            # 'command_templates': [], # taku: 不需要这个
            # 'objective': infos['objective'],
            'max_score': infos['max_score'],
            # 'has_won': infos['has_won'],
            'won': infos['won'],
            # 'has_lost': infos['has_lost'],
            'lost': infos['lost'],
            'walkthrough': walkthrough,
            'seen_cookbook': seen_cookbook,
            'command': cmd
        }
        obs, scores, dones, infos = env.step(cmd)
        step['reward'] = scores
        step['obs'] = obs
        if DEBUG:
            print('obs:', obs)
        gamesteps.append(step)
        if cmd == 'examine cookbook':
            seen_cookbook = True
        idx += 1
    assert dones # ?
    return gamesteps

def extract_datasets(datapath, outputpath, need_valid = True, need_train = True, suffix = ''):
    """ runs all games and saves dataframes with state/command """
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    if need_train:
        train_games = get_cv_games(datapath, 'train')
        train_data = extract_walkthrough_dataset(train_games)
        train_data.to_csv(os.path.join(outputpath,
                        f'walkthrough_train{suffix}.csv'), index=False)
    if need_valid:
        valid_games = get_cv_games(datapath, 'valid')
        valid_data = extract_walkthrough_dataset(valid_games)
        valid_data.to_csv(os.path.join(outputpath,
                        f'walkthrough_valid{suffix}.csv'), index=False)


# DONE: 2025.3.24
def generate_dataset():
    output = 'good_dataset'
    datapath = '/home/taku/Downloads/cog2019_ftwp/games'
    extract_datasets(datapath, output, need_valid=True, need_train=True)