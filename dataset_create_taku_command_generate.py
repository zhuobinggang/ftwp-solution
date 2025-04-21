from dataset_create_taku import *
from game_command_generate import Game_command_generate
import re

def extract_walkthrough_dataset(split = 'fake_test', skip_bad_actions = True):
    datapath = BASE_PATH
    train_games = get_cv_games(datapath, split)
    gamesteps = []
    for game_path in tqdm(train_games):
        game = Game_command_generate(game_path)
        game.reset()
        counter = 0
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


def create_csv_dataset(outputpath = 'good_dataset', suffix = '_command_generate'):
    for split in ['fake_test', 'train', 'valid', 'test']:
        df = extract_walkthrough_dataset(split)
        df.to_csv(os.path.join(outputpath,
                        f'walkthrough_{split}{suffix}.csv'), index=False)