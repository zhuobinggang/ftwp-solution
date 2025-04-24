import dataset_create_taku
from dataset_create_taku import BASE_PATH, get_cv_games, get_game_name, tqdm, dbg, logger, pd, os, common
from game_command_generate import Game_command_generate, Game_command_generate_bert_filter
import re
import random

GAME_INIT_FUNC = Game_command_generate_bert_filter

def extract_walkthrough_dataset(split = 'fake_test', skip_bad_actions = True):
    from game_command_generate import game_state_from_game
    assert GAME_INIT_FUNC == Game_command_generate_bert_filter, 'We need to use Game_command_generate_bert_filter to generate the dataset.'
    datapath = BASE_PATH
    train_games = get_cv_games(datapath, split)
    gamesteps = []
    for game_path in tqdm(train_games):
        game = GAME_INIT_FUNC(game_path)
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
        

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv generate open & go commands filter dataset vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

def extract_open_go_command_filter_dataset(split = 'fake_test', skip_bad_actions = True):
    """
    - 对于每一个step, 获得所有open和close的指令列表。
    - 对于每个指令, 我们设定50%的几率调用。生成desc, open, close, go的训练用案例。
    - 复原状态。
    - 对于一个游戏, 单个房间只生成一个训练用案例且只有在open, close指令数量大于1的时候生成。
    """
    from game import game_state_from_game
    datapath = BASE_PATH
    train_games = get_cv_games(datapath, split)
    gamesteps = []
    for game_path in tqdm(train_games):
        game = Game_command_generate(game_path)
        obs, info = game.reset()
        logger.debug(f'New Game: {game_path}\nWTS: {game.clean_walkthrough()}\n')
        act_counter = 0
        visited_rooms = set()
        for walkthrough_command in game.clean_walkthrough():
            if game.done:
                break
            game_state = game_state_from_game(game)
            if game_state.room in visited_rooms:
                pass
            else: # 准备生成训练用案例
                visited_rooms.add(game_state.room)
                reversed_cmds = []
                # 遍历，随机执行开关操作
                available_open_close_cmds = [cmd for cmd in game.info['admissible_commands'] if cmd.startswith('open') or cmd.startswith('close')]
                if len(available_open_close_cmds) < 2:
                    logger.debug(f'Available open/close commands < 2, I will skip it.')
                    pass
                else:
                    for cmd in available_open_close_cmds:
                        # 50%几率执行
                        if random.random() < 0.5:
                            obs, reward, done, info = game.act(cmd)
                            act_counter += 1
                            reversed_cmd = cmd.replace('open', 'close') if cmd.startswith('open') else cmd.replace('close', 'open')
                            reversed_cmds.append(reversed_cmd)
                    updated_cheat_open_go_cmds = [cmd for cmd in game.info['admissible_commands'] if cmd.startswith('open') or cmd.startswith('go')]
                    # 基于当前状态生成指令
                    gen_open_close_go_cmds =  [cmd for cmd in game.generate_admissible_commands() if cmd.startswith('open') or cmd.startswith('go')]
                    for cmd in updated_cheat_open_go_cmds:
                        assert cmd in gen_open_close_go_cmds, f'Cheat command {cmd} not in generated commands {gen_open_close_go_cmds}\n {game_state.description_clean()}'
                    game_state = game_state_from_game(game)
                    game_step = {
                        'game_path': get_game_name(game_path),
                        'description': game_state.description_clean(),
                        'cheat_open_go_cmds': updated_cheat_open_go_cmds,
                        'generated_open_go_cmds': gen_open_close_go_cmds,
                    }
                    logger.debug(f'Saved! description: {game_step["description"]}\nopen_go_cmds: {updated_cheat_open_go_cmds}\n')
                    gamesteps.append(game_step)
                    # 复原状态
                    for cmd in reversed_cmds:
                        obs, reward, done, info = game.act(cmd)
                        act_counter += 1
                    game_state = game_state_from_game(game)
                    updated_cheat_open_go_cmds = [cmd for cmd in game.info['admissible_commands'] if cmd.startswith('open') or cmd.startswith('go')]
                    logger.debug(f'Recoverd! description: {game_state.description_clean()}\nopen_go_cmds: {updated_cheat_open_go_cmds}\n')
            admissible_commands = common.filter_commands_default(game.info['admissible_commands'])
            # 执行指令
            if act_counter > 98:
                logger.error(f'Act counter > 98, I will skip it.')
                break
            # if walkthrough_command.startswith('take'):
            #     walkthrough_command = re.sub(r'\sfrom.*$', '', walkthrough_command)
            if walkthrough_command == 'eat meal':
                admissible_commands.append(walkthrough_command)
            if skip_bad_actions and walkthrough_command not in admissible_commands:
                logger.error(f'Command {walkthrough_command} not in {admissible_commands}, I will skip it.\nDescription: {game_state.description_clean()}\n')
            else:
                obs, reward, done, info = game.act(walkthrough_command)
                logger.debug(f'EXE Command {walkthrough_command}\nOBS: {obs}\n')
                act_counter += 1
        assert (game.done or act_counter > 98), f'Game done {game.done}, act_counter = {act_counter}'
    return pd.DataFrame(gamesteps)

def create_csv_dataset_open_go_filter(outputpath = 'good_dataset', suffix = '_command_generate'):
    for split in ['fake_test', 'train', 'valid', 'test']:
    # for split in ['fake_test']:
        df = extract_open_go_command_filter_dataset(split)
        df.to_csv(os.path.join(outputpath,
                        f'open_go_filter_{split}{suffix}.csv'), index=False)
        