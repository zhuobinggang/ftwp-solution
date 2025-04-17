from game import Game_handle_recipe, game_state_from_game
from dataset_create_taku import get_cv_games
import common_new as common
import logging
import re

logger = logging.getLogger('game_command_generate')

kitchenware = ['oven', 'stove', 'BBQ']
COOK_COMMAND_RESTRICT = True # True的情况，如果库存和食谱中有相同的物品，才会生成cook命令

class Game_command_generate(Game_handle_recipe):
    def filtered_available_commands(self):
        # NOTE: 需要使用tiny bert和规则来生成命令集
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate cook commands')
            return []
        game_state = game_state_from_game(self)
        desc = game_state.description_clean()
        exist_kitchenware = []
        for item in kitchenware:
            # BUG: Match whole words only, considering possible non-word boundaries like: 'east' in 'chicken breast'
            # if re.search(rf'(?<!\w){re.escape(item)}(?!\w)', desc):
            if common.whole_word_inside(item, desc):
                # NOTE: 这里需要使用tiny bert和规则来生成命令集
                exist_kitchenware.append(item)
                # logger.debug(f'Kitchenware {item} found in description')
        if len(exist_kitchenware) == 0:
            # logger.debug('No kitchenware found in description, no need to generate cook commands')
            return []
        cook_commands = []
        ingredients = game_state.ingredients_from_recipe()
        inventory = game_state.inventory_clean()
        entities = self.info['entities']
        ingredients = ingredients + ' meal ' # BUG: 为了匹配'cook meal'，这里加上了'meal'
        for entity in entities:
            # if entity in ingredients:
            if common.whole_word_inside(entity, inventory):
                if COOK_COMMAND_RESTRICT and not common.whole_word_inside(entity, ingredients):
                    continue
                else:
                    for ware in exist_kitchenware:
                        cook_commands.append(f'cook {entity} with {ware}')
        return cook_commands


def check_walkthrough_cook_command_generate():
    # 1. 遍历valid set生成game
    # 2. 使用过滤的walkthrough将game跑一遍
    # 3. 每一个step，如果command == 'cook'，确认我们的生成方法能够生成对应的指令
    valid_paths = get_cv_games(split='valid')
    for game_path in valid_paths:
        check_one_game_full_admissible_cook_command(game_path)

def check_one_game(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    entities = info['entities']
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        if cmd.startswith('cook'):
            # NOTE: 这里需要使用tiny bert和规则来生成命令集
            our_commands = game.filtered_available_commands()
            logger.debug(f'cmd: {cmd} our_commands: {our_commands}')
            assert cmd in our_commands, f"Command {cmd} not in admissible commands {our_commands}"
        game.act(cmd)

def check_one_game_full_admissible_cook_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    entities = info['entities']
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        game_state = game_state_from_game(game)
        admissible_commands = game_state.filtered_available_commands()
        cook_commands = [cmd for cmd in admissible_commands if cmd.startswith('cook')]
        if len(cook_commands) == 0 or game_state.recipe_clean() == '':
            pass
        else:
            our_commands = game.filtered_available_commands()
            logger.debug(f'cook_commands: {cook_commands} our_commands: {our_commands}')
            for cook_cmd in cook_commands:
                assert cook_cmd in our_commands, f"Command {cook_cmd} not in admissible commands {our_commands}"
        game.act(cmd)