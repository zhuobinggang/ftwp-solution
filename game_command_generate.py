from game import Game_move_action_augment
from dataset_create_taku import get_cv_games
import common_new as common
import logging
import re
from tqdm import tqdm
from fasttext_classifier import is_openable_entity
from bert_command_filter import use_bert_to_filter_command, PredictResult
from game import Game_state
from bert_utils import bert_prompt_from_game_state

class Fake_logger:
    def debug(self, msg):
        pass
# logger = logging.getLogger('game_command_generate')
logger = Fake_logger()

KITCHENWARES = ['oven', 'stove', 'BBQ']
COOK_COMMAND_RESTRICT = True # True的情况，如果库存和食谱中有相同的物品，才会生成cook命令
CUT_COMMANDS = ['slice', 'chop', 'dice']




class Game_command_generate(Game_move_action_augment):
    
    def get_admissible_commands(self):
        return self.filtered_available_commands()
    
    def generate_admissible_commands(self):
        return self.filtered_available_commands()

    def filtered_available_commands(self):
        cook_commands = self.cook_command_generate()
        knife_commands = self.knife_command_generate()
        drop_commands = self.drop_command_generate()
        eat_commands = self.eat_command_generate()
        take_commands = self.take_command_generate()
        open_commands = self.open_command_generate()
        prepare_meal_commands = self.prepare_meal_command_generate()
        go_commands = self.go_command_generate()
        examine_cookbook = self.examine_cookbook_command_generate()
        all_commands = cook_commands + knife_commands + take_commands + drop_commands + \
            open_commands + go_commands + prepare_meal_commands + eat_commands + examine_cookbook
        return all_commands
    
    def filter_enetities_in_ingredients(self, candidate_entities = None):
        if self.recipe == '':
            logger.debug('No recipe found in game state')
            return []
        results = []
        game_state = game_state_from_game(self)
        ingredients = game_state.ingredients_from_recipe()
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, ingredients):
                results.append(entity)
        return results
    
    def filter_enetities_in_inventory(self, candidate_entities = None):
        entities = []
        game_state = game_state_from_game(self)
        inventory = game_state.inventory_clean()
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, inventory):
                entities.append(entity)
        return entities
    
    def entities_in_description(self, candidate_entities = None, use_raw_description = False):
        entities = []
        game_state = game_state_from_game(self)
        desc = game_state.description_clean() if not use_raw_description else self.info['description']
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, desc):
                entities.append(entity)
        return entities
    
    def kitchenware_in_description(self):
        exist_kitchenware = self.entities_in_description(candidate_entities=KITCHENWARES)
        return exist_kitchenware
    
    def drop_command_generate(self):
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        if len(entities_in_inventory) == 0:
            logger.debug('No entities found in inventory, no need to generate drop commands')
            return []
        drop_commands = []
        for entity in entities_in_inventory:
            drop_commands.append(f'drop {entity}')
        return drop_commands

    def knife_command_generate(self):
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate cut commands')
            return []
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        if 'knife' not in entities_in_inventory:
            logger.debug('No knife found in inventory, no need to generate cut commands')
            return []
        foods_in_inventory = [entity for entity in entities_in_inventory if entity != 'knife']
        if COOK_COMMAND_RESTRICT:
            foods_in_inventory = self.filter_enetities_in_ingredients(foods_in_inventory)
        retults = []
        for food in foods_in_inventory:
            for cut_command in CUT_COMMANDS:
                retults.append(f'{cut_command} {food} with knife')
        return retults

    def cook_command_generate(self):
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate cook commands')
            return []
        exist_kitchenware = self.kitchenware_in_description()
        if len(exist_kitchenware) == 0:
            # logger.debug('No kitchenware found in description, no need to generate cook commands')
            return []
        cook_commands = []
        entities = self.filter_enetities_in_inventory(self.info['entities'])
        if COOK_COMMAND_RESTRICT:
            entities = self.filter_enetities_in_ingredients(entities)
        for entity in entities:
            for ware in exist_kitchenware:
                cook_commands.append(f'cook {entity} with {ware}')
        return cook_commands
    
    def eat_command_generate(self):
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate eat commands')
            return []
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        if 'meal' in entities_in_inventory:
            return ['eat meal']
        else:
            return []
    
    def take_command_generate(self):
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate take commands')
            return []
        entities_to_take = self.info['entities']
        if COOK_COMMAND_RESTRICT:
            entities_to_take = self.filter_enetities_in_ingredients(self.info['entities'])
            entities_to_take = entities_to_take + ['knife']
        # 判断环境中是否有这些物品
        entities_to_take = self.entities_in_description(candidate_entities=entities_to_take)
        return [f'take {entity}' for entity in entities_to_take]
    
    def open_command_generate(self):
        # 判断环境中是否有这些物品
        # 清除frosted-glass door中的横杠
        entities = self.entities_in_description(candidate_entities=self.info['entities'], use_raw_description=True)
        entities_to_open = []
        for entity in entities:
            if entity.endswith('door') or is_openable_entity(entity):
                entities_to_open.append(entity)
        return [f'open {entity}' for entity in entities_to_open]
    
    def prepare_meal_command_generate(self):
        # TODO: 判断，当身处厨房，有食谱，且背包里的物品和食谱中的物品相同，才会生成prepare meal命令
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate prepare meal commands')
            return []
        game_state = game_state_from_game(self)
        room = game_state.room
        if room.lower() != 'kitchen':
            logger.debug('Not in kitchen, no need to generate prepare meal commands')
            return []
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        entities_in_recipe = self.filter_enetities_in_ingredients(self.info['entities'])
        for entity in entities_in_recipe:
            if entity not in entities_in_inventory:
                logger.debug(f'Entity {entity} not in inventory, no need to generate prepare meal commands')
                return []
        return ['prepare meal']
    
    def go_command_generate(self):
        directions = ['north', 'south', 'east', 'west']
        filtered_directions = self.entities_in_description(candidate_entities=directions)
        if len(filtered_directions) == 0:
            logger.debug('No directions found in description, no need to generate go commands')
            return []
        go_commands = ['go ' + direction for direction in filtered_directions]
        return go_commands
    
    def examine_cookbook_command_generate(self):
        if self.recipe != '':
            return []
        entities_in_description = self.entities_in_description(candidate_entities=self.info['entities'])
        if 'cookbook' not in entities_in_description:
            logger.debug('No cookbook found in description, no need to generate examine cookbook commands')
            return []
        return ['examine cookbook']
    

class Game_command_generate_bert_filter(Game_command_generate):
    def filtered_available_commands(self):
        cook_commands = self.cook_command_generate()
        knife_commands = self.knife_command_generate()
        drop_commands = self.drop_command_generate()
        eat_commands = self.eat_command_generate()
        take_commands = self.take_command_generate()
        open_commands = self.open_command_generate()
        prepare_meal_commands = self.prepare_meal_command_generate()
        go_commands = self.go_command_generate()
        examine_cookbook = self.examine_cookbook_command_generate()
        # NOTE: Use bert to filter open & go commands
        open_go_commands = open_commands + go_commands
        open_go_commands = self.use_bert_filter(common.description_simplify(self.info['description']), open_go_commands)
        all_commands = cook_commands + knife_commands + take_commands + drop_commands + \
            open_go_commands + prepare_meal_commands + eat_commands + examine_cookbook
        return all_commands
    def use_bert_filter(self, desc_clean, open_go_commands):
        predict_result = use_bert_to_filter_command(desc_clean, open_go_commands)
        return predict_result.filtered_cmds
    def dd(self, action = None):
        if action is None:
            self.reset()
        else:
            self.act(action)
        print(self.info['description'])
        print_prompt(self)
    
    
def default_game():
    return Game_command_generate_bert_filter('/home/taku/Downloads/cog2019_ftwp/games/valid/tw-cooking-recipe1+cook+cut+drop+go6-M2qEFeOXcol3H1ql.ulx')

def print_prompt(game: Game_command_generate_bert_filter):
    game_state_lack = game_state_from_game(game)
    def fake_func():
        return game.filtered_available_commands()
    game_state_lack.filtered_available_commands = fake_func
    print(bert_prompt_from_game_state(game_state_lack))


def game_state_from_game(game: Game_command_generate):
    state = Game_state()
    state.room = common.extract_room_name(game.info['description'])
    state.description_raw = game.info['description']
    state.recipe_raw = game.recipe_raw
    state.inventory_raw = game.info['inventory']
    state.action_obs_pairs = game.action_obs_pairs
    # state.admissible_commands = game.get_admissible_commands() # NOTE: 4.21 Game将代理取得可能选项
    return state

# ============================================================


def check_walkthrough_cook_command_generate():
    # 1. 遍历valid set生成game
    # 2. 使用过滤的walkthrough将game跑一遍
    # 3. 每一个step，如果command == 'cook'，确认我们的生成方法能够生成对应的指令
    valid_paths = get_cv_games(split='valid')
    for game_path in valid_paths:
        # check_one_game_full_admissible_cut_command(game_path)
        # check_one_game(game_path)
        # check_one_game_full_admissible_drop_command(game_path)
        # check_one_game_full_admissible_eat_command(game_path)
        # check_one_game_full_admissible_take_command(game_path)
        # check_one_game_full_admissible_open_command(game_path)
        # check_one_game_full_admissible_prepare_meal_command(game_path)
        # check_one_game_walkthrough_prepare_meal_command(game_path)
        # check_one_game_full_admissible_go_command(game_path)
        check_one_game_walkthrough_command(game_path) # NOTE: TEST DONE 2025.4.21

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
            assert cmd in our_commands, f"Command {cmd} not in admissible commands {our_commands}\ndescription: {info['description']}"
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
            our_commands = game.cook_command_generate()
            logger.debug(f'cook_commands: {cook_commands} our_commands: {our_commands}')
            for cook_cmd in cook_commands:
                assert cook_cmd in our_commands, f"Command {cook_cmd} not in admissible commands {our_commands}"
        game.act(cmd)

def check_one_game_full_admissible_cut_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        game_state = game_state_from_game(game)
        admissible_commands = game_state.filtered_available_commands()
        cut_commands = [cmd for cmd in admissible_commands if cmd.split()[0] in CUT_COMMANDS]
        if len(cut_commands) == 0 or game_state.recipe_clean() == '':
            pass
        else:
            our_commands = game.knife_command_generate()
            logger.debug(f'cut_commands: {cut_commands} our_commands: {our_commands}')
            for cook_cmd in cut_commands:
                assert cook_cmd in our_commands, f"Command {cook_cmd} not in admissible commands {our_commands}"
        game.act(cmd)

def check_one_game_full_admissible_drop_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        game_state = game_state_from_game(game)
        admissible_commands = game_state.filtered_available_commands()
        drop_cmds = [cmd for cmd in admissible_commands if cmd.startswith('drop')]
        if len(drop_cmds) == 0:
            pass
        else:
            our_commands = game.drop_command_generate()
            logger.debug(f'cut_commands: {drop_cmds} our_commands: {our_commands}')
            for cook_cmd in drop_cmds:
                assert cook_cmd in our_commands, f"Command {cook_cmd} not in admissible commands {our_commands}"
        game.act(cmd)

def check_one_game_full_admissible_eat_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        game_state = game_state_from_game(game)
        admissible_commands = game_state.filtered_available_commands()
        eat_cmds = [cmd for cmd in admissible_commands if cmd.startswith('eat')]
        if len(eat_cmds) == 0:
            pass
        else:
            our_commands = game.eat_command_generate()
            logger.debug(f'eat_cmds: {eat_cmds} our_commands: {our_commands}')
            for cook_cmd in eat_cmds:
                assert cook_cmd in our_commands, f"Command {cook_cmd} not in admissible commands {our_commands}"
        game.act(cmd)


def check_one_game_full_admissible_take_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        game_state = game_state_from_game(game)
        admissible_commands = game_state.filtered_available_commands()
        take_cmds = [cmd for cmd in admissible_commands if cmd.startswith('take')]
        if len(take_cmds) == 0 or game_state.recipe_clean() == '':
            pass
        else:
            our_commands = game.take_command_generate()
            logger.debug(f'take_cmds: {take_cmds} our_commands: {our_commands}')
            for cmd in take_cmds:
                cmd = re.sub(r'\sfrom.*$', '', cmd)
                assert cmd in our_commands, f"Command {cmd} not in admissible commands {our_commands}\nentities: {game.info['entities']}\ndescription: {game.info['description']}"
        game.act(cmd)


def check_one_game_full_admissible_open_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    entities = info['entities']
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        game_state = game_state_from_game(game)
        admissible_commands = game_state.filtered_available_commands()
        open_commands = [cmd for cmd in admissible_commands if cmd.startswith('open')]
        if len(open_commands) == 0:
            pass
        else:
            our_commands = game.open_command_generate()
            logger.debug(f'open_commands: {open_commands} our_commands: {our_commands}')
            for cook_cmd in open_commands:
                # assert cook_cmd in our_commands, f"Command {cook_cmd} not in admissible commands {our_commands}\nentities: {game.info['entities']}\ndescription: {game_state.description_clean()}"
                if cook_cmd not in our_commands:
                    print(f"Command {cook_cmd} not in admissible commands {our_commands}\nentities: {game.info['entities']}\ndescription: {game_state.description_clean()}\n\n")
        game.act(cmd)



def check_one_game_full_admissible_prepare_meal_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    entities = info['entities']
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        game_state = game_state_from_game(game)
        admissible_commands = game_state.filtered_available_commands()
        CMD = 'prepare meal'
        if CMD in admissible_commands:
            our_commands = game.prepare_meal_command_generate()
            logger.debug(f'prepare meal: our_commands: {our_commands}')
            assert CMD in our_commands, f"Command {CMD} not in admissible commands {our_commands}\nroom: {game_state.room}\nrecipe: {game_state.recipe_clean()}\ninventory: {game_state.inventory_clean()}\n"
        game.act(cmd)

def check_one_game_walkthrough_prepare_meal_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    entities = info['entities']
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        if cmd == 'prepare meal':
            game_state = game_state_from_game(game)
            our_commands = game.prepare_meal_command_generate()
            logger.debug(f'["prepare meal"]: our_commands: {our_commands}')
            assert cmd in our_commands, f"Command {cmd} not in admissible commands {our_commands}\nroom: {game_state.room}\nrecipe: {game_state.recipe_clean()}\ninventory: {game_state.inventory_clean()}\n"
        game.act(cmd)

def check_one_game_full_admissible_go_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    entities = info['entities']
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        game_state = game_state_from_game(game)
        admissible_commands = game_state.filtered_available_commands()
        go_commands = [cmd for cmd in admissible_commands if cmd.startswith('go')]
        if len(go_commands) == 0:
            pass
        else:
            our_commands = game.go_command_generate()
            logger.debug(f'go_commands: {go_commands} our_commands: {our_commands}')
            for cook_cmd in go_commands:
                # assert cook_cmd in our_commands, f"Command {cook_cmd} not in admissible commands {our_commands}\nentities: {game.info['entities']}\ndescription: {game_state.description_clean()}"
                assert cook_cmd in our_commands, f"Command {cook_cmd} not in admissible commands {our_commands}\nentities: {game.info['entities']}\ndescription: {game_state.description_clean()}\n\n"
        game.act(cmd)


# NOTE: TEST DONE 2025.4.21
def check_one_game_walkthrough_command(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    entities = info['entities']
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        if cmd.startswith('take'):
            cmd = re.sub(r'\sfrom.*$', '', cmd)
        game_state = game_state_from_game(game)
        admissible_commands = game.filtered_available_commands()
        assert cmd in admissible_commands, f"Command {cmd} not in admissible commands {admissible_commands}\nentities: {game.info['entities']}\ndescription: {game_state.description_clean()}"
        game.act(cmd)


# vvvvvvvvvvvvvvvvvvvvvvvv 遍历所有游戏，然后提取openable和unopenable的物品 vvvvvvvvvvvvvvvvvvvvvv

def all_openable_entities():
    valid_paths = get_cv_games(split='train')
    openable_entities = set()
    unopenable_entities = set()
    for game_path in tqdm(valid_paths):
        openable, unopenable = filter_openable_entities(game_path)
        openable_entities.update(openable)
        unopenable_entities.update(unopenable)
    unopenable_entities_final = unopenable_entities - openable_entities
    return openable_entities, unopenable_entities_final


def filter_openable_entities(game_path):
    game = Game_command_generate(game_path)
    obs, info = game.reset()
    all_entities = set(info['entities'])
    openable_entities = set()
    clean_walkthrough = game.clean_walkthrough()
    for cmd in clean_walkthrough:
        game_state = game_state_from_game(game)
        admissible_commands = game_state.filtered_available_commands()
        for command in admissible_commands:
            if command.startswith('open'):
                entity = command.replace('open ', '')
                openable_entities.add(entity)
        game.act(cmd)
    unopenable_entities = all_entities - openable_entities
    return openable_entities, unopenable_entities


