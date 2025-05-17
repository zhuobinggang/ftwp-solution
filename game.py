import common_new as common
import textworld.gym
from textworld import EnvInfos, gym
from functools import lru_cache
from recordclass import recordclass
import logging

logger = logging.getLogger('game.py')
dbg = logger.debug

MAX_STEP = 100

# 重新实现game
def init_env(game_file):
    requested_infos = EnvInfos(description=True, inventory=True,
                               admissible_commands=True, objective=False,
                               # verbs=True, command_templates=True,
                               entities=True, max_score=True, won=True, score=True,
                               moves = True,
                               lost=True, extras=["walkthrough"]) # 注意，取不到recipe，只能从obs中获取
    env_id = textworld.gym.register_games([game_file], requested_infos, max_episode_steps = MAX_STEP)
    env = gym.make(env_id)
    return env

TestResult = recordclass('TestResult', 'step score max_score info')

class Game:
    def __init__(self, game_path):
        self.game_path = game_path
        self.env = init_env(game_path)
        self.obs, self.info = None, None
        self.reward, self.done = 0, False
    def reset(self):
        self.obs, self.info = self.env.reset()
        return self.obs, self.info
    def act(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        return self.obs, self.reward, self.done, self.info
    def clean_walkthrough(self):
        return common.filter_commands_default(self.info['extra.walkthrough'])


class Fake_model:
    def __init__(self):
        self.counter = 0
    def predict(self, obs, info):
        # 这里需要根据obs和info来选择动作
        # 这里简单返回一个随机动作
        action = info['extra.walkthrough'][self.counter]
        self.counter += 1
        if self.counter >= len(info['extra.walkthrough']):
            self.counter = 0
        return action
    def eval(self):
        pass
    def cuda(self):
        pass

# ============

class Game_with_history(Game):
    def __init__(self, game_path):
        super().__init__(game_path)
        self.action_obs_pairs = []
    def act(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        self.action_obs_pairs.append((action, self.obs))
        return self.obs, self.reward, self.done, self.info
    
class Game_handle_recipe(Game_with_history):
    def __init__(self, game_path):
        super().__init__(game_path)
        self.recipe_raw = ''
        self.recipe = ''
        self.obs_raw = ''
    def act(self, action):
        self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
        obs = self.obs_raw
        # obs simplify
        if action == 'examine cookbook' and common.is_recipe_feedback(obs):
            self.recipe_raw = common.extract_recipe(obs, need_clean=False)
            self.recipe = common.extract_recipe(self.recipe_raw, need_clean=True)
        self.action_obs_pairs.append((action, obs)) # 不在这里处理，而是放到game_state中处理
        self.obs = obs
        return self.obs, self.reward, self.done, self.info
    def to_game_state(self):
        return game_state_from_game(self)
    def get_admissible_commands(self):
        return common.filter_commands_default(self.info['admissible_commands'])
    
    
# NOTE: 在移动命令被执行后，obs改为prev_room to current_room。这样能够给模型一个直观的记忆。因为在prompt中没有上一个房间的信息，应该很有帮助。
class Game_move_action_augment(Game_handle_recipe):
    def act(self, action):
        # 这里处理移动命令
        if action.startswith('go'):
            prev_room = common.extract_room_name(self.info['description'])
            self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
            current_room = common.extract_room_name(self.obs_raw)
            obs = f'From {prev_room} to {current_room}.'
        else:
            self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
            obs = self.obs_raw
        # obs simplify
        if action == 'examine cookbook' and common.is_recipe_feedback(obs):
            self.recipe_raw = common.extract_recipe(obs, need_clean=False)
            self.recipe = common.extract_recipe(self.recipe_raw, need_clean=True)
        self.action_obs_pairs.append((action, obs)) # 不在这里处理，而是放到game_state中处理
        self.obs = obs
        return self.obs, self.reward, self.done, self.info
    

class Game_handle_worldmap(Game_move_action_augment):
    def __init__(self, game_path):
        super().__init__(game_path)
        self.worldMap = {}
        self.itemMap = {}
    def reset(self):
        self.obs, self.info = self.env.reset()
        if True: # 初始化第一个房间
            room_name = common.extract_room_name(self.info['description'])
            self.worldMap[room_name] = {}
        return self.obs, self.info
    def act(self, action):
        # 这里处理移动命令
        if action.startswith('go'):
            prev_room = common.extract_room_name(self.info['description'])
            self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
            current_room = common.extract_room_name(self.info['description'])
            obs = f'From {prev_room} to {current_room}.'
            if True: # 更新worldMap 
                if prev_room not in self.worldMap:
                    self.worldMap[prev_room] = {}
                if current_room not in self.worldMap:
                    self.worldMap[current_room] = {}
                direction = action.split()[1]
                op_direction = common.get_opposite_direction(direction)
                self.worldMap[prev_room][direction] = current_room
                self.worldMap[current_room][op_direction] = prev_room
        else:
            self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
            obs = self.obs_raw
        # obs simplify
        if action == 'examine cookbook' and common.is_recipe_feedback(obs):
            self.recipe_raw = common.extract_recipe(obs, need_clean=False)
            self.recipe = common.extract_recipe(self.recipe_raw, need_clean=True)
        self.action_obs_pairs.append((action, obs)) # 不在这里处理，而是放到game_state中处理
        self.obs = obs
        if True: # NOTE: 每一步根据环境描述来更新itemMap
            # 每一步根据recipe & 环境描述来更新itemList。item包含字段：room。
            entities = self.info['entities']
            for entity in entities:
                if common.whole_word_inside(entity, self.info['description']):
                    if entity not in self.itemMap:
                        self.itemMap[entity] = {'room': ''}
                    room_name = common.extract_room_name(self.info['description'])
                    if room_name != self.itemMap[entity]['room']:
                        # logger.debug(f'Update itemMap: {entity} from {self.itemMap[entity]["room"]} to {room_name}')
                        pass
                    self.itemMap[entity]['room'] = room_name
                if common.whole_word_inside(entity, self.info['inventory']):
                    if entity not in self.itemMap:
                        self.itemMap[entity] = {'room': ''}
                    self.itemMap[entity]['room'] = 'inventory'
        return self.obs, self.reward, self.done, self.info

    
def default_game():
    return Game_handle_worldmap('/home/taku/Downloads/cog2019_ftwp/games/valid/tw-cooking-recipe1+cook+cut+drop+go6-M2qEFeOXcol3H1ql.ulx')


def test_game(game: Game_move_action_augment, model = Fake_model(), max_step = 100):
    # dbg('Testing: Model eval on, model cuda on.')
    if model.training:
        model.eval()
        dbg('Model eval on.')
    if not next(model.parameters()).is_cuda:
        model.cuda()
        dbg('Model cuda on.')
    obs, info = game.reset()
    counter = 0
    final_action = ''
    while counter < max_step:
        counter += 1
        game_state = game_state_from_game(game)
        game_state.admissible_commands = game.get_admissible_commands() # BUG: 如果在game_state_from_game中调用这个会导致无限循环
        action = model.predict(game_state)
        obs, reward, done, info = game.act(action)
        final_action = action
        if done:
            break
    # result = (counter, info['score'], info['max_score'], info)
    dbg(f'Game done: {info["score"]} / {info["max_score"]}, steps {counter}, won: {info["won"]}, lost: {info["lost"]}, path: {game.game_path}')
    if info['lost']:
        from model_danger_command import use_bert_to_identify_danger_command
        is_danger = use_bert_to_identify_danger_command(game_state.recipe_clean(), final_action, logging=True)
        logger.warning(f'Game lost: final action: {final_action}, is_danger: {is_danger}')
    result = TestResult(counter, info['score'], info['max_score'], info)
    return result

@lru_cache(maxsize=128) # 一个episode最多为100步，因此128足够了
def clean_action_obs(action, obs):
    ACT, OBS = action, obs
    if action == 'examine cookbook' and common.is_recipe_feedback(obs):
        OBS = 'recipe got!'
    elif common.is_description_feedback(obs): # NOTE: 如果使用Game_move_action_augment的话，obs会是"From room1 to room2."，不会进入这个分支
        room_name = common.extract_room_name(obs)
        OBS = f'you entered {room_name}.'
    OBS = ' '.join(OBS.split()).strip()
    return ACT, OBS

class Game_state:
    def __init__(self):
        self.room = ''
        self.description_raw = ''
        self.recipe_raw = ''
        self.recipe_clean_cache = ''
        self.inventory_raw = ''
        self.action_obs_pairs = []
        self.admissible_commands = []
        self.filtered_commands = None
        self.worldMap = {}
    def recipe_clean(self):
        if self.recipe_raw == '':
            return ''
        if self.recipe_clean_cache != '':
            return self.recipe_clean_cache
        else:
            self.recipe_clean_cache = common.extract_recipe(self.recipe_raw, need_clean=True)
        return self.recipe_clean_cache
    def inventory_clean(self):
        if self.inventory_raw == '':
            return ''
        return common.handle_inventory_text(self.inventory_raw)
    def ingredients_from_recipe(self):
        return common.ingredients_from_recipe(self.recipe_clean())
    def description_clean(self):
        return common.description_simplify(self.description_raw)
    def clean_action_obs_pairs(self):
        return [clean_action_obs(action, obs) for action, obs in self.action_obs_pairs]
    def action_history(self, history_window = 100, seperator='>', no_action_text=''):
        action_obs_pairs = self.clean_action_obs_pairs()
        action_history_text = common.action_obs_pairs_to_history(action_obs_pairs, seperator=seperator, no_action_text=no_action_text, history_window = history_window)        
        return action_history_text
    def filtered_available_commands(self):
        if self.filtered_commands is not None:
            return self.filtered_commands
        self.filtered_commands = common.filter_commands_default(self.admissible_commands)
        return self.filtered_commands
    def available_commands_text(self):
        return common.actions_to_list_number(self.filtered_available_commands())
    def __str__(self):
        return f'Game_state(room={self.room}, description={self.description_clean()}, recipe={self.recipe_clean()}, inventory={self.inventory_clean()}, action_obs_pairs={self.action_history()}, admissible_commands={self.available_commands_text()})'


def game_state_from_game(game: Game_handle_worldmap, need_admissible_commands = True):
    state = Game_state()
    state.room = common.extract_room_name(game.info['description'])
    state.description_raw = game.info['description']
    state.recipe_raw = game.recipe_raw
    state.inventory_raw = game.info['inventory']
    state.action_obs_pairs = game.action_obs_pairs
    if need_admissible_commands:
        state.admissible_commands = game.get_admissible_commands() # NOTE: 4.21 Game将代理取得可能选项
    if hasattr(game, 'worldMap'):
        state.worldMap = game.worldMap
    return state



def test():
    from bert_utils import bert_prompt_from_game_state
    game = default_game()
    _ = game.reset()
    game.act('go east')
    game.act('go west')
    game.act('go east')
    game.act('examine cookbook')
    print(bert_prompt_from_game_state(game_state_from_game(game)))

# =========== create csv dataset ==============