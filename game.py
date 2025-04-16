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
    
def default_game():
    return Game_handle_recipe('/home/taku/Downloads/cog2019_ftwp/games/valid/tw-cooking-recipe1+cook+cut+drop+go6-M2qEFeOXcol3H1ql.ulx')


def test_game(game: Game_handle_recipe, model = Fake_model()):
    # dbg('Testing: Model eval on, model cuda on.')
    if model.training:
        model.eval()
        dbg('Model eval on.')
    if not next(model.parameters()).is_cuda:
        model.cuda()
        dbg('Model cuda on.')
    obs, info = game.reset()
    counter = 0
    while counter < 100:
        counter += 1
        action = model.predict(game_state_from_game(game))
        obs, reward, done, info = game.act(action)
        if done:
            break
    # result = (counter, info['score'], info['max_score'], info)
    result = TestResult(counter, info['score'], info['max_score'], info)
    return result

@lru_cache(maxsize=128) # 一个episode最多为100步，因此128足够了
def clean_action_obs(action, obs):
    ACT, OBS = action, obs
    if action == 'examine cookbook' and common.is_recipe_feedback(obs):
        OBS = 'recipe got!'
    elif common.is_description_feedback(obs):
        room_name = common.extract_room_name(obs)
        OBS = f'you entered {room_name}.'
    OBS = ' '.join(OBS.split()).strip()
    return ACT, OBS

class Game_state:
    def __init__(self):
        self.room = ''
        self.description_raw = ''
        self.recipe_raw = ''
        self.inventory_raw = ''
        self.action_obs_pairs = []
        self.admissible_commands = []
        self.filtered_commands = None
    def recipe_clean(self):
        if self.recipe_raw == '':
            return ''
        return common.extract_recipe(self.recipe_raw, need_clean=True)
    def inventory_clean(self):
        if self.inventory_raw == '':
            return ''
        return common.handle_inventory_text(self.inventory_raw)
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

def game_state_from_game(game: Game_handle_recipe):
    state = Game_state()
    state.room = common.extract_room_name(game.info['description'])
    state.description_raw = game.info['description']
    state.recipe_raw = game.recipe_raw
    state.inventory_raw = game.info['inventory']
    state.action_obs_pairs = game.action_obs_pairs
    state.admissible_commands = game.info['admissible_commands']
    return state



def test():
    from bert_utils import bert_prompt_from_game_state
    game = default_game()
    _ = game.reset()
    game.act('go east')
    game.act('examine cookbook')
    print(bert_prompt_from_game_state(game_state_from_game(game)))

# =========== create csv dataset ==============