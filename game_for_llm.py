# 2025.7.7 手动复制输出到chatgpt聊天窗口
import common_new as common
from game_command_generate import Game_command_generate_bert_filter, game_state_from_game
from bert_utils import bert_prompt_from_game_state
import pyperclip
from functools import lru_cache

def print_prompt(game: Game_command_generate_bert_filter):
    game_state_lack = game_state_from_game(game)
    def fake_func():
        return game.filtered_available_commands()
    game_state_lack.filtered_available_commands = fake_func
    return bert_prompt_from_game_state(game_state_lack, need_action_history=False, seperater = '\n')

class Game_for_llm(Game_command_generate_bert_filter):
    def __call__(self, action = None):
        self.dd(action)
    def reset(self):
        self.obs, self.info = self.env.reset()
        if True: # 初始化第一个房间
            room_name = common.extract_room_name(self.info['description'])
            self.worldMap[room_name] = {}
        # return self.obs, self.info
    def dd(self, action = None): # 需要考虑的情况： examine cookbook -> 返回清理过的cookbook内容；
        texts_to_print = ''
        if action is None:
            texts_to_print += 'Task: Find and examine the cookbook, follow the instructions to prepare the meal, and eat it.'
            self.reset()
        else:
            texts_to_print += f'Executed action {self.info["moves"] + 1}: {action}'
            self.act(action)
            if self.obs.startswith("You're carrying too many things"):
                texts_to_print += '\n' + "You're carrying too many things already!"
        # print(self.info['description'])
        if self.reward > 0:
            texts_to_print += f'\nReward received: {self.reward}'
        # texts_to_print += f'\nCurrent description: {self.info["description"]}'
        texts_to_print += ('\n' + print_prompt(self))
        texts_to_print += f'\nNext action (answer with the action directly):'
        print(texts_to_print)
        # TODO: 复制到剪贴板
        pyperclip.copy(texts_to_print)


def default_game_for_llm():
    game = Game_for_llm('/home/taku/Downloads/cog2019_ftwp/games/valid/tw-cooking-recipe1+cook+cut+drop+go6-M2qEFeOXcol3H1ql.ulx')
    game.reset()
    return game

@lru_cache(maxsize=None)
def random_games_for_test(n = 10):
    from dataset_create_taku import get_cv_games
    game_paths = get_cv_games(split = 'test')
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(game_paths)
    return game_paths[: n]

def game_n(index = 0):
    game_paths = random_games_for_test(100)
    game = Game_for_llm(game_paths[index])
    game()
    return game