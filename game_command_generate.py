from game import Game_handle_recipe

class Game_command_generate(Game_handle_recipe):
    def filtered_available_commands(self):
        # NOTE: 需要使用tiny bert和规则来生成命令集
        pass