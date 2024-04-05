from bot4 import Grid2
from bot4 import plot_world_state
from bot4 import Alien
from bot4 import bot4
from bot3 import bot3
import copy

def main():
    for k in range(2, 10, 2):
        bot1_success = []
        bot1_deaths, bot2_deaths = 0, 0
        bot2_success = []

        for i in range(1):
            g = Grid2(bot=4, debug=False)
            b1 = bot3(g, k=k, debug=False)
            bot_pos = b1.pos
            a1 = Alien(g._grid, bot_pos, k, None)
            g2 = copy.deepcopy(g)

            alien_pos = a1.ind
            a2 = Alien(g2._grid, bot_pos, k, alien_pos)
            print(f"alien 1 pos: {a1.ind}, alien 2 pos: {a2.ind}")
            turns = 0
            dead = False

            for _ in range(500):
                dead = b1.move()
                a1.move()
                turns += 1
                plot_world_state(g2, b1)
                if a1.ind == b1.pos:
                    dead = True

                if dead or b1.pos == g.crew_pos:
                    break
            
            if dead:
                # this means that the bot died
                dead = False
                bot1_deaths += 1
            else:
                bot1_success.append(turns)

            turns = 0
            b2 = bot4(g2, k=k, bot_pos=bot_pos, debug=False)
            b2.reset_beliefs()
            del a1
            
            for _ in range(500):
                dead, grid_coor = b2.move()
                a2.move()
                turns += 1
                plot_world_state(g2, b2)
                if a2.ind == b2.pos:
                    dead = True
                if dead or b2.pos == g2.crew_pos:
                    break
            
            if dead:
                # this means that the bot died
                dead = False
                bot2_deaths += 1
            else:
                bot2_success.append(turns)


        print(f"For k: {k}")
        print(f"Bot 1 success list: {bot1_success}")
        print(f"Bot 2 success list: {bot2_success}")
        print()

        bot1_success_rate = len(bot1_success) / 200
        bot2_success_rate = len(bot2_success) / 200

        bot1_avg = sum(bot1_success) / 200
        bot2_avg = sum(bot2_success) / 200

        print(f"The success rate of bot 1 is: {bot1_success_rate}")
        print(f"The success rate of bot 2 is: {bot2_success_rate}")
        print()

        print(f"Bot 1 died {bot1_deaths} times")
        print(f"Bot 2 died {bot2_deaths} times")
        print()

        print(f"The average number of steps taken by bot 1 are: {bot1_avg}")
        print(f"The average number of steps taken by bot 2 are: {bot2_avg}")
        print()
        print()

main()