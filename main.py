import pygame
import math
import argparse

from game.player.human_player import Human_Player

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "-m",
                    help="Select Between Train mode or Play mode. Play mode is default.",
                    nargs='?',
                    choices=['play', 'train', 'PLAY', 'TRAIN'],
                    default="play",
                    type=str)
parser.add_argument("--players", "-p",
                    help="Add a custom selection of player agents. \n Options: h=human-controlled, r=random-agent, q=dq-agent \n Format: [q,h,r] = 1 dq-agent, 1 human-controlled, 1 random. \n Maximum players is four. \n Default is [q,r,h]. ",
                    nargs='?',
                    default=['q', 'r', 'h'],
                    type=list[str])
parser.add_argument("--model",
                    help="Select a pretrained model to use.",
                    nargs='?',
                    default=None,
                    type=str)
args = parser.parse_args()

from game.constants import LR, WIDTH, HEIGHT, DEVICE
from game.game_manager import Game_Manager

from game.player.DQL_player.Agent import Agent
from game.player.DQL_player.Memory import Memory
from game.player.DQL_player.Model import CQNet
from game.player.DQL_player.Trainer import QTrainer
from game.player.random_player import Random_Player

PLOT = False
DISABLE_GRAPHICS = False
MAX_GAMES = math.inf
PRETRAINED_MODEL = args.model

if args.mode.upper() == "PLAY":
    PLOT = False
    DISABLE_GRAPHICS = False
    PRETRAINED_MODEL = 'model-34.11-final-version.pth'
elif args.mode.upper() == "TRAIN":
    PLOT = True
    DISABLE_GRAPHICS = True

main_net = CQNet()
target_net = CQNet()
main_net.to(DEVICE)
target_net.to(DEVICE)
trainer = QTrainer(main_net, target_net, lr=LR, gamma=0.95, pretrained_model=PRETRAINED_MODEL, plot=PLOT)
memory = Memory()

PLAYERS = [Agent(memory, trainer, args.mode.upper() == "TRAIN"), Random_Player(), Human_Player()]

if args.mode.upper() == "TRAIN":
    PLAYERS = [Agent(memory, trainer, True)]

if args.players != ['q', 'r', 'h']:
    PLAYERS = []
    for i in range(4):
        match args.players[i]:
            case 'q':
                PLAYERS.append(Agent(memory, trainer, args.mode.upper() == "TRAIN"))
            case 'r':
                PLAYERS.append(Random_Player())
            case 'h':
                if DISABLE_GRAPHICS == True:
                    print("Human Controlled Agents cannot be in games with Disabled Graphics!")
                    exit()
                else:
                    PLAYERS.append(Human_Player())
            case _:
                continue


WIN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Calico')


def main():

    running = True
    game = Game_Manager(WIN, PLAYERS, disable_graphics=DISABLE_GRAPHICS, plot=PLOT)
    n_games = 0
    
    while running:

        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

        if game.step(events):

            n_games += 1
            
            if PLOT: game.plotter.plot_scores(game.scores, n_games)

            if n_games >= MAX_GAMES:
                exit()

            game.restart_game()
    
    pygame.QUIT()
    quit()

main()
            
    