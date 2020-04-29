import sys
import time
import pygame
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt  # for graphing our mean rewards over time

from matplotlib import style
from pygame.locals import *
size = width, height = 10, 10

white = 255, 255, 255
black = 0, 0, 0
blue = 9, 67, 60
red = 233, 32, 32
green = 20, 255, 20
randcolor = np.random.randint(0, 255), np.random.randint(
    0, 255), np.random.randint(0, 255)

HM_EPISODES = 2
MOVE_PENALTY = 50
ENEMY_PENALTY = 300
FOOD_REWARD = 200  # vielleicht z.B. wenn der Rand getroffen wird
PLACE_ALL_REWARD = 2000

epsilon = 0.99
EPS_DECAY = 0.9

SHOW_EVERY = 1  # what episode to show

LEARNING_RATE = 0.001
DISCOUNT = 0.95


rectanglelist = []


class Frame:
    def __init__(self, x, y, width, height):

        self.x = x
        self.y = y
        self.w = width
        self.h = height

    def __str__(self):
        return f"{self.x}, {self.y}, {self.w}, {self.h}"

    def __sub__(self, other):
        return (self.x, self.y, self.w, self.h)

    def position(self):
        return self.rect

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)  # move dia down
        elif choice == 1:
            self.move(x=-1, y=-1)  # move dia up
        elif choice == 2:
            self.move(x=-1, y=1)  # move dia down left
        elif choice == 3:
            self.move(x=1, y=-1)
        """if choice == 0:
            self.move(x=1, y=0)  # move right
        elif choice == 1:
            self.move(x=-1, y=0)  # move left
        elif choice == 2:
            self.move(x=0, y=1)  # move down
        elif choice == 3:
            self.move(x=1, y=-1)  # move up"""

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif (self.x + self.w) > size[0]:
            self.x = size[0] - self.w

        if self.y < 0:
            self.y = 0
        elif (self.y + self.h) > size[1]:
            self.y = size[1] - self.h


# or filename or none"qtable-1580289897.pickle"
start_q_table = None  # "qtable-1234.pickle"
if start_q_table is None:
    q_table = {}
    for x1 in range(-size[0]+1, size[1]):
        for y1 in range(-size[0]+1, size[1]):
            for w in range(-size[0]+1, size[1]):
                for h in range(-size[0]+1, size[1]):
                        # every possible combination of state/action pair
                    q_table[(x1, y1, w, h)] = [np.random.uniform(-5, 0)
                                               for i in range(4)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


numberofframes = 4
#fremes = [[-1, -1, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5], [0, 0, 5, 5]]
fr1 = 1, 1
fr2 = 2, 2
fr3 = 7, 8
fr4 = 2, 2

framestotal = []
episode_rewards = []
for episode in range(HM_EPISODES):
    fremes = [[-1, -1, np.random.randint(4, 5), np.random.randint(4, 5)], [0, 0, np.random.randint(4, 5), np.random.randint(
        4, 6)], [0, 0, np.random.randint(4, 5), np.random.randint(4, 5)], [0, 0, np.random.randint(3, 6), np.random.randint(4, 5)]]
    ok = []
    if episode % SHOW_EVERY == 0 and episode != 1322233:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(
            f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    if episode % SHOW_EVERY == 10000:
        print(1)
    framestotal.clear()
    rectanglelist.clear()
    for frameindex in range(len(fremes)):

        framestotal.append(Frame(
            fremes[frameindex][0], fremes[frameindex][1], fremes[frameindex][2], fremes[frameindex][3]))

        # print("Frame in framestotal: " + str(framestotal[frameindex]))

        episode_reward = 0
        for i in range(50):
            obs = (framestotal[frameindex].x, framestotal[frameindex].y,
                   framestotal[frameindex].w, framestotal[frameindex].h)

            if np.random.random() > epsilon:
                action = np.argmax(q_table[obs])
            else:
                action = np.random.randint(0, 4)

            framestotal[frameindex].action(action)

            if pygame.Rect(
                    framestotal[frameindex].x, framestotal[frameindex].y, framestotal[frameindex].w, framestotal[frameindex].h).collidelist(rectanglelist) == -1:
                reward = FOOD_REWARD

            elif pygame.Rect(
                    framestotal[frameindex].x, framestotal[frameindex].y, framestotal[frameindex].w, framestotal[frameindex].h).collidelist(rectanglelist) != -1 and frameindex == len(fremes)-1:
                reward = -ENEMY_PENALTY
            else:
                reward = -MOVE_PENALTY

            new_obs = (framestotal[frameindex].x, framestotal[frameindex].y,
                       framestotal[frameindex].w, framestotal[frameindex].h)
            max_future_q = np.max(q_table[new_obs])
            current_q = q_table[obs][action]

            if reward == FOOD_REWARD:
                new_q = FOOD_REWARD

            else:
                new_q = (1 - LEARNING_RATE) * current_q + \
                    LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[obs][action] = new_q

            if show:
                print("New Observation after action: " +
                      str(obs) + " at Frame " + str(frameindex))

                print("Overlap? " +
                      str(pygame.Rect(
                          framestotal[frameindex].x, framestotal[frameindex].y, framestotal[frameindex].w, framestotal[frameindex].h, ).collidelist(rectanglelist)))

                pygame.init()
                screen = pygame.display.set_mode((1000, 1000))

                screen.fill(white)
                surface = pygame.Surface((size[0], size[1]))
                surface.fill(white)

                try:
                    for lol in range(len(fremes)):
                        pygame.draw.rect(surface, (np.random.randint(0, 255), np.random.randint(
                            0, 255), np.random.randint(0, 255)), pygame.Rect(
                            framestotal[lol].x, framestotal[lol].y, framestotal[lol].w, framestotal[lol].h))

                except:
                    pass

                screensurface = pygame.Surface((1000, 1000))

                pygame.transform.scale(
                    surface,
                    # surface to be scaled
                    (1000, 1000),  # scale up to (width, height)

                    screensurface)  # surface that game_surface will be scaled onto

                screen.blit(screensurface, (0, 0))
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.display.update()
                        pygame.quit()
                        sys.exit(0)

                # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if reward == FOOD_REWARD:
                    if cv2.waitKey(2000) & 0xFF == ord('q'):
                        pygame.display.update()
                        pygame.quit()
                        sys.exit(0)
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            episode_reward += reward

            if reward == FOOD_REWARD:
                rectanglelist.append(pygame.Rect(
                    framestotal[frameindex].x, framestotal[frameindex].y, framestotal[frameindex].w, framestotal[frameindex].h))
                #print("\n Success")
                break

        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones(
    (SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()


with open(f"qtable-1234.pickle", "wb") as f:
    pickle.dump(q_table, f)
