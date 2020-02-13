import pygame
import time
import random
import numpy as np
from math import sqrt

from feed_forward_neural_network import *


pygame.init()
pygame.display.init()
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
tangerine = (255, 153, 102)
 
dis_width = 200
dis_height = 200

dis_height_complete = dis_height + 30
 
dis = pygame.display.set_mode((dis_width, dis_height_complete))
pygame.display.set_caption('Snake Game by Edureka')
 
clock = pygame.time.Clock()
 
snake_block = 10
snake_speed = 15
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
 
#Score display 
def Your_score(score):
    value = score_font.render(str(score), True, black)
    dis.blit(value, [0, dis_height-10])
 
 
#Draws the snake 
def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])
 
#makes the input layer for neural network
def make_array(snake_List, foodx, foody):
    snake_Head = list(map(int, snake_List[-1]))
    input_layer = []

    #record if there is any danger to the left, right, above, or below the snake
    #can only view 1 block away from head
    for x in range(snake_Head[0] - 10, snake_Head[0] + 20, 10):
        for y in range(snake_Head[1] - 10, snake_Head[1] + 20, 10):
            if [x,y] == snake_Head:
                continue
            elif x != snake_Head[0] and y != snake_Head[1]:
                continue
            elif [x, y] in snake_List or x < 0 or x >= dis_width or \
            y < 0 or y >= dis_height:
                input_layer.append(0)
            else:
                input_layer.append(1)

    #sees where the apple is relative to snakes head in the x coordinate
    if snake_Head[0] > foodx:
        input_layer.append(-1)
    elif snake_Head[0] < foodx:
        input_layer.append(1)
    else:
        input_layer.append(0)

    #sees where the apple is relative to snakes head in the y coordinate
    if snake_Head[1] > foody:
        input_layer.append(-1)
    elif snake_Head[1] < foody:
        input_layer.append(1)
    else:
        input_layer.append(0)
    
    return input_layer
                    
#checks to see if the snake turned to the opposite direction from where it
#was going.
def opposite_direction(prev_direction, current_direction):
    if (prev_direction == "left" and current_direction == "right") or \
       (prev_direction == "right" and current_direction == "left"):
           return True
    if (prev_direction == "down" and current_direction == "up") or \
       (prev_direction == "up" and current_direction == "down"):
           return True
    return False

def run_game_with_ML(weights):
    game_close = False
    steps_taken = 0

    
    #start the snake in the middle of the board
    x1 = int(dis_width / 2)
    y1 = int(dis_height / 2)
 
    x1_change = 0
    y1_change = 0
    directions = ["left", "right", "up", "down"]
    
    snake_List = [[x1, y1]]
    Length_of_snake = 1

    #place the apple at a random location
    foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

    prev_direction = ""
    count_same_direction = 0

    
    score1 = 0 #penalized for every step taken or if the snake dies
    steps_per_game = 200
    score2 = 0 #rewarded if the snake move in the same direction
    while not game_close:
        input_layer = make_array(snake_List, foodx, foody)

        #calculate and take the best action based upon our neural network
        action = np.argmax(np.array(forward_propagation(np.array(input_layer).reshape(-1, 6), weights)))
        if directions[action]== "left":
            x1_change = -snake_block
            y1_change = 0
        elif directions[action]== "right":
            x1_change = snake_block
            y1_change = 0
        elif directions[action]== "up":
            y1_change = -snake_block
            x1_change = 0
        elif directions[action]== "down":
            y1_change = snake_block
            x1_change = 0

        
        #the snake hit the boundry
        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            score1 -= 400
            game_close = True

        #update the board if the snake is still alive
        x1_old = x1
        y1_old = y1
        x1 += x1_change
        y1 += y1_change
        dis.fill(blue, rect=[0, 0, dis_width, dis_height])
        dis.fill(tangerine, rect=[0, dis_height, dis_width, dis_height_complete - dis_height])
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)

        #if the snake collided with its own body
        for x in snake_List[:-1]:
            if x == snake_Head:
                score1 -= 400
                game_close = True
        score1 -= 1  #lose 1 point for every step taken

        #delete the tail of the snake after we update the snakes head location
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        #make sure the snakes movements are consistent 
        if (count_same_direction > 4 and prev_direction != directions[action]) or \
            opposite_direction(prev_direction, directions[action]):
            score2 -= 1
        else:
            score2 += 2
        
        if prev_direction == directions[action]:
            count_same_direction += 1

        else:
            prev_direction = directions[action]
            count_same_direction = 0
        
        our_snake(snake_block, snake_List)
        Your_score(Length_of_snake - 1)
        
        pygame.display.update()
        
        #the snake ate the apple
        if x1 == foodx and y1 == foody: 
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            Length_of_snake += 1
        
        steps_taken += 1
        clock.tick(snake_speed)

        #if the snake does not find an apple in a given amount of steps
        #end the game
        if steps_taken >= steps_per_game * Length_of_snake:
            game_close = True
        
    return score1 + score2 + Length_of_snake * 5000
    
