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
 
dis_width = 100
dis_height = 100

dis_height_complete = dis_height + 30
 
dis = pygame.display.set_mode((dis_width, dis_height_complete))
pygame.display.set_caption('Snake Game by Edureka')
 
clock = pygame.time.Clock()
 
snake_block = 10
snake_speed = 20
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
 
 
def Your_score(score):
    value = score_font.render(str(score), True, black)
    dis.blit(value, [0, dis_height-10])
 
 
 
def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])
 
 
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])
 
def make_array(snake_List, foodx, foody):
    # 0 indicates empty space, 1 indicates snakes body, 2 indicates food
    #np_snake_array = np.array(snake_List)
    snake_Head = list(map(int, snake_List[-1]))
    input_layer = []
##    if snake_Head[0] > foodx:
##        input_layer.append(0)
##    elif snake_Head[0] < foodx:
##        input_layer.append(1)
##    else:
##        input_layer.append(2)
##
##    if snake_Head[1] > foody:
##        input_layer.append(0)
##    elif snake_Head[1] < foody:
##        input_layer.append(1)
##    else:
##        input_layer.append(2)
##    input_layer.append(foodx-snake_Head[0])
##    input_layer.append(foody-snake_Head[1])
##    if snake_Head[0] > foodx: #food is left of the snake
##        input_layer.append(1 if foody < snake_Head[1] else 0)
##        input_layer.append(1 if foody == snake_Head[1] else 0)
##        input_layer.append(1 if foody > snake_Head[1] else 0)
##    else:
##        input_layer.append(0)
##        input_layer.append(0)
##        input_layer.append(0)
##        
##    if snake_Head[0] == foodx: 
##        input_layer.append(1 if foody < snake_Head[1] else 0)
##        input_layer.append(1 if foody > snake_Head[1] else 0)
##    else:
##        input_layer.append(0)
##        input_layer.append(0)
##    
##    if snake_Head[0] < foodx:
##        input_layer.append(1 if foody < snake_Head[1] else 0)
##        input_layer.append(1 if foody == snake_Head[1] else 0)
##        input_layer.append(1 if foody > snake_Head[1] else 0)
##    else:
##        input_layer.append(0)
##        input_layer.append(0)
##        input_layer.append(0)
    
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
    if snake_Head[0] > foodx:
        input_layer.append(-1)
    elif snake_Head[0] < foodx:
        input_layer.append(1)
    else:
        input_layer.append(0)

    if snake_Head[1] > foody:
        input_layer.append(-1)
    elif snake_Head[1] < foody:
        input_layer.append(1)
    else:
        input_layer.append(0)
    
    return input_layer
                    




##def run_game_with_ML(weights):
##    max_score = 0
##    avg_score = 0
##    test_games = 1
##    score1 = 0
##    steps_per_game = 2500
##    score2 = 0
##    directions = ["left", "right", "up", "down"]
##    
##    for _ in range(test_games):
##        snake_start, snake_position, apple_position, score = starting_positions()
##
##        count_same_direction = 0
##        prev_direction = 0
##
##        for _ in range(steps_per_game):
##            predictions = []
##            predicted_direction = np.argmax(np.array(forward_propagation(np.array(
##                make_array(snake_list, foodx, foody).reshape(-1, 7), weights)))
##            print("predicted_direction:", predicted_direction)
##            
####            if predicted_direction == prev_direction:
####                count_same_direction += 1
####            else:
####                count_same_direction = 0
####                prev_direction = predicted_direction
####
####            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
####            if predicted_direction == -1:
####                new_direction = np.array([new_direction[1], -new_direction[0]])
####            if predicted_direction == 1:
####                new_direction = np.array([-new_direction[1], new_direction[0]])
####
####            button_direction = generate_button_direction(new_direction)
####
####            next_step = snake_position[0] + current_direction_vector
##            next_step = directions[predicted_direction]
##            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(),
##                                                                                        snake_position) == 1:
##                score1 += -150
##                break
##
##            else:
##                score1 += 0
##
##            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
##                                                              button_direction, score, display, clock)
##
##            if score > max_score:
##                max_score = score
##
##            if count_same_direction > 8 and predicted_direction != 0:
##                score2 -= 1
##            else:
##                score2 += 2
##
##
##    return score1 + score2 + max_score * 5000

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

    
    
    x1 = int(dis_width / 2)
    y1 = int(dis_height / 2)
 
    x1_change = 0
    y1_change = 0
    directions = ["left", "right", "up", "down"]
    
    snake_List = [[x1, y1]]
    Length_of_snake = 1
    
    foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

    prev_direction = ""
    count_same_direction = 0
    
    score1 = 0
    steps_per_game = 100
    score2 = 0
    while not game_close:
        input_layer = make_array(snake_List, foodx, foody)
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

        
        
        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            score1 -= 200
            game_close = True
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

        for x in snake_List[:-1]:
            if x == snake_Head:
                score1 -= 200
                game_close = True
        score1 -= 1       
        if len(snake_List) > Length_of_snake:
            del snake_List[0]
 
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
        if x1 == foodx and y1 == foody: 
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            Length_of_snake += 1
        
        steps_taken += 1
        clock.tick(snake_speed)
        if steps_taken >= steps_per_game * Length_of_snake:
            game_close = True

    return score1 + score2 + Length_of_snake * 5000
    
