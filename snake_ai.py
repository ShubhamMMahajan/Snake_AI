import pygame
import time
import random
from dqn_agent import DQNAgent

pygame.init()
pygame.display.init()
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
 
dis_width = 600
dis_height = 400
 
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game by Edureka')
 
clock = pygame.time.Clock()
 
snake_block = 10
snake_speed = 50
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
 
 
def Your_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])
 
 
 
def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])
 
 
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])
 
def make_array(snake_List, foodx, foody):
    # 0 indicates empty space, 1 indicates snakes body, 2 indicates food
    board = []
    for x in range(0, dis_width + 10, 10):
        for y in range(0, dis_height + 10, 10):
            if [x, y] in snake_List:
                board.append(1)
            elif x == foodx and y == foody:
                board.append(2)
            else:
                board.append(0)
    return board

state_size = 2501
action_size = 4
learning_rate = 0.1
discount_rate = 0.95
epsilon = 1.00
epsilon_decay = 0.9999
epsilon_min = 0.01
batch_size = 25
agent = DQNAgent(state_size, action_size, learning_rate, discount_rate, epsilon, epsilon_min, epsilon_decay)

def gameLoop(e):
    game_over = False
    game_close = False
    e = e + 1
    x1 = dis_width / 2
    y1 = dis_height / 2
 
    x1_change = 0
    y1_change = 0
    directions = ["left", "right", "up", "down"]
    snake_List = [[x1, y1]]
    Length_of_snake = 1
    checkpoints = [25, 50, 100, 200, 300, 400, 500]
    foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
    print("episode:", e)
    run = 0
    while not game_over:
        board = make_array(snake_List, foodx, foody)
        #print(run)
        if game_close == True:
            if e in checkpoints:
                agent.save('./models/model_{}'.format(e))
            gameLoop(e)
#             dis.fill(blue)
#             message("You Lost! Press C-Play Again or Q-Quit", red)
#             Your_score(Length_of_snake - 1)
#             pygame.display.update()
 
#             for event in pygame.event.get():
#                 if event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_q:
#                         game_over = True
#                         game_close = False
#                     if event.key == pygame.K_c:
#                         gameLoop()
 
        #for event in pygame.event.get():
            #if event.type == pygame.QUIT:
            #    game_over = True
            #if event.type == pygame.KEYDOWN:
        reward = -1
        action = agent.act(board)
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
            reward -= 1000
            game_close = True
        x1_old = x1
        y1_old = y1
        x1 += x1_change
        y1 += y1_change
        dis.fill(blue)
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]
 
        for x in snake_List[:-1]:
            if x == snake_Head:
                reward -= 1000
                game_close = True
 
        our_snake(snake_block, snake_List)
        Your_score(Length_of_snake - 1)
 
        pygame.display.update()
        if not game_close:
            reward += (foodx - x1_old)**2 - (foodx - x1)**2 + (foody - y1_old)**2 - (foody - y1)**2
        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            Length_of_snake += 1
            #rint(snake_List)
        new_board = make_array(snake_List, foodx, foody)
        agent.remember(board, action, reward, new_board, game_close)
        if e > batch_size:
            agent.replay(batch_size)
        clock.tick(snake_speed)
        run += 1
        
    pygame.quit()
    quit()
e = 0 
gameLoop(e)
