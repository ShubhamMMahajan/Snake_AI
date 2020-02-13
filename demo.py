from genetic_algorithm import *
import csv


# n_x -> no. of input units
# n_h -> no. of units in hidden layer 1
# n_h2 -> no. of units in hidden layer 2
# n_y -> no. of output units

n_x = 6
n_h = 9
n_h2 = 15
n_y = 4


runs = 50
num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y

#get the snake with the highest score
def get_weight():
    weights_file = open("weights_sorted.csv", "r")
    weight = weights_file.readline()
    weight_values = np.array([float(i) for i in weight.split(",")[2:]])
    weights_file.close()
    return weight_values
        
#create 50 snakes using the weights with the highest score
population = np.array([get_weight()] * runs)


for game in range(runs):
    fitness = cal_pop_fitness(population)
