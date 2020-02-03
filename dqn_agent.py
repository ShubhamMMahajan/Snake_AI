from collections import deque
import numpy as np
import keras
import pickle
import random
import time

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Memory:
        def __init__(self, max_size):
                self.buffer = deque(maxlen = max_size)

        def add(self, experience):
                self.buffer.append(experience)

        def load(self, buffer):
                self.buffer = buffer

        def sample(self, batch_size):
                buffer_size = len(self.buffer)
                index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)
                return [self.buffer[i] for i in index]

        def __len__(self):
                return len(self.buffer)

class DQNAgent:
        def __init__(self, state_size, action_size, learning_rate, discount_rate, epsilon, epsilon_min, epsilon_decay):
                self.state_size = state_size
                self.action_size = action_size
                self.memory = Memory(1000000)
                self.learning_rate = learning_rate
                self.discount_rate = discount_rate
                self.epsilon = epsilon
                self.epsilon_min = epsilon_min
                self.epsilon_decay = epsilon_decay
                self.model = self.build_model()
                self.actions = ["left", "right", "up", "down"]

        def build_model(self):
                model = keras.models.Sequential()
                model.add(keras.layers.Dense(48, input_dim=self.state_size, activation='relu'))
                model.add(keras.layers.Dense(20, activation='relu'))
                model.add(keras.layers.Dense(self.action_size, activation='softmax'))
                model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
                return model

        def remember(self, state, action, reward, next_state, done):
                self.memory.add((state, action, reward, next_state, done))

        def act(self, state):
                if np.random.rand() < self.epsilon:
                        if self.epsilon < 0.75:
                                print("Random Action...")
                        #print("Random Action....: ")
                        return random.randrange(self.action_size)
                
                print("Action Rewards: ", self.model.predict(np.reshape(state, (1, self.state_size)))[0])
                print("Calculated Action: ", self.actions[np.argmax(self.model.predict(np.reshape(state, (1, self.state_size)))[0])])
                return np.argmax(self.model.predict(np.reshape(state, (1, self.state_size)))[0])

        def replay(self, batch_size):
                minibatch = self.memory.sample(batch_size)
                for state, action, reward, next_state, done in minibatch:
                        target = reward
                        if not done:
                                target = (reward + self.discount_rate * np.amax(self.model.predict(np.reshape(state, (1, self.state_size)))[0]))
                        target_f = self.model.predict(np.reshape(state, (1, self.state_size)))
                        target_f[0][action] = target
                        self.model.fit(np.reshape(state, (1, self.state_size)), target_f, epochs=1, verbose=0)
                if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay

        def load(self, name):
                data = pickle.load(open(name, 'rb'))
                self.model.set_weights(data['weights'])
                self.memory.load(data['buffer'])
                self.epsilon = data['epsilon']

        def save(self, name):
                data = {'epsilon': self.epsilon, 'buffer': self.memory.buffer, 'weights': self.model.get_weights()}
                pickle.dump(data, open(name, 'wb'))
