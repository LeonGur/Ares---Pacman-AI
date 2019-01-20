#Leon Gurtler
#Ares - Pacman AI -ram version
import tensorflow as tf
import numpy as np
import gym, h5py, os, keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

#declaring variables
learning_rate         = 0.001
initial_games         = 5000
max_memory_length     = 50000
load_previous_weights = True
load_model            = False
weights_filename      = "Pacman-v4-weights.h5"
env                   = gym.make('MsPacman-ram-v0')

#Neural Network
model = Sequential()
model.add(Dense(128, activation = 'relu', input_dim = 129)) #ram grid contains 128 numbers + 1 action number between 0 and 8
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1))         #the nn gives the predicted reward as a sinlge integer output
opt = optimizers.adam(lr = learning_rate)
model.compile(loss = 'mse', optimizer = opt, metrics = ['accuracy'])

#load previous model weights if it exist
if load_previous_weights:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/" + weights_filename
    print("filepath ", fn)
    if os.path.isfile(fn):
        print("loading weights")
        load_model = True
        model.load_weights(weights_filename)
    else:
        print("File ", weights_filename, " does not exist. Retaining... ")

#this function predicts and returns the expected reward for the input observation and action
def predict_total_rewards(observation, action):
    observation_action = np.concatenate((observation, [action]), axis = 0)
    pred = model.predict([[observation_action]])
    return pred[0]

#declaring game variables
training_data       = []
training_label      = []
avg_score           = 0
game_nr             = 1

for _ in range(initial_games):
    #Those variables are reseted between each game
    done                    = False
    score                   = 0
    prev_observation        = []
    game_memory             = []
    total_score             = 0
    steps                   = 0
    env.reset()

    while done == False:
        #if this is the first action, or the game number is divisible by 7 a random action is exectued
        if len(prev_observation) == 0 or game_nr%7 == 0:
            action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        prev_observation = observation
        if reward > 10:
            total_score += 25
        else:
            total_score += reward
        score       += reward
        steps       += 1
        avg_score   += reward

        #this part of the program appends the observation and action into the game_memory
        if len(prev_observation) > 0:
            prev_observation_action = np.concatenate((prev_observation, [action]), axis = 0)
            game_memory += [prev_observation_action]

        #after the initial amount of games, the program starts to actually play the game, insted of executing random moves
        if game_nr > 200 or load_model == True:
            env.render()
            predicted_rewards = 0
            idle_reward = 0
            #the following for loop predicts the expected reward for each of the 8 possible actions and chooses the highes one
            for x in range(8):
                predicted_rewards = predict_total_rewards(prev_observation, x)
                if predicted_rewards > idle_reward:
                    idle_reward = predicted_rewards
                    action = x

            #if the initial amount of games is not reached, Ares executes random moves
        else:
            action = env.action_space.sample()


    #before being reset the game_memory is added to the training_data. The reward label is equal to the overall score of the game
    training_data += game_memory
    for _ in range(0, len(game_memory)):
        training_label.append([total_score])

    if game_nr > 2500:
        training_data  = []
        training_label = []
        game_nr = 0

    #prints the game number, the score and the avarage score
    print(game_nr, " | score:", score, " |  avg score:", str(int(avg_score/(game_nr))))
    game_nr += 1

    #after the initial games, the nn is retrained every tenth game
    if game_nr % 10 == 0 and game_nr > 190:
        model.fit(np.asarray(training_data), np.asarray(training_label), batch_size = 128, epochs = 5, verbose = 2)
        print("Saving weights")
        model.save_weights(weights_filename)
