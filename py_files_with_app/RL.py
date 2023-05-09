####################################################
#                                                  #
# Author(s): Dynamic Asset Allocation   - Zachary  #
#            Burpee,                               #
# Class: Adv. Big Data Analytics Project           #
# Professor: Professor Ching-Yung Lin              #
# Description: RL Implementation                   #
#                                                  #
#                                                  #
#                                                  #
####################################################

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import error, spaces, utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TradingEnv(gym.Env):
    def __init__(self, start_date, end_date, ticker, df, tc=0.05/100):
        self.start = start_date
        self.end = end_date
        self.tc = tc
        self.ticker = ticker
        self.df = df

        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(low=-1,high=1,dtype=np.float32)
        returns = self.load_dataset()
        self.data_df = self.create_features(returns)
        self.curr_index = 0
        self.data_len = self.data_df.shape[0]
        self.action = 0

    def step(self, action):

        done = False
        stock_return = self.extract_return(self.data_df, self.curr_index)
        change_in_position = np.abs(self.action - action)
        cost = change_in_position * self.tc
        reward = action*stock_return - cost

        if self.curr_index == self.data_len - 2:
            done = True
        self.curr_index += 1
        self.action = action

        obs = self.extract_state(self.data_df, self.curr_index).values

        info = { 'date' : self.data_df.index[self.curr_index], 'return' : stock_return }

        return obs, reward, done, info

    def reset(self):
        self.curr_index = 0
        return self.extract_state(self.data_df, self.curr_index).values

    def render(self,mode='human'):
        pass

    def extract_return(self, df, i):
        return df.iloc[i]['Y']

    def extract_state(self, df, i):
        return df.iloc[i][['r%d' % i for i in range(30)]]
    
    def downloader(self):
        history = yf.download(tickers = self.ticker,  # list of tickers
                        period = "3y",         # time period
                        interval = "1d",       # trading interval
                        prepost = False,       # download pre/post market hours data?
                        repair = True)
        return history
    
    def load_dataset(self):
        df = self.downloader()
        mask = ( self.start <= df.index ) & ( df.index <= self.end )
        df = df[mask]
        df = df.sort_values(by='Date')
        returns = df['Close'].pct_change()
        
        return returns

    def create_features(self, returns):
        dfs = []
        for i in range(30):
            dfs.append(returns.shift(i).rename('r%d'%i))
        dfs.append(returns.shift(-1).rename('Y'))
        df_net = pd.concat(dfs, axis=1)
        df_net = df_net.dropna()
        
        return df_net

def instant(df, ticker):
    ENV_NAME = 'TradingEnv-v0'

    reg = register(
        id=ENV_NAME,
        entry_point='RL:TradingEnv',
        kwargs={
            'start_date' : '2019-01-01',
            'end_date' : '2023-01-10',
            'ticker' : ticker,
            'df' : df
        }
    )
    env = gym.make(ENV_NAME)

    return env

def model_creator(batch_size):
    model = keras.Sequential()
    model.add(keras.Input(shape=(30)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    # you should later modify this neural network
    model.add(layers.Dense(3, activation='softmax')) # you should have one output for each possible action

    return model

def training(env):
    batch_size = 32
    # Model used for selecting actions (principal)
    model = model_creator(batch_size)
    # Then create the target model. This will periodically be copied from the principal network 
    model_target = model_creator(batch_size)

    model.build((batch_size,30,))
    model_target.build((batch_size,30,))

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    tf.keras.utils.disable_interactive_logging()
    gamma = 0.99
    # Our Experience Replay memory 
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward_tracker_train = []
    reward_returns = []

    # Replay memory size
    max_memory = 1000 # You can experiment with different sizes.

    running_reward = 0
    episode_count = 0
    timestep_count = 0
    actsize = 3

    update_after_actions = 4

    # How often to update the target network
    target_update_every = 25
    loss_function = keras.losses.MeanSquaredError() # You can use the Huber loss function or the mean squared error function 

    max_episodes = 50
    max_steps_per_episode = 1000
    last_n_reward = 100
    decayRate = 0.99
    epsilon = 0.2
    eps_delta = epsilon / 300

    for episode in range(max_episodes):
        state = np.array(env.reset())
        episode_reward = 0
        
        # change epsilon after most states have been explored
        epsilon = np.max((epsilon - eps_delta), 0)

        for timestep in range(1, max_steps_per_episode):
            timestep_count += 1

            # exploration
            if np.random.rand() < epsilon:
                # Take random action
                action = np.random.uniform(-1, 1)
            else:
                # Predict action Q-values
                # From environment state
                state_t = tf.convert_to_tensor(state)
                state_t = tf.expand_dims(state_t, 0)
                action_vals = model.predict(state_t)
                # Choose the best action
                action = np.argmax(action_vals)
                if action == 0:
                    action = -1
                elif action == 1:
                    action = 0
                else:
                    action = 1
                
            # follow selected action
            state_next, reward, done, info = env.step(action);
            state_next = np.array(state_next)
            episode_reward += reward

            # Save action/states and other information in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            rewards_history.append(reward)
            done_history.append(done)

            state = state_next

            # Update every Xth frame to speed up (optional)
            # and if you have sufficient history
            if timestep_count % update_after_actions == 0 and len(action_history) > batch_size:
                
                # Gather slices for correct implementation 
                slices = np.random.choice(len(action_history), batch_size)
                state_sample = np.array(state_history)[slices]
                state_next_sample = np.array(state_next_history)[slices]
                rewards_sample = np.array(rewards_history)[slices]
                action_sample = np.array(action_history)[slices]
                done_sample = np.array(done_history)[slices]

                # Create for the sample states the targets (r+gamma * max Q(...) )
                Q_next_state = model_target.predict(state_next_sample);
                Q_targets = rewards_sample + gamma * np.max(Q_next_state, axis=1)
                
                penalty = 1
                # If the episode was ended (done_sample value is 1)
                # you can penalize the Q value of the target by some value `penalty`
                Q_targets = Q_targets * (1 - done_sample) - penalty*done_sample
                
                # What actions are relevant and need updating
                relevant_actions = tf.one_hot(action_sample, actsize)
                # we will use Gradient tape to do a custom gradient 
                # in the `with` environment we will record a set of operations
                # and then we will take gradients with respect to the trainable parameters
                # in the neural network
                with tf.GradientTape() as tape:
                    # Train the model on your action selecting network
                    q_values = model(state_sample) 
                    # We consider only the relevant actions
                    Q_of_actions = tf.reduce_sum(tf.multiply(q_values, relevant_actions), axis=1)
                    # Calculate loss between principal network and target network
                    loss = loss_function(Q_targets, Q_of_actions)

                # Nudge the weights of the trainable variables towards 
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if timestep_count % target_update_every == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())

            # Don't let the memory grow beyond the limit
            if len(rewards_history) > max_memory:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]
            if done: break
                
        # reward of last 100
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > last_n_reward: del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        running_reward_tracker_train.append(running_reward)
        episode_count += 1
        template = "running reward: {:.2f} at episode {}, frame count {}, epsilon {}"
        print(template.format(abs(1-running_reward), episode_count, timestep_count, epsilon))

def predictNextWindow(env):
    obs = env.reset()
    done = False
    # 0 - hold, -1 - sell, 1 - buy
    state_t = tf.convert_to_tensor(obs)
    state_t = tf.expand_dims(state_t, 0)
    action_vals = model.predict(state_t)
    # Choose the best action
    action = np.argmax(action_vals)
    if action == 0:
        action = -1
    elif action == 1:
        action = 0
    else:
        action = 1
        
    obs, reward, done, info = env.step(action);

    return obs

