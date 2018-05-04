#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python3.5 run_expert.py experts/Ant-v1.pkl Ant-v2 --render --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.wrappers.scikit_learn import KerasRegressor

def env_dims(env):
    return (env.observation_space.shape[0], env.action_space.shape[0])


def policy(X,Y, envname):
    import gym
    env = gym.make(envname)
    state_len, action_len = env_dims(env)
    model = Sequential()
    model.add(Dense(state_len, input_dim=11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(action_len, kernel_initializer='normal'))

    model.summary()

    tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=9, batch_size=50,  verbose=1, validation_split=0.3, callbacks=[tbCallBack])
    return model

def simulate(envname, steps_, num_rollouts):

    import gym
    env = gym.make(envname)

    max_steps =  steps_ or env.spec.timestep_limit

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy("experts/"+envname+".pkl")
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        state_len, action_len = env_dims(env)
        observations = np.zeros(shape=(1,state_len))
        actions = np.zeros(shape=(1,action_len))

        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            # We start with a set of observation from the env and thus, we go to step 3

            while not done:
                #Step 3: labling the observation by human/expert, which means finding the optimal actions for each obs
                action = policy_fn(obs[None,:])

                #Step 4: aggregate observation and action data
                if steps==0:
                    observations[steps,:]=obs
                    actions[steps,:]=action
                else:
                    observations=np.vstack((observations, obs))
                    actions=np.vstack((actions, action))
                ##################### GO BACK TO STEP 1

                #Step1: terrain the NN to learn from these obs and actions


                #Step 2: Apply the model to obtain a new a set of observations
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                #if args.render:
                    #env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            return observations, actions



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    observations, actions= simulate(args.envname,args.max_timesteps,args.num_rollouts)
    print(actions.shape)
    
    model=policy(observations,actions,args.envname)


if __name__ == '__main__':
    main()
