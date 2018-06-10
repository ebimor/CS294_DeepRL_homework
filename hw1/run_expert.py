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

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.wrappers.scikit_learn import KerasRegressor

def env_dims(env):
    return (env.observation_space.shape[0], env.action_space.shape[0])

class policy:
    def __init__(self, envname,trained=0):
        self.trainedBefore=trained
        self.envname=envname
        self.env = gym.make(envname)
        self.state_len, self.action_len = env_dims(self.env)
        self.model = Sequential()
        self.model.add(Dense(units=128, input_dim=self.state_len, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(units=64, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(units=self.action_len, kernel_initializer='normal'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X,Y, epchs=10, bch_size=50):

        if self.trainedBefore:
            del self.model
            print("Loading previously trained model")
            self.model = load_model('partly_trained.h5')

        self.model.fit(X, Y, epochs=epchs, batch_size=bch_size,  verbose=1, validation_split=0.02)
        self.trainedBefore=1;
        self.model.save('partly_trained.h5')
        del self.model
    def evaluate(self, steps_, num_rollouts, render):

        self.model = load_model('partly_trained.h5')

        observations = np.zeros(shape=(1,self.state_len))
        actions = np.zeros(shape=(1,self.action_len))
        max_steps =  steps_ or self.env.spec.timestep_limit

        for i in range(num_rollouts):
            print('iter', i)
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0

            # We start with a set of observation from the env and thus, we go to step 3
            while not done:
                action = self.model.predict(obs[None,:])
                if steps==0:
                    observations[steps,:]=obs
                    actions[steps,:]=action
                else:
                    observations=np.vstack((observations, obs))
                    actions=np.vstack((actions, action))

                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if render:
                    self.env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            #print("total reward is ", totalr)
        return observations, actions, totalr

    def labelData(self, observations):

        print('loading and building expert policy to label Data')
        policy_fn = load_policy.load_policy("experts/"+self.envname+".pkl")
        print('loaded and built')

        with tf.Session():
            tf_util.initialize()

            data_len, _ =observations.shape
            print("Labling a dataset of length: ", data_len)
            labels = np.zeros(shape=(1,self.action_len))

            for i in range(data_len):
                label = policy_fn(observations[i,None])
                if i==0:
                    labels[i,:]=label
                else:
                    labels=np.vstack((labels, label))
        print("Lables: ",labels.shape)
        print("observations: ",observations.shape)
        return labels




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument("--simulate", type=int, default=0)
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    TrainingNum=0
    if args.simulate==0:  #cloning
        R=0
        CB=policy(args.envname)
        observations, actions= simulate_expert(args.envname,args.max_timesteps,10,0)
        CB.train(observations,actions,100,100)
        observations, actions, R = CB.evaluate(args.max_timesteps, 2, 0)
        print("Evaluation reward of CB is: ", R)
    elif args.simulate==1: #run dagger
        print("Running DAGGER Algorithm")
        R=0
        Dagger=policy(args.envname)  #initiate a learning model
        observations, actions= simulate_expert(args.envname,args.max_timesteps,10,0) #generate human data

        # train on human data
        Dagger.train(observations,actions,120,100)
        while R<10000 and TrainingNum<200:
            #apply trained policy and save observation_space
            observationsSim, _, R = Dagger.evaluate(args.max_timesteps, 1, 0)
            print("Dagger reward is: ",R)

            #label observations by human/expert
            labeled_action=Dagger.labelData(observationsSim)
            print("number of labeled data is: ", labeled_action.shape)

            #Concatenate data
            actions=np.concatenate((actions,labeled_action))
            observations=np.concatenate((observations,observationsSim))

            print("new dataset dimentions is: ", observations.shape)
            print("new dataset dimentions is: ", actions.shape)

            # train on labeled data
            Dagger.train(observations,actions,10,100)

            TrainingNum=TrainingNum+1
            print("TrainingNum is: ", TrainingNum)



    elif args.simulate==2: #load the learned model and simulate
        CB=policy(args.envname,1)
        CB.evaluate(args.max_timesteps, 20, args.render)


def simulate_expert(envname, steps_, num_rollouts, render=1):

    #import gym
    env = gym.make(envname)

    max_steps =  steps_ or env.spec.timestep_limit

    print('loading and building expert policy for data generation')
    policy_fn = load_policy.load_policy("experts/"+envname+".pkl")
    print('loaded and built')


    with tf.Session():
        tf_util.initialize()

        state_len, action_len = env_dims(env)
        observations = np.zeros(shape=(1,state_len))
        actions = np.zeros(shape=(1,action_len))

        for i in range(num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = policy_fn(obs[None,:])

                if steps==0:
                    observations[steps,:]=obs
                    actions[steps,:]=action
                else:
                    observations=np.vstack((observations, obs))
                    actions=np.vstack((actions, action))

                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps >= max_steps:
                    break
            print("generating data iteration is: ", i)
            print("reward of expert is: ", totalr)
        return observations, actions



if __name__ == '__main__':
    main()
