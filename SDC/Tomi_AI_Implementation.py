# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 19:46:30 2018

@author: Tomi
"""

#importing the libraries
import numpy as np
import random 
import os 
import torch #because we will use pytorch to implement
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#creating the architecture of the neural network

class Network(nn.Module):#inheriting from nn.module
    #every object created below will have 5 input neurons, 3 output neurons and 30 hidden layer neurons
    def __init__(self, input_size,nb_action): 
    #whenever i want to use a variblae for this object, i will write self to specify that it is a varibale of this object
        super(Network,self).__init__()#to be able to use the tools of module
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size,30)#input_size is our input here and 30 will be the number of hidden layer neurons
        self.fc2 = nn.Linear(30,nb_action)#the input layer here is the hidden layer and the output layer here has nb_action neurons
        
    def forward(self,state):
    #the function for forward propagation, will activate neurons and return Q values of each of the three possible actions 
        x = F.relu(self.fc1(state)) #rectifier function to activate hidden neurons
        q_values = self.fc2(x)
        return q_values
    
#Implementing Experience Replay 
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity#the maximum number of transitions we want to have in our memory of events
        self.memory = [] #transitions will be added to this memory list whenever they happen
        
    def push(self, event):    
        self.memory.append(event) #appends the new event composed of the last state, new state, last action and last reward to the memory
        if len(self.memory) >self.capacity: #if memory has reached capacity
            del self.memory[0] #remove the first event in the memory
            
    def sample(self, batch_size):#each sample has a total of batch_size elements
        #a zip function reshapes your list eg:
        #if list = ((1,2,3),(4,5,6)), then zip(*list) = ((1,4), (2,3), (5,6)) 
        #we need the zip function because we want to separate states from actions from rewards 1 & 2. we want each of these three things to be in batches
        #the random.sample function allows us to take a random sample from the memory that have a fixed batch size
        samples = zip(*random.sample(self.memory, batch_size)) 
        #torch.cat(x,0) concatenates x to the first dimension
        #torch variables hzve both a tensor and a gradient
        #we apply this lambda funtion onto all of our samples
        return map(lambda x: Variable(torch.cat(x,0)),samples) #the function changes sample into a pytorch variable
        
#Implementing the Deep Q Learning
        
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = [] #keeps track of how training is going by keeping track of the last 100 rewards and showing us how the mean is evolving
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)#used to perform stocastic gradient descent. .parameters is used to access the parameters of model
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #the first dimension will now correspond to the batch
        self.last_action = 0 #will end up being only 0, 1 or 2 so we can initialise it as 0 here
        self.last_reward = 0 #reward is a float between 0 and 1 so we will initialise it to 0
         
    def select_action(self, state):
        #we use self.model below because that is the output of our neural network
        #we changed the state torch tensor into  a torch variable
        #we add a temperature parameter of 7 by multiplying the output by 7. the larger this parameter, the more sure the network will be of the action it decides to take
        #the higher the temperature parameter, the higher the probability of q values eg:
        #softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3]*3) which will result in higher values
        probs = F.softmax(self.model(Variable(state, volatile = True))*0)#softmax-it will give the largest probability to the highest q value   
        action = probs.multinomial() #returns the dimension corresponding to the "fake" batch
        return action.data[0,0] #will return 0,1, or 2
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #to gather the chosen action for each state that the batch is in. batch action must be unsqueezed so it has the same dimensions as the batch_state
        outputs = self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1) #squeeze to kill the fake dimension and get back the simple form of out outputs
        #the next output is the max of the current outputs.
        next_outputs = self.model(batch_next_state).detach().max(1)[0] #this is how we get the max of all q values of the enxt state (represented by the index 0) according to all the actions that are represented by the index 1
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_11_loss(outputs, target)#recommended if you want to implement deep q learning        
        self.optimizer.zero_grad() #reinitialising at every iteration of the loop
        #now that our optimizer is initialised, we can use it to perform back propagation
        td_loss.backward(retain_variables = True) #setting retain_variables to true helps to free some memory and improve the training performance
        self.optimizer.step()#uses the optimiser to update the weights
        
      #for updating everything that needs to be updated when we are in a new state
    def update(self, reward, new_signal): #I have to use here because I make use of a varibale i initialised in the init function 
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #updating the new state of the environment
        self.memory.push((self.last_state,new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) 
        action = self.select_action(new_state) #plays the new action after reaching the new stae
        if len(self.memory.memory)>100: #getting the memory attribute from the instance of the replayMemory class that is also called memory and is instantiated in this dqn class
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100) #making the AI learn from 100 transitions, which is why memory must be larger than 100
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action #update the action
        self.last_state = new_state #update the state
        self.last_reward = reward #update last reward
        self.reward_window.append(reward) #add the reward to the window
        if len(self.reward_window) >1000:#making the window onw of fixed size so that we can observe the mean reward and if the training is going well
            del self.reward_window[0]
        return action #the action that was just played in reaching the new state    
    
    def score(self): #to compute the mean of all the rewards in the reward window
       return sum(self.reward_window) / (len(self.reward_window) +1.) #adding 1 is to avoid teh case in which the reward window's length is 0
   
    def save(self): #saving our neural network and our optimiser
        #whenever we want to reuse our saved model later, we want it to predict the action to play with the weights that were already trained so that is why we need the last version of the weights and the optimiser
        #our dictionary below has two keys - optimizer and the model
        #state_dict is a function that saves the parameters of an object
        torch.save({'state_dict' : self.model.state_dict(), #the parameters of the model are now saved in state_dict
                    'optimizer' : self.optimizer.state_dict() #the optimizer's parameters are now stored 
                    }, 'last_brain.pth') #this is  added so that the optimizer and the model will be saved to the last_brain file 
        
    def load(self): #to load what was last saved
        if os.path.isfile('last_brain.pth'): #ospath is the path that leads to the working directory folder
            print (" => loading checkpoint")
            checkpoint = torch.load('last_brain.pth') #load this file only if it exists
            self.model.load_state_dict(checkpoint['state_dict']) #update the model 
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print ("done!")
        else:
           print("no checkpoint found")

    
    
    
    
    
    
    
    
    
    
    
    
    