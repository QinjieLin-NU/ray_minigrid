import matplotlib.pyplot as plt
import pickle
import numpy as np

data_array = []
labels = ['ppo','a3c','dqn']
with open('data/ppo5.pkl', 'rb') as f:
    data = pickle.load(f)
    data_array.append(data)   
# with open('data/a3c.pkl', 'rb') as f:
    # data = pickle.load(f) 
    # data_array.append(data)  
# with open('data/dqn.pkl', 'rb') as f:
    # data = pickle.load(f) 
    # data_array.append(data)  

for i in range(len(data_array)):
    plt.plot(data_array[i],label=labels[i])
plt.legend()
plt.show()