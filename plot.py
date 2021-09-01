import matplotlib.pyplot as plt
import pickle
import numpy as np
def smooth(scalars, weight):   # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

class FileLabel:
    def __init__(self,gama,iter_num,mini_batchsize,train_batchsize):
        self.gama = gama
        self.iter_num = iter_num
        self.mini_batchsize = mini_batchsize
        self.train_batchsize = train_batchsize

    def get_filename(self):
        res = f"ppo_MiniGrid-KeyCorridorS3R3-v0_{self.gama}_{self.iter_num}_{self.mini_batchsize}_{self.train_batchsize}.pkl"
        return res

    def get_label(self):
        res = f"gama{self.gama}_iter{self.iter_num}_minibatch{self.mini_batchsize}_trainbatch{self.train_batchsize}.pkl"
        return res

data_array = []
# labels = ['ppo','a3c','dqn']
# with open('data/ppo4.pkl', 'rb') as f:
#     data = pickle.load(f)
#     data_array.append(data)   
# with open('data/a3c.pkl', 'rb') as f:
    # data = pickle.load(f) 
    # data_array.append(data)  
# with open('data/dqn.pkl', 'rb') as f:
    # data = pickle.load(f) 
    # data_array.append(data)  

t1 = FileLabel(0.995, 2, 3200 ,160000)
t2 = FileLabel(0.9, 2, 3200, 160000)
t3 = FileLabel(0.995, 20, 3200 ,160000)
t4 = FileLabel(0.995, 2, 1600 ,160000)
t5 = FileLabel(0.995, 2, 3200 ,80000)
ts = [t1,t2,t3,t4,t5]
labels = [t1.get_label(),t2.get_label(),t3.get_label(),t4.get_label(),t5.get_label()]

for t in ts:
    file_name = 'data/%s'%t.get_filename()
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        data_array.append(data) 

for i in range(len(data_array)):
    smooth_data= smooth(data_array[i],0.5)
    plt.plot(smooth_data,label=labels[i])
plt.legend()
plt.show()