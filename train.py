import numpy as np
from model import *
from optimizer import *
from tools import *
from tqdm import tqdm
import pickle

data = np.load("train_data.npy")
label = np.load("train_label.npy")


label = one_hot(10,label)      
data = normalize(data)
data = data.reshape(-1,1,28,28)
train_data,val_data,train_label,val_label = split_data(data,label,split_size=0.9)


network = NewLeNet()

# 超参数
epoch = 30
batch_size = 100
seed = 1234
np.random.seed(seed)
# 优化器
opt = Adam(0.001)
load = False


if load:
    with open("cykparams2/epoch21_0.92.pkl","rb") as file:
        params = pickle.load(file)
        file.close()
        network.loadparams(params)
predict_val = network.forward(val_data)

acc0 = network.accuracy(predict_val,val_label)
print(f"=============== Test Accuracy:{acc0} ===============")
params = network.params

for i in range(epoch):
        
    iteration = int(data.shape[0]/batch_size)
        
    pbar = tqdm(range(iteration))
        
    for j in pbar:
        start = j*batch_size
        end = start + batch_size
        train_data_batch = data[start:end]
        train_label_batch = label[start:end]
            
        network.forward(train_data_batch)
        grads = network.gradient(train_label_batch)
        loss = network.layers[-1].loss
        opt.update(params,grads)
            
        pbar.set_description(f"epoch:{i+1} loss:{loss} lr:{opt.lr}")
        
    predict_val = network.forward(val_data)
    acc1 = network.accuracy(predict_val,val_label)
    print(f"=============== Test Accuracy:{acc1} ===============")

    if (abs(acc1-acc0)<0.0001 and opt.lr>=1e-6):
        lr = lr/5
        opt.lr = lr
    acc0 = acc1
    with open(f"./cykparams3/epoch{i}_{acc1}.pkl",'wb') as file:
        pickle.dump(network.params,file)

   

    
    




