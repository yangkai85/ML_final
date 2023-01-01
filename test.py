import numpy as np
from model import *
from optimizer import *
from tools import *
from tqdm import tqdm
import pickle
import csv
network1 = CNN2()#512 128
network2 = CNN()#1024 128
network3 = NewLeNet()#512,256,128

dataset = np.load("test_data.npy")
dataset = normalize(dataset)
dataset = dataset.reshape(-1,1,28,28)

with open("save1.pkl","rb") as file1:#
    params1 = pickle.load(file1)
    file1.close()
with open("save2.pkl","rb") as file2:#
    params2 = pickle.load(file2)
    file2.close()
with open("save3.pkl","rb") as file3:#
    params3 = pickle.load(file3)
    file3.close()
network1.loadparams(params1)
network2.loadparams(params2)
network3.loadparams(params3)
total = int(dataset.shape[0])

with open("save.csv",'w') as savefile:
    writer = csv.writer(savefile)
    writer.writerow(['Id','Category'])
    pbar = tqdm(range(total))
    for t in pbar:
        data_ = np.expand_dims(dataset[t],axis=0)
        out1 = network1.forward(data_)
        out2 = network2.forward(data_)
        out3 = network3.forward(data_)
        maxindex1 = np.argmax(out1,axis=1)
        maxindex2 = np.argmax(out2,axis=1)
        maxindex3 = np.argmax(out3,axis=1)
        outs = np.array([maxindex1[0],maxindex2[0],maxindex3[0]])
        answer = np.argmax(np.bincount(outs))
        writer.writerow([t,answer])