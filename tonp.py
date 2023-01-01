import cupy as cp
import pickle


with open(r"epoch1_0.9208.pkl","rb") as file:
     params = pickle.load(file)
     file.close()

savedict = {}
for key,value in params.items():
    savedict[key] = cp.asnumpy(value)

with open(r"C:\Users\27914\Desktop\ML_final\save3.pkl","wb") as savefile:
    pickle.dump(savedict,savefile)