from collections import OrderedDict
import numpy as np
import layer 
from layer import *
import pickle

class CNN():
    
    def __init__(self, input_dim=(1, 28, 28),config=None,conv_params=None):
        

        self.pre_node_nums = np.array([1*3*3, 6*3*3, 16*3*3, 32*3*3,120*3*3,120*7*7,1024,128])
        weight_init_scales = np.sqrt(2.0 / self.pre_node_nums)
        
        conv_param1 = {'filter_num':6, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param4 = {'filter_num':120, 'filter_size':3, 'pad':1, 'stride':1}
        
        self.params = {}
        
        self.params['W1'] = weight_init_scales[0] * np.random.randn(conv_param1['filter_num'],input_dim[0],conv_param1['filter_size'],conv_param1['filter_size'])
        self.params['b1'] = np.zeros(conv_param1['filter_num'])
        
        self.params['W2'] = weight_init_scales[1] * np.random.randn(conv_param2['filter_num'],conv_param1['filter_num'],conv_param2['filter_size'],conv_param2['filter_size'])
        self.params['b2'] = np.zeros(conv_param2['filter_num'])
        
        self.params['W3'] = weight_init_scales[2] * np.random.randn(conv_param3['filter_num'],conv_param2['filter_num'],conv_param3['filter_size'],conv_param3['filter_size'])
        self.params['b3'] = np.zeros(conv_param3['filter_num'])
        
        self.params['W4'] = weight_init_scales[3] * np.random.randn(conv_param4['filter_num'],conv_param3['filter_num'],conv_param4['filter_size'],conv_param4['filter_size'])
        self.params['b4'] = np.zeros(conv_param4['filter_num'])
        self.params['W5'] = weight_init_scales[4] * np.random.randn(120*7*7,1024)
        self.params['b5'] = np.zeros(1024)
        
        self.params['W6'] = weight_init_scales[5] * np.random.randn(1024, 128)
        self.params['b6'] = np.zeros(128)
        
        self.params['W7'] = weight_init_scales[6] * np.random.randn(128,10)
        self.params['b7'] = np.zeros(10)
        
        
        self.layers = []
        self.layers.append(Convoluntion(self.params['W1'],self.params['b1'],conv_param1['stride'],conv_param1 ['pad'])) # 0
        self.layers.append(Relu())
        self.layers.append(Convoluntion(self.params['W2'],self.params['b2'],conv_param2['stride'],conv_param2['pad'])) # 2
        self.layers.append(Relu())
        self.layers.append(Max_Pooling(pool_h=2, pool_w=2, stride=2)) 
        
        self.layers.append(Convoluntion(self.params['W3'],self.params['b3'],conv_param3['stride'],conv_param3['pad'])) # 5
        self.layers.append(Relu())
        self.layers.append(Convoluntion(self.params['W4'],self.params['b4'],conv_param4['stride'],conv_param4['pad'])) #7
        self.layers.append(Relu())
        self.layers.append(Max_Pooling(pool_h=2, pool_w=2, stride=2))
        

        self.layers.append(Affine(self.params['W5'],self.params['b5']))
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W6'],self.params['b6']))
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W7'],self.params['b7']))
        self.layers.append(SoftmaxWithLoss())
        
    def forward(self,x):

        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    
    def gradient(self,t):
        
        dout = 1

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        dout = tmp_layers[0].backward(t,dout)
        for layer in tmp_layers[1:]:
            dout = layer.backward(dout)
            
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12,14)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db 
            
        return grads 
    
    def accuracy(self, y, t):
        acc = 0.0
        length =y.shape[0]

        y_max = np.argmax(y,axis=1)
        t_max = np.argmax(t,axis=1)

        acc = np.sum(y_max==t_max)/length

        return acc

    def loadparams(self,params):
        i = 0
        keys = list(params.keys())
        for layer in self.layers:
            if isinstance(layer,Convoluntion) or isinstance(layer,Affine):
                layer.W = params[keys[i]]
                self.params[keys[i]] = params[keys[i]]
                i+=1
                layer.b = params[keys[i]]
                self.params[keys[i]] = params[keys[i]]
                i+=1



class CNN2():
    
    def __init__(self, input_dim=(1, 28, 28),hidden_size=50, output_size=10):
        pre_node_nums = np.array([1*3*3,6*3*3,16*3*3,32*3*3,120*7*7,512,128])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)
        
        conv_param1 = {'filter_num':6, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param4 = {'filter_num':120, 'filter_size':3, 'pad':1, 'stride':1}
        
        
        self.params = {}
        
        self.params['W1'] = weight_init_scales[0] * np.random.randn(conv_param1['filter_num'],input_dim[0],conv_param1['filter_size'],conv_param1['filter_size'])
        self.params['b1'] = np.zeros(conv_param1['filter_num'])
        
        self.params['W2'] = weight_init_scales[1] * np.random.randn(conv_param2['filter_num'],conv_param1['filter_num'],conv_param2['filter_size'],conv_param2['filter_size'])
        self.params['b2'] = np.zeros(conv_param2['filter_num'])
        
        self.params['W3'] = weight_init_scales[2] * np.random.randn(conv_param3['filter_num'],conv_param2['filter_num'],conv_param3['filter_size'],conv_param3['filter_size'])
        self.params['b3'] = np.zeros(conv_param3['filter_num'])
        
        self.params['W4'] = weight_init_scales[3] * np.random.randn(conv_param4['filter_num'],conv_param3['filter_num'],conv_param4['filter_size'],conv_param4['filter_size'])
        self.params['b4'] = np.zeros(conv_param4['filter_num'])
        
        self.params['W5'] = weight_init_scales[4] * np.random.randn(128*7*7, 512)
        self.params['b5'] = np.zeros(512)
        

        self.params['W6'] = weight_init_scales[5] * np.random.randn(512, 128)
        self.params['b6'] = np.zeros(128)

        self.params['W7'] = weight_init_scales[6] * np.random.randn(128,10)
        self.params['b7'] = np.zeros(10)
        
        
        self.layers = []
        self.layers.append(Convoluntion(self.params['W1'],self.params['b1'],conv_param1['stride'],conv_param1 ['pad'])) # 0
        self.layers.append(Relu())
        self.layers.append(Convoluntion(self.params['W2'],self.params['b2'],conv_param2['stride'],conv_param2['pad'])) # 2
        self.layers.append(Relu())
        self.layers.append(Max_Pooling(pool_h=2, pool_w=2, stride=2)) 
        
        self.layers.append(Convoluntion(self.params['W3'],self.params['b3'],conv_param3['stride'],conv_param3['pad'])) # 5
        self.layers.append(Relu())
        self.layers.append(Convoluntion(self.params['W4'],self.params['b4'],conv_param4['stride'],conv_param4['pad'])) #7
        self.layers.append(Relu())
        self.layers.append(Max_Pooling(pool_h=2, pool_w=2, stride=2))
        
        self.layers.append(Affine(self.params['W5'],self.params['b5'])) # 10
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W6'],self.params['b6'])) # 12
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W7'],self.params['b7'])) # 14
        self.layers.append(SoftmaxWithLoss())
        
    def forward(self,x):

        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    def gradient(self,t):
        
        dout = 1

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        dout = tmp_layers[0].backward(t,dout)
        for layer in tmp_layers[1:]:
            dout = layer.backward(dout)
            
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12,14)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db 
            
        return grads 
    
    def accuracy(self, y, t):
        acc = 0.0
        length =y.shape[0]

        y_max = np.argmax(y,axis=1)
        t_max = np.argmax(t,axis=1)

        acc = np.sum(y_max==t_max)/length

        return acc

    def loadparams(self,params):
        i = 0
        keys = list(params.keys())
        for layer in self.layers:
            if isinstance(layer,Convoluntion) or isinstance(layer,Affine):
                layer.W = params[keys[i]]
                self.params[keys[i]] = params[keys[i]]
                i+=1
                layer.b = params[keys[i]]
                self.params[keys[i]] = params[keys[i]]
                i+=1


    
class NewLeNet():
    
    def __init__(self, input_dim=(1, 28, 28),hidden_size=50, output_size=10):
        

        pre_node_nums = np.array([1*3*3,6*3*3,16*3*3,32*3*3,120*7*7,512,256,128])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)
        
        conv_param1 = {'filter_num':6, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param4 = {'filter_num':120, 'filter_size':3, 'pad':1, 'stride':1}
        
        self.params = {}
        
        self.params['W1'] = weight_init_scales[0] * np.random.randn(conv_param1['filter_num'],input_dim[0],conv_param1['filter_size'],conv_param1['filter_size'])
        self.params['b1'] = np.zeros(conv_param1['filter_num'])
        
        self.params['W2'] = weight_init_scales[1] * np.random.randn(conv_param2['filter_num'],conv_param1['filter_num'],conv_param2['filter_size'],conv_param2['filter_size'])
        self.params['b2'] = np.zeros(conv_param2['filter_num'])
        
        self.params['W3'] = weight_init_scales[2] * np.random.randn(conv_param3['filter_num'],conv_param2['filter_num'],conv_param3['filter_size'],conv_param3['filter_size'])
        self.params['b3'] = np.zeros(conv_param3['filter_num'])
        
        self.params['W4'] = weight_init_scales[3] * np.random.randn(conv_param4['filter_num'],conv_param3['filter_num'],conv_param4['filter_size'],conv_param4['filter_size'])
        self.params['b4'] = np.zeros(conv_param4['filter_num'])

        self.params['W5'] = weight_init_scales[4] * np.random.randn(120*7*7, 512)
        self.params['b5'] = np.zeros(512)
        

        self.params['W6'] = weight_init_scales[5] * np.random.randn(512, 256)
        self.params['b6'] = np.zeros(256)

        self.params['W7'] = weight_init_scales[6] * np.random.randn(256, 128)
        self.params['b7'] = np.zeros(128)

        self.params['W8'] = weight_init_scales[7] * np.random.randn(128,10)
        self.params['b8'] = np.zeros(10)
        
        
        self.layers = []
        self.layers.append(Convoluntion(self.params['W1'],self.params['b1'],conv_param1['stride'],conv_param1 ['pad'])) # 0
        self.layers.append(Relu())
        self.layers.append(Convoluntion(self.params['W2'],self.params['b2'],conv_param2['stride'],conv_param2['pad'])) # 2
        self.layers.append(Relu())
        self.layers.append(Max_Pooling(pool_h=2, pool_w=2, stride=2)) 
        
        self.layers.append(Convoluntion(self.params['W3'],self.params['b3'],conv_param3['stride'],conv_param3['pad'])) # 5
        self.layers.append(Relu())
        self.layers.append(Convoluntion(self.params['W4'],self.params['b4'],conv_param4['stride'],conv_param4['pad'])) # 7
        self.layers.append(Relu())
        self.layers.append(Max_Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params['W5'],self.params['b5'])) # 10
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W6'],self.params['b6'])) # 12
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W7'],self.params['b7'])) # 14
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W8'],self.params['b8']))
        self.layers.append(SoftmaxWithLoss())
        
    def forward(self,x):

        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    def gradient(self,t):
        
        dout = 1

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        dout = tmp_layers[0].backward(t,dout)
        for layer in tmp_layers[1:]:
            dout = layer.backward(dout)
            
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12,14,16)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db 
            
        return grads 
    
    def accuracy(self, y, t):
        acc = 0.0
        length =y.shape[0]

        y_max = np.argmax(y,axis=1)
        t_max = np.argmax(t,axis=1)

        acc = np.sum(y_max==t_max)/length

        return acc

    def loadparams(self,params):
        i = 0
        keys = list(params.keys())
        for layer in self.layers:
            if isinstance(layer,Convoluntion) or isinstance(layer,Affine):
                layer.W = params[keys[i]]
                self.params[keys[i]] = params[keys[i]]
                i+=1
                layer.b = params[keys[i]]
                self.params[keys[i]] = params[keys[i]]
                i+=1