import numpy as np
import h5py

def RELU(g):
    return np.where(g>0,g,0)

def res(w,x,b):
    return (np.matmul(w.transpose(),x)+b) 

def sigmoid(g):
    return 1/(1+np.exp(-1*g)) 

def fowrardPropagation(Weights,Biases,Inp,R,V,numLayers):
    for i in range(numLayers):
        R[i] = res(Weights[i],Inp[i],Biases[i])
        V[i] = RELU(R[i])
        if (i+1) < numLayers:
            Inp[i+1] = V[i]
        elif i == numLayers-1:
            V[i] = sigmoid(R[i])




def main():
    f=h5py.File('train_catvnoncat.h5','r')
    imgs=np.array(f['train_set_x'])
    vals=np.array(f['train_set_y'])
    f.close()
    imgs=np.reshape(imgs,(imgs.shape[0],imgs.shape[1]*imgs.shape[2]*imgs.shape[3]),order='C').transpose()/255 # NUMPY ARRAY CORRESPONDING TO X Standardized
    vals=np.reshape(vals,(1,vals.shape[0]),order='C') # row vector
    numExamples=imgs.shape[1]
    W1=np.random.rand(imgs.shape[0],5) #Weights for layer 1 with 5 hidden units
    W2=np.random.rand(W1.shape[1],5) #Weights for layer 2 with 5 hidden units
    W3=np.random.rand(W2.shape[1],5) #Weights for layer 3 with 5 hidden units
    W4=np.random.rand(W3.shape[1],3) #Weights for layer 4 with 3 hidden units
    Wout=np.random.rand(W4.shape[1],1) #Weights for output layer with 1 unit
    Ws=(W1,W2,W3,W4,Wout)
    B1=np.zeros((W1.shape[1],1)) #Biases for layer 1 
    B2=np.zeros((W2.shape[1],1)) #Biases for layer 2
    B3=np.zeros((W3.shape[1],1)) #Biases for layer 3
    B4=np.zeros((W4.shape[1],1)) #Biases for layer 4
    Bout=np.zeros((Wout.shape[1],1)) #Biases for layer out
    Bs=(B1,B2,B3,B4,Bout)
    numLayers=len(Ws)
    #Inputs res and vals for the 5 layers
    Inps=[imgs,0,0,0,0]
    res=[0,0,0,0,0]
    vs=[0,0,0,0,0]
    dWs=[0,0,0,0,0]
    dBs=[0,0,0,0,0]
    dInps=[0,0,0,0,0] 
    for i in range(5000):
        dWs[numLayers-1]=np.matmul(Inps[numLayers-1],(vs[numLayers-1]-vals).transpose())/numExamples
        dBs[numLayers-1]=np.sum((vs[numLayers-1]-vals))/numExamples
        dInps[numLayers-1]=Ws[numLayers-1]*(vs[numLayers-1]-vals)/numExamples #(3,209)
        dinp_res=np.where(res[numLayers-2]>0,res[numLayers-2],0)
        dres=dinp_res*dInps[numLayers-1] 








 
main()