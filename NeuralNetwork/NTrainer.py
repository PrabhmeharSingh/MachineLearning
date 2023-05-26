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
        

def loss(v,p):
    return (-1*(v*np.log(p)+(1-v)*np.log((1-p)))) #(1,209)
def cost(l,m):
    return np.sum(l)/m #number 


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
    Ws=[W1,W2,W3,W4,Wout]
    B1=np.zeros((W1.shape[1],1)) #Biases for layer 1 
    B2=np.zeros((W2.shape[1],1)) #Biases for layer 2
    B3=np.zeros((W3.shape[1],1)) #Biases for layer 3
    B4=np.zeros((W4.shape[1],1)) #Biases for layer 4
    Bout=np.zeros((Wout.shape[1],1)) #Biases for layer out
    Bs=[B1,B2,B3,B4,Bout]
    numLayers=len(Ws)
    #Inputs res and vals for the 5 layers
    Inps=[imgs,0,0,0,0]
    res=[0,0,0,0,0]
    vs=[0,0,0,0,0]
    dWs=[0,0,0,0,0]
    dBs=[0,0,0,0,0]
    dInps=[0,0,0,0,0] 
    #fowrardPropagation(Ws,Bs,Inps,res,vs,numLayers)
    for x in range(100000):
        fowrardPropagation(Ws,Bs,Inps,res,vs,numLayers)
        dWs[numLayers-1]=np.matmul(Inps[numLayers-1],(vs[numLayers-1]-vals).transpose())/numExamples
        dBs[numLayers-1]=np.sum((vs[numLayers-1]-vals))/numExamples
        dInps[numLayers-1]=Ws[numLayers-1]*(vs[numLayers-1]-vals) #(3,209)
        for i in range(numLayers-2,-1,-1):
            dinp_res=np.where(res[i]>0,1,0)
            dres=dinp_res*dInps[i+1]
            dWs[i]=np.matmul(Inps[i],dres.transpose())/numExamples
            dBs[i]=np.expand_dims(np.sum(dres,axis=1)/numExamples,1)
            dInps[i]=np.matmul(Ws[i],dres)
        for j in range(numLayers):
            Ws[j]=Ws[j]-0.01*dWs[j]
            Bs[j]=Bs[j]-0.01*dBs[j]
        if(x%100==0):
            print(vs[4],vals)
        
        
    









 
main()