import numpy as np
import h5py

def res(w,x,b):
    return (np.matmul(w.transpose(),x)+b) #(1,209)
def sigmoid(g):
    return 1/(1+np.exp(-1*g)) #(1,209)
def loss(v,p):
    return (-1*(v*np.log(p)+(1-v)*np.log((1-p)))) #(1,209)
def cost(l,m):
    return np.sum(l)/m #number

#basic differentiation not used
def gradientchangew(i,r,v,p,m):
    er=((np.exp(-r))/((1+np.exp(-r))**2)) #(1,209)
    dp=i*er #(12288,209)
    v_p=((1-v)/(1-p)-v/p).transpose() #(209,1)
    return np.matmul(dp,v_p)/m #(12288,1)

def main():
    f=h5py.File('train_catvnoncat.h5','r')
    imgs=np.array(f['train_set_x'])
    vals=np.array(f['train_set_y'])
    f.close()
    imgs=np.reshape(imgs,(209,12288),order='C').transpose()/255 # NUMPY ARRAY CORRESPONDING TO X Standardized
    vals=np.reshape(vals,(1,209),order='C') # row vector
    w=np.random.rand(12288,1)
    b=np.random.random()   
    for i in range(5000):
        r=res(w,imgs,b)
        a=sigmoid(r)
        dw=np.matmul(imgs,(a-vals).transpose())/209
        #dw=gradientchangew(imgs,r,vals,a,209) not used because divide by zero error in gradientchange
        db=np.sum(a-vals)/209
        w=w-0.01*dw
        b=b-0.01*db
    
    np.savez_compressed('./wb.npz',w,b)


main()

 
