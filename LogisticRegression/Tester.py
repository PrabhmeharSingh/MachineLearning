import numpy as np
import h5py
import Trainer as T
def main():
    f=h5py.File('test_catvnoncat.h5','r')
    imgs=np.array(f['test_set_x'])
    vals=np.array(f['test_set_y'])
    f.close()
    imgs=np.reshape(imgs,(imgs.shape[0],12288),order='C').transpose()
    vals=np.reshape(vals,(1,vals.shape[0]),order='C')
    wb=np.load('./wb.npz')
    w=wb['arr_0']
    b=wb['arr_1']
    wb.close()
    prs=T.sigmoid(T.res(w,imgs,b))
    print(prs)
    #print('Cost = ',T.cost(T.loss(vals,prs),imgs.shape[0])) Divide by zero error in res
    prs=np.where(prs>0.5,1,0)
    print(prs)
    print(vals)
    
    #LOOP TO BE REMOVED IF POSSIBLE
    #eq=len(np.where((prs==1 & vals==1) | (prs==0 & vals==0)))
    eq=0
    for i in range(vals.shape[1]):
        if vals[0][i]==prs[0][i]:
            eq+=1
    print('Correctness Percentage = ',eq/vals.shape[1])
    print(eq,vals.shape[1])

main()