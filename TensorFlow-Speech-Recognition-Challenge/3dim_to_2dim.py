import numpy as np

arr = np.arange(24).reshape(2,3,4)#(sample,nstep,ndim)をイメージ
#目標は(nstep*nsample,ndim)

arr=arr.reshape(-1,4)#(-1,ndim)でオッケー(sample*nstep,ndim)

print(arr)
print(arr.shape)