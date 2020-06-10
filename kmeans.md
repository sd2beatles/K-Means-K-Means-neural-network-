```python
import numpy as np
from scipy import stats
import copy
import sys
from operator import itemgetter





class Kmean(object):
    def __init__(self,data,k,max_iteration=10):
        self.data=data
        self.k=k
        self.n=data.shape[0]
        self.m=data.shape[1]
        self.max_iteration=max_iteration
    
    def train(self):
        
        start=stats.norm.rvs(loc=0,scale=1,size=(self.m,self.k))
        maxima=self.data.max(axis=0)[:,np.newaxis]
        minima=self.data.min(axis=0)[:,np.newaxis]
        centers=start*(maxima-minima)+maxima
        old_centers=0
        count=0
       
        while np.sum(np.sum(centers-old_centers))!=0 and count<self.max_iteration:
            old_centers=centers.copy()
            centers=np.zeros((self.m,self.k))
            count+=1
           
            distance=np.zeros((self.k,self.n))
            for j in range(self.k):
                d=np.sum((self.data-old_centers[:,j])**2,axis=1)
                index,dist=zip(*sorted(enumerate(d),key=itemgetter(1)))
                distance[j]=dist
                index=index[:self.k]
                p=np.mean(self.data[index,:],axis=0)
                centers[:,j]=p
        return distance
    
    def classficiation(self):
        distance=self.train()
        distance=np.argsort(distance,axis=0)
        labels=np.zeros(self.n)
        count=1
        for i in range(self.k):
            s=np.where(distance[i]==0,True,False)
            labels[s]+=count
            count+=1
            
        return labels
 
def main():
    data=np.random.randint(1,20,size=(100,10))
    distance=Kmean(data,3).classficiation()
    print(distance)
    
if __name__=='__main__':
    main()
    ```
