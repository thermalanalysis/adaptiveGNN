import gaussian_random_fields as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import random
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def comparepredict_golden(predict, golden):
    result = np.absolute(np.array(predict) - np.array(golden))
    # maps = np.mean(np.absolute((np.array(predict) - np.array(golden))/np.array(golden)  ))



    return result,None





def checkneighbours(idxx,idxy,idxz,numx,numy,numz):
    if idxx<0 or idxx>=numx:
        return False
    if idxy<0 or idxy>=numy:
        return False
    if idxz<0 or idxz>=numz:
        return False
    return True


def id_(idxx, idxy, idxz, numx, numy, numz):
    return idxz*(numx*numy)+ idxx*(numy)+ idxy

    
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)



def plot_im(plot_data, title, save_name, vmin=None,vmax=None):
    
    fig,ax = plt.subplots()
    if vmin is None:
        im = ax.imshow(plot_data , cmap = 'jet')
    else:
        im = ax.imshow(plot_data , cmap = 'jet',vmin=vmin,vmax=vmax)
    ax.set_title(title, pad=20)
    fig.colorbar(im)
    fig.savefig(save_name, bbox_inches='tight',dpi=100)
    plt.show()
    plt.close()





def find_idx(y,deltay,n,data01):
    idx_k0 = []
    idx_k1 = []
    count=0
    for i in range(len(y)):
        value = y[i,:].item()
        if data01:
            threshold=0.5
        else:
            threshold=0
             
        if value<threshold:
            idx_k0.append(count)
        else:
            idx_k1.append(count)
        count+=1
    return idx_k0,idx_k1





def duplicate_power(power_l, nodecount , idx=None, k_l= None):
    power_out=[]
    for i, power in enumerate(power_l):
        if idx is None:
            if k_l is not None:
                k=k_l[i]
                # print(power.shape)
                # print(k.shape)
                # exit()
            power_curr = torch.tile(power, (nodecount,1))
            # assert torch.equal(power_curr[0,:],power_curr[1,:])
            if len(power_curr)>=2:
                assert torch.equal(power_curr[0,:], power_curr[1,:])
        else:
            power = power[idx]
            power_curr = torch.tile(power, (nodecount,1))
        # print(power_curr.shape)
        # assert torch.equal(power_curr[0,:],power_curr[1,:])
        # assert torch.equal(power_curr[0,:],power_curr[-1,:])
        power_out.append(power_curr)

    out = torch.cat(power_out,dim=0)
    return out
    

def smooth_power_map_v2(power_map):
    visited = torch.zeros(power_map.shape)
    row = power_map.shape[0]
    col = power_map.shape[1]
    for r in range(row):
        for c in range(col):
            if power_map[r,c]!=0 and visited[r,c]==0:
                # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooyyyyyyyyyyyy  written by Lingling
                if r-1>=0 and power_map[r-1,c]==0 :
                    power_map[r-1,c]=power_map[r,c]/2
                    visited[r-1,c]=1
                    
                if r+1<row and power_map[r+1,c]==0 :
                    power_map[r+1,c]=power_map[r,c]/2
                    visited[r+1,c]=1

                if c-1>=0 and power_map[r,c-1]==0 :
                    power_map[r,c-1]=power_map[r,c]/2
                    visited[r,c-1]=1

                if c+1<col and power_map[r,c+1]==0 :
                    power_map[r,c+1]=power_map[r,c]/2
                    visited[r,c+1]=1


                # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooyyyyyyyyyyyy  written by Lingling
                if r-1>=0 and c-1>=0 and power_map[r-1,c-1]==0 :
                    power_map[r-1,c-1]=power_map[r,c]/2
                    visited[r-1,c-1]=1
                if r-1>=0 and c+1<col and power_map[r-1,c+1]==0:
                    power_map[r-1,c+1]=power_map[r,c]/2
                    visited[r-1,c+1]=1
                if r+1<row and c+1<col and power_map[r+1,c+1]==0:
                    power_map[r+1,c+1]=power_map[r,c]/2
                    visited[r+1,c+1]=1
                if r+1<row and c-1>=0 and power_map[r+1,c-1]==0:
                    power_map[r+1,c-1]=power_map[r,c]/2
                    visited[r+1,c-1]=1


    return power_map


def smooth_power_map(power_map):
    visited = torch.zeros(power_map.shape)
    row = power_map.shape[0]
    col = power_map.shape[1]
    for r in range(row):
        for c in range(col):
            if power_map[r,c]!=0 and visited[r,c]==0:
                # oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooyyyyyyyyyyyy  written by Lingling
                if r-1>=0 and c-1>=0 and power_map[r-1,c-1]==0 :
                    power_map[r-1,c-1]=power_map[r,c]/2
                    visited[r-1,c-1]=1
                if r-1>=0 and c+1<col and power_map[r-1,c+1]==0:
                    power_map[r-1,c+1]=power_map[r,c]/2
                    visited[r-1,c+1]=1
                if r+1<row and c+1<col and power_map[r+1,c+1]==0:
                    power_map[r+1,c+1]=power_map[r,c]/2
                    visited[r+1,c+1]=1
                if r+1<row and c-1>=0 and power_map[r+1,c-1]==0:
                    power_map[r+1,c-1]=power_map[r,c]/2
                    visited[r+1,c-1]=1
    return power_map
   







