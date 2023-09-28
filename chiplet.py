from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import math

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





class chiplet():
    def __init__(self,xsize,ysize):
        self.xsize   = math.floor(xsize)  #mm
        self.ysize   = math.floor(ysize)  #mm
        self.envT = {0:0,1:0,2:0,3:0}
        self.power_set = False
        self.power_tile_l=[]

        self.min_max_x=MinMaxScaler(feature_range=(0.0, 1.0))
        self.min_max_y=MinMaxScaler(feature_range=(0.0, 1.0))

        self.scaler_set = False


    def add_power_map(self, power_l, power):

        px_start =  self.xsize* power_l[0]
        px_end   =  self.xsize* power_l[1]
        py_start =  self.ysize* power_l[2]
        py_end   =  self.ysize* power_l[3]



        assert 0<=px_start<px_end<=self.xsize
        assert 0<=py_start<py_end<=self.ysize
        self.power_set = True
        power_tile= [[px_start, py_start], [px_end,py_end], power]
        self.power_tile_l.append(power_tile)


    def set_envT(edge,T):
        self.envT[edge]=T


    def generate_res_sample(self,xn_res,yn_res,save_scaler=True, normalize= True):
        xdelta  = self.xsize/xn_res
        ydelta  = self.ysize/yn_res

        x_res_sample = np.linspace(xdelta/2, self.xsize-xdelta/2, xn_res)
        y_res_sample = np.linspace(ydelta/2, self.ysize-ydelta/2, yn_res)
        xy_res_np    = np.array(np.meshgrid(x_res_sample, y_res_sample),dtype=np.float64).T.reshape(-1,2)
        x_res_np     = xy_res_np[:,0].reshape((len(xy_res_np),1))
        y_res_np     = xy_res_np[:,1].reshape((len(xy_res_np),1))

        if save_scaler:
            x_res_tmp = np.linspace(0, self.xsize, 1000)
            y_res_tmp = np.linspace(0, self.ysize, 1000)
            xy_res_np_tmp    = np.array(np.meshgrid(x_res_tmp, y_res_tmp),dtype=np.float64).T.reshape(-1,2)
            x_res_np_tmp     = xy_res_np_tmp[:,0].reshape((len(xy_res_np_tmp),1))
            y_res_np_tmp     = xy_res_np_tmp[:,1].reshape((len(xy_res_np_tmp),1))
            self.min_max_x.fit(x_res_np_tmp) 
            self.min_max_y.fit(y_res_np_tmp)
            self.scaler_set = True

        if normalize:
            assert self.scaler_set
            x_res_np_normalized = self.min_max_x.transform(x_res_np) 
            y_res_np_normalized = self.min_max_y.transform(y_res_np)

        x_res_torch  = torch.tensor(x_res_np).clone().double().to(device).reshape(len(x_res_np),1)
        y_res_torch  = torch.tensor(y_res_np).clone().double().to(device).reshape(len(y_res_np),1)

        x_res_torch.requires_grad=True
        y_res_torch.requires_grad=True

        x_res_torch_normalized  = torch.tensor(x_res_np_normalized).clone().double().to(device).reshape(len(x_res_np_normalized),1)
        y_res_torch_normalized  = torch.tensor(y_res_np_normalized).clone().double().to(device).reshape(len(y_res_np_normalized),1)

        x_res_torch_normalized.requires_grad=True
        y_res_torch_normalized.requires_grad=True

        return x_res_torch, y_res_torch, x_res_torch_normalized, y_res_torch_normalized


    #====================================================================================================
    # 0 top 1 bot 2 left 3 right
    #====================================================================================================
    def generate_bc_sample(self,edge,n_bc,reshape=True, normalize=True):
        assert 0<=edge<=3
        xdelta  = self.xsize/n_bc
        ydelta  = self.ysize/n_bc
        if edge==0:  # top, x=0, y is random
            y_bc_sample = np.linspace(ydelta/2, self.ysize-ydelta/2, n_bc)
            x_bc_sample = np.zeros(y_bc_sample.shape)
        elif edge==1:  # bot, x=self.xsize, y is random
            y_bc_sample = np.linspace(ydelta/2, self.ysize-ydelta/2, n_bc)
            x_bc_sample = np.ones(y_bc_sample.shape)*self.xsize
        elif edge==2:  # left, x is random, y =0
            x_bc_sample = np.linspace(xdelta/2, self.xsize-xdelta/2, n_bc)
            y_bc_sample = np.zeros(x_bc_sample.shape)
        elif edge==3:  # left, x is random, y is ysize
            x_bc_sample = np.linspace(xdelta/2, self.xsize-xdelta/2, n_bc)
            y_bc_sample = np.ones(x_bc_sample.shape)*self.ysize

        if normalize:
            assert self.scaler_set
            x_bc_sample = self.min_max_x.transform(x_bc_sample.reshape((n_bc,1))).squeeze() 
            y_bc_sample = self.min_max_y.transform(y_bc_sample.reshape((n_bc,1))).squeeze()

        x_bc_torch  = torch.tensor(x_bc_sample).clone().double().to(device)
        y_bc_torch  = torch.tensor(y_bc_sample).clone().double().to(device)
        if reshape:
            x_bc_torch=x_bc_torch.reshape((len(x_bc_torch),1))
            y_bc_torch=y_bc_torch.reshape((len(y_bc_torch),1))

        return x_bc_torch, y_bc_torch



    def generate_bc_sample_multiEdge(self,edge_l,n_bc,normalize=True):
        assert len(edge_l)>0
        x_container=[]
        y_container=[]
        for edge in edge_l:
            x_bc_torch, y_bc_torch = self.generate_bc_sample(edge=edge,n_bc=n_bc,reshape=False,normalize=normalize)
            x_container.append(x_bc_torch)
            y_container.append(y_bc_torch)

        x_bc_torch = torch.cat(x_container).reshape((len(x_container)*len(x_container[-1]),1))
        y_bc_torch = torch.cat(y_container).reshape((len(y_container)*len(y_container[-1]),1))

        x_bc_torch.requires_grad = True
        y_bc_torch.requires_grad = True
        return x_bc_torch, y_bc_torch


    def generate_power_map_res(self,x,y):
        assert self.power_set 
        assert len(x)==len(y)
        power_res_np = np.zeros(len(x))
        for idx in range(len(x)):
            x_ = x[idx].item()
            y_ = y[idx].item()
            power_ = 0
            for powertile in self.power_tile_l:
                px_start = powertile[0][0]
                py_start = powertile[0][1]
                px_end = powertile[1][0]
                py_end = powertile[1][1]
                if px_start<=x_<=px_end and py_start<=y_<=py_end:
                    power_=powertile[2]
            power_res_np[idx]=power_
        power_res_torch = torch.tensor(power_res_np).reshape((len(power_res_np),1))
        power_res_torch = power_res_torch.clone().double().to(device)
        return power_res_torch


    def plot_power_map(self,xn,yn):
        assert self.power_set 
        x_torch, y_torch,_,_ = self.generate_res_sample(xn_res=xn,yn_res=yn,save_scaler=False,normalize=True)
        power_res = self.generate_power_map_res(x_torch, y_torch).reshape((xn,yn)).detach().cpu()
        fig,ax = plt.subplots()
        im = ax.imshow(power_res , cmap = 'jet')
        ax.set_title('Power map', pad=20)
        fig.colorbar(im)
        fig.savefig('power map', bbox_inches='tight',dpi=100)
        plt.show()
        return power_res









# test=chiplet(xsize=2,ysize=5,k=0.1)
# x_res_torch, y_res_torch = test.generate_res_sample(xn_res=20,yn_res=50)
# test.generate_bc_sample_multiEdge(edge_l=[0,2],n_bc=20)

# test.add_power_map(power_l=[0,1,0,5], power= 1) # mm and mw
# power_res = test.generate_power_map_res(x_res_torch, y_res_torch)

# power_res= power_res.reshape((20,50))
# test.plot_power_map(power_res)
# exit()


# exit()



