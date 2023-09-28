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





def gen_train_continual(power_map_all,fem_engine, numofcontinualtrain,n,load_old):
    if load_old:
        with open('./pickle/train_data.pickle', "rb") as f:
            train_data =pickle.load(f)
            assert len(train_data[0])==numofcontinualtrain
            return train_data


    goldengen_new = fem_engine
    idx_= np.arange(len(power_map_all))
    idx_=np.random.permutation(idx_)[:numofcontinualtrain]
    power_train = []
    thermal_train = []


    for idx,poweridx_for_train in enumerate(idx_):
        print('INFO: Generating train samples {}/{}'.format(idx+1, len(idx_)))
        power_curr              = power_map_all[poweridx_for_train]
        power_train.append(power_curr)
        power_plot              = power_curr.cpu().clone().reshape(n,n)
        # put power plot into 0,1 range
        minpower = power_curr.min()
        maxpower = power_curr.max()
        power_fem = (power_curr-minpower)/(maxpower-minpower)



        goldengen_new.p[-1,:,:] = power_fem.reshape(n,n).cpu().numpy()
        golden_t                = goldengen_new.gen()
        vmin = golden_t.min()
        vmax = golden_t.max()
        golden_t = (golden_t-vmin)/(vmax-vmin)
        golden_t = torch.from_numpy(golden_t)
        thermal_train.append(golden_t)
        # if idx==20:
        #     abs_err,maps = comparepredict_golden(golden_t, golden_t)
        #     plot_im_multi(power_plot.cpu(), golden_t, golden_t, abs_err, 'all')
        #     exit()


    train_data = [power_train,   thermal_train]
    with open('./pickle/train_data.pickle', "wb") as f:
        pickle.dump(train_data, f,protocol=pickle.HIGHEST_PROTOCOL)

    return train_data






def gen_train_continual_v2(power_map_all,fem_engine, numofcontinualtrain,n,load_old):
    if load_old:
        with open('./pickle/train_data.pickle', "rb") as f:
            train_data =pickle.load(f)
            assert len(train_data[0])==numofcontinualtrain
            return train_data


    goldengen_new = fem_engine
    idx_= np.arange(len(power_map_all))
    idx_=np.random.permutation(idx_)[:numofcontinualtrain]
    power_train = []
    thermal_train = []


    for idx,poweridx_for_train in enumerate(idx_):
        print('INFO: Generating train samples {}/{}'.format(idx+1, len(idx_)))
        power_curr              = power_map_all[poweridx_for_train]
        power_train.append(power_curr)
        power_plot              = power_curr.cpu().clone().reshape(n,n)
        # put power plot into 0,1 range
        minpower = power_curr.min()
        maxpower = power_curr.max()
        power_fem = (power_curr-minpower)/(maxpower-minpower)



        goldengen_new.p[-1,:,:] = power_fem.reshape(n,n).cpu().numpy()
        golden_t                = goldengen_new.gen()
        vmin = golden_t.min()
        vmax = golden_t.max()
        golden_t = (golden_t-vmin)/(vmax-vmin)
        golden_t = torch.from_numpy(golden_t)
        thermal_train.append(golden_t)
        # if idx==20:
        #     abs_err,maps = comparepredict_golden(golden_t, golden_t)
        #     plot_im_multi(power_plot.cpu(), golden_t, golden_t, abs_err, 'all')
        #     exit()


    train_data = [power_train,   thermal_train]
    with open('./pickle/train_data.pickle', "wb") as f:
        pickle.dump(train_data, f,protocol=pickle.HIGHEST_PROTOCOL)

    return train_data





def plot_im_multi(power, golden_t, predict_t, abs_err, save_name):
    
    fig,ax = plt.subplots(3,2)
    vmin = 0
    vmax = 1
    # im = ax.imshow(plot_data , cmap = 'jet',vmin=vmin,vmax=vmax)
    ax[0,0].imshow(power,cmap = 'jet',vmin=power.min().item(),vmax=power.max().item())
    ax[0,0].axis("off")
    ax[1,0].set_visible(False)
    ax[2,0].set_visible(False)


    mean_arr = []
    for i in range(1):
        ax[0,i+1].axis('off')
        ax[1,i+1].axis('off')
        ax[2,i+1].axis('off')

        ax[0,i+1].imshow(golden_t[i,:,:],cmap = 'jet',vmin=vmin,vmax=vmax)
        ax[1,i+1].imshow(predict_t[i,:,:],cmap = 'jet',vmin=vmin,vmax=vmax)
        ax[2,i+1].imshow(abs_err[i,:,:],cmap = 'jet',vmin=vmin,vmax=vmax)

        meanarr = np.mean(abs_err[i,:,:])
        mean_arr.append(round(meanarr,4))


    # print(mean_arr)
    ax[0,0].set_title("power map", pad=20)
    ax[0,1].set_title("layer 1", pad=20)
    # ax[0,2].set_title("layer 2", pad=20)
    # ax[0,3].set_title("layer 3", pad=20)

    ax[2,1].set_title("mean_err= {}".format(mean_arr[0]), pad=20)
    # ax[2,2].set_title("mean_err= {}".format(mean_arr[1]), pad=20)
    # ax[2,3].set_title("mean_err= {}".format(mean_arr[2]), pad=20)

    # ax[0,1].set_title("golden t, layer 3", pad=20)
    # ax[0,2].set_title("golden t, layer 2", pad=20)
    # ax[0,3].set_title("golden t, layer 1", pad=20)
    # ax[1,1].set_title("predict t, layer 3", pad=20)
    # ax[1,2].set_title("predict t, layer 2", pad=20)
    # ax[1,3].set_title("predict t, layer 1", pad=20)
    # ax[2,1].set_title("abs err, layer 3", pad=20)
    # ax[2,2].set_title("abs err, layer 2", pad=20)
    # ax[2,3].set_title("abs err, layer 1", pad=20)


    # fig.colorbar(tmp)
    fig.set_figheight(12)
    fig.set_figwidth(16)
    fig.savefig(save_name, bbox_inches='tight',dpi=100)
    # plt.show()
    plt.close()




def generate_k_map_fortrain(numoftrain=10,batchsize=5, grid=20, nz=3, data01= True, skipcood= False):
    #====================================================================================================
    # generate multiple power_map here
    #====================================================================================================
    power_map_l = []
    #====================================================================================================
    for i in range(numoftrain):

        complex_ = random.randint(5, 7)
        power_map_np = gr.gaussian_random_field(alpha = complex_, size = grid*2)
        max_power = power_map_np.max()
        min_power = power_map_np.min()
        power_map_np= (power_map_np-min_power)/(max_power-min_power)
        assert power_map_np.max()<=1 and power_map_np.min()>=0
        power_map_np = power_map_np*2-1
        #====================================================================================================
        #====================================================================================================
        #====================================================================================================
        x = random.randint(0, grid*2-grid)
        y = random.randint(0, grid*2-grid)
        power_map_np = power_map_np[x:x+grid,y:y+grid]
        max_power = power_map_np.max()
        min_power = power_map_np.min()
        power_map_np= (power_map_np-min_power)/(max_power-min_power)
        assert power_map_np.max()<=1 and power_map_np.min()>=0
        # power_map_np = power_map_np*2-1
        # power_map_np = power_map_np/10
        power_map_flatten = power_map_np.reshape(grid*grid)
        # count=0
        # for x_idx in range(grid):
        #     for y_idx in range(grid):
        #         power_map_flatten[count] = power_map_np[x_idx,y_idx]
        #         count+=1


        power_map = torch.tensor(power_map_flatten).double().to(device) #400
        # plot_im(plot_data=power_map[0,:].clone().reshape((20,20)).cpu(), title="power map", save_name='power_map{}.png'.format(i))
        # plot_im(plot_data=power_map[:].clone().reshape((grid,grid)).cpu(), title="power map", save_name='power_map{}.png'.format(i))
        # exit()
        power_map_l.append(power_map)
        
    #====================================================================================================
    # generater power map coordinate
    #====================================================================================================
    torch_nodes_x = None 
    torch_nodes_y = None 
    torch_nodes_z = None 
    
    if not skipcood:
        torch_nodes_x,torch_nodes_y,torch_nodes_z,power_edge,_ = generate_graph_xyz01(numx=grid,numy=grid,numz=1,numoftrain=1, addrandom=False,data01=data01, grid=grid)
        if data01:
            delta = 1./grid
        else:
            delta = 2./grid
        z_cood_top = nz*delta
        torch_nodes_z = torch.ones(torch_nodes_z.shape).double().to(device)*z_cood_top
        #====================================================================================================
        
    return power_map_l, torch_nodes_x,torch_nodes_y,torch_nodes_z, power_edge






def generate_power_map_fortrain(numoftrain=10,batchsize=5, grid=20, nz=3, data01= True, skipcood= False):
    #====================================================================================================
    # generate multiple power_map here
    #====================================================================================================
    power_map_l = []
    #====================================================================================================
    for i in range(numoftrain):

        complex_ = random.randint(5, 6)
        power_map_np = gr.gaussian_random_field(alpha = complex_, size = grid*2)
        max_power = power_map_np.max()
        min_power = power_map_np.min()
        power_map_np= (power_map_np-min_power)/(max_power-min_power)
        assert power_map_np.max()<=1 and power_map_np.min()>=0
        power_map_np = power_map_np*2-1
        #====================================================================================================
        #====================================================================================================
        #====================================================================================================
        x = random.randint(0, grid*2-grid)
        # y = random.randint(0, grid*2-grid)
        # power_map_np = power_map_np[x:x+grid,y:y+grid]
        power_map_np = power_map_np[x:x+grid,:]
        max_power = power_map_np.max()
        min_power = power_map_np.min()
        power_map_np= (power_map_np-min_power)/(max_power-min_power)
        assert power_map_np.max()<=1 and power_map_np.min()>=0
        if data01:
            pass
        else :
            power_map_np = power_map_np*2-1
        # power_map_np = power_map_np*0.1
        power_map_flatten = power_map_np.reshape(2*grid*grid)
        # count=0
        # for x_idx in range(grid):
        #     for y_idx in range(grid):
        #         power_map_flatten[count] = power_map_np[x_idx,y_idx]
        #         count+=1
        power_map = torch.tensor(power_map_flatten).double().to(device) #400
        # plot_im(plot_data=power_map[0,:].clone().reshape((20,20)).cpu(), title="power map", save_name='power_map{}.png'.format(i))
        # if i==11:
        #     plot_im(plot_data=power_map[:].clone().reshape((grid,2*grid)).cpu(), title="power map", save_name='power_map{}.png'.format(0))
        #     exit()
        power_map_l.append(power_map)
    # exit()
    #====================================================================================================
    # generater power map coordinate
    #====================================================================================================
    torch_nodes_x = None 
    torch_nodes_y = None 
    torch_nodes_z = None 
    
    if not skipcood:
        torch_nodes_x,torch_nodes_y,torch_nodes_z,power_edge,_ = generate_graph_xyz01(numx=grid,numy=grid,numz=1,numoftrain=1, addrandom=False,data01=data01, grid=grid)
        if data01:
            delta = 1./grid
        else:
            delta = 2./grid
        z_cood_top = nz*delta
        torch_nodes_z = torch.ones(torch_nodes_z.shape).double().to(device)*z_cood_top
        #====================================================================================================
        
    return power_map_l, torch_nodes_x,torch_nodes_y,torch_nodes_z, power_edge




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



                    
                




            





    



def open_xyz(xyz):
    return xyz[0],xyz[1],xyz[2]




def generate_bc(numx,numy,numz,data01= False):
    grid= numx
    #====================================================================================================
    x_cood_lr = []
    y_cood_lr = []
    z_cood_lr = []
    x_cood_tb = []
    y_cood_tb = []
    z_cood_tb = []
    #====================================================================================================
    # left
    #====================================================================================================
    torch_nodes_x_l,torch_nodes_y_l,torch_nodes_z_l,edge_l,_ = generate_graph_xyz01(numx=numx,numy=1,numz=numz,numoftrain=1, addrandom=False,data01=data01, grid=grid)
    if data01:
        torch_nodes_y_l = torch.zeros(torch_nodes_y_l.shape).double().to(device)
    else:
        torch_nodes_y_l = torch.zeros(torch_nodes_y_l.shape).double().to(device)-1

         
    x_cood_lr.append(torch_nodes_x_l)
    y_cood_lr.append(torch_nodes_y_l)
    z_cood_lr.append(torch_nodes_z_l)
    #====================================================================================================
    # right
    #====================================================================================================
    torch_nodes_x_r,torch_nodes_y_r,torch_nodes_z_r,edge_r,_ = generate_graph_xyz01(numx=numx,numy=1,numz=numz,numoftrain=1, addrandom=False,data01=data01, grid=grid)
    torch_nodes_y_r = torch.ones(torch_nodes_y_r.shape).double().to(device)
    x_cood_lr.append(torch_nodes_x_r)
    y_cood_lr.append(torch_nodes_y_r)
    z_cood_lr.append(torch_nodes_z_r)
    
    edge_r = edge_r+numx*numz
    edge_lr=torch.cat([edge_l, edge_r], dim=1)
    # print(edge_l.shape)
    # print(edge_r.shape)
    # print(edge_lr.shape)




    #====================================================================================================
    # top
    #====================================================================================================
    torch_nodes_x_t,torch_nodes_y_t,torch_nodes_z_t,edge_t,_ = generate_graph_xyz01(numx=1,numy=numy,numz=numz,numoftrain=1, addrandom=False,data01=data01,grid=grid)
    if data01:
        torch_nodes_x_t = torch.zeros(torch_nodes_x_t.shape).double().to(device)
    else:
        torch_nodes_x_t = torch.zeros(torch_nodes_x_t.shape).double().to(device)-1
    x_cood_tb.append(torch_nodes_x_t)
    y_cood_tb.append(torch_nodes_y_t)
    z_cood_tb.append(torch_nodes_z_t)
    # print(x_cood[-1])
    # print(y_cood[-1])
    # print(z_cood[-1])
    #====================================================================================================
    # bottom 
    #====================================================================================================
    torch_nodes_x_b,torch_nodes_y_b,torch_nodes_z_b,edge_b,_ = generate_graph_xyz01(numx=1,numy=numy,numz=numz,numoftrain=1, addrandom=False,data01=data01, grid=grid)
    torch_nodes_x_b = torch.ones(torch_nodes_x_b.shape).double().to(device)
    x_cood_tb.append(torch_nodes_x_b)
    y_cood_tb.append(torch_nodes_y_b)
    z_cood_tb.append(torch_nodes_z_b)

    edge_b = edge_b+numy*numz
    edge_tb=torch.cat([edge_t, edge_b], dim=1)
    #====================================================================================================
    x_bc_lr=torch.cat(x_cood_lr).double().to(device)
    y_bc_lr=torch.cat(y_cood_lr).double().to(device)
    z_bc_lr=torch.cat(z_cood_lr).double().to(device)
    x_bc_tb=torch.cat(x_cood_tb).double().to(device)
    y_bc_tb=torch.cat(y_cood_tb).double().to(device)
    z_bc_tb=torch.cat(z_cood_tb).double().to(device)
    #====================================================================================================
    # print(x_bc.shape)
    # print(y_bc.shape)
    # print(z_bc.shape)
    #====================================================================================================
    # very bottom boundary condition
    #====================================================================================================
    torch_nodes_x_bot,torch_nodes_y_bot,torch_nodes_z_bot,edge_bot,_ = generate_graph_xyz01(numx=numx,numy=numy,numz=1,numoftrain=1, addrandom=False,data01=data01, grid=grid)
    torch_nodes_z_bot = torch.zeros(torch_nodes_z_bot.shape).double().to(device)
    #====================================================================================================
    return x_bc_lr,y_bc_lr,z_bc_lr,x_bc_tb,y_bc_tb,z_bc_tb,torch_nodes_x_bot,torch_nodes_y_bot,torch_nodes_z_bot, edge_lr, edge_tb, edge_bot









    


def generate_graph_xyz01(numx,numy,numz,numoftrain,addrandom,data01=True,debug=False,grid=20):
    
    direction_map = dict()
    direction_map[0]=(0,1,0)  #east,0
    direction_map[1]=(0,-1,0) #west,1
    direction_map[2]=(1,0,0)  #north,2
    direction_map[3]=(-1,0,0) #south,3
    direction_map[4]=(0,0,1)  #top,4
    direction_map[5]=(0,0,-1) #bottom,5

    torch_nodes_x = torch.zeros((numx*numy*numz,1))
    torch_nodes_y = torch.zeros((numx*numy*numz,1))
    torch_nodes_z = torch.zeros((numx*numy*numz,1))
    #====================================================================================================
    nodeid = 0
    edges_l = []
    #====================================================================================================
    boundary_idx =[]  
    
    # range01 = False
    # range01 = True 
    if data01:
        xstart = 0.5/grid
        ystart = 0.5/(grid+grid)
        # zstart = 0.5/numz
        zstart = 0.5/grid
        deltax =1./grid
        deltay =1./(grid+grid)
        deltaz =1./grid
    else:
        xstart = -1+1./grid
        ystart = -1+1./(grid+grid)
        # zstart = -1+1./numz
        zstart =1./grid
        deltax =2./grid
        deltay =2./(grid+grid)
        deltaz =2./grid

    center_cood_x = xstart
    center_cood_y = ystart
    center_cood_z = zstart
    #====================================================================================================
    for z_idx in range(numz):
        for x_idx in range(numx):
            for y_idx in range(numy):
                
                # assert 0<=center_cood_x <=1
                # assert 0<=center_cood_y <=1 
                # assert 0<=center_cood_z <=1

                torch_nodes_x[nodeid,0] = center_cood_x
                torch_nodes_y[nodeid,0] = center_cood_y
                torch_nodes_z[nodeid,0] = center_cood_z
                center_cood_y+=deltay

                
                centernodeid = id_(x_idx, y_idx, z_idx, numx, numy,numz )
                #====================================================================================================

                for direction in range(6):
                    assert direction in direction_map.keys()
                    neighbour_x_idx= (direction_map[direction][0]+x_idx)
                    neighbour_y_idx= (direction_map[direction][1]+y_idx)
                    neighbour_z_idx= (direction_map[direction][2]+z_idx)

                    if checkneighbours(neighbour_x_idx,neighbour_y_idx,neighbour_z_idx,numx,numy,numz):
                        neighbour_id = id_(neighbour_x_idx, neighbour_y_idx,neighbour_z_idx, numx, numy,numz)
                        edges_l.append([centernodeid, neighbour_id])

                nodeid+=1
            center_cood_x+=deltax
            center_cood_y=ystart
            
        center_cood_z+=deltaz
        center_cood_x=xstart
        center_cood_y=ystart
    #====================================================================================================
    if addrandom:
    #====================================================================================================
        torch_nodes_x_all = []
        torch_nodes_y_all = []
        torch_nodes_z_all = []
        for i in range(numoftrain):
            randomchange_x = (-deltax/2 - deltax/2) * torch.rand(torch_nodes_x.shape) + deltax/2
            randomchange_y = (-deltay/2 - deltay/2) * torch.rand(torch_nodes_y.shape) + deltay/2
            randomchange_z = (-deltaz/2 - deltaz/2) * torch.rand(torch_nodes_z.shape) + deltaz/2
            
            driftx=torch_nodes_x+randomchange_x
            drifty=torch_nodes_y+randomchange_y
            driftz=torch_nodes_z+randomchange_z
            torch_nodes_x_all.append(driftx.clone())
            torch_nodes_y_all.append(drifty.clone())
            torch_nodes_z_all.append(driftz.clone())
            
        torch_nodes_x = torch.cat(torch_nodes_x_all, dim=0)
        torch_nodes_y = torch.cat(torch_nodes_y_all, dim=0)
        torch_nodes_z = torch.cat(torch_nodes_z_all, dim=0)
        # print(torch_nodes_x.shape)
        # print(torch_nodes_y.shape)
        # print(torch_nodes_z.shape)
    #====================================================================================================
    else:
    #====================================================================================================
        torch_nodes_x = torch.tile(torch_nodes_x,(numoftrain,1))
        torch_nodes_y = torch.tile(torch_nodes_y,(numoftrain,1))
        torch_nodes_z = torch.tile(torch_nodes_z,(numoftrain,1))
    #====================================================================================================
    edges = torch.tensor(edges_l).T
    numofnode = numx*numy*numz
    edges_list = []
    for idx in range(numoftrain):
        newedges=edges.clone()+numofnode*idx
        edges_list.append(newedges)
    edges = torch.cat(edges_list,dim=1).to(device)
    #====================================================================================================
    torch_nodes_x = torch_nodes_x.double().to(device)
    torch_nodes_y = torch_nodes_y.double().to(device)
    torch_nodes_z = torch_nodes_z.double().to(device)
    # edges = torch.tensor(edges_l).T.to(device)
    #====================================================================================================
    return torch_nodes_x,torch_nodes_y,torch_nodes_z,edges,boundary_idx



def duplicate_edge(edges, numoftrain, numofnode):
    edges_list = []
    for idx in range(numoftrain):
        newedges=edges.clone()+numofnode*idx
        edges_list.append(newedges)
    edges = torch.cat(edges_list,dim=1).to(device)
    return edges

