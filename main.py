#====================================================================================================
import math
import torch
import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#====================================================================================================
# seed=259
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
#====================================================================================================
from graphmodels import *
from create_golden import *
from util import *
from chiplet import *
from args import *
from scipy.ndimage import gaussian_filter
from datetime import datetime
import traceback
#====================================================================================================
args = parser.parse_args()
numofepoch              = args.numofepoch
inilr                   = args.inilr
train_newgnn            = args.train_newgnn
train_defectnode        = args.train_defectnode
cell_dim                = args.subblock_size_xy
numoftile_perbatch      = args.numofsubblock_xy
generate_new_sample     = args.generate_new_sample
generate_sample_badedge = args.generate_sample_badedge
k1                      = args.k1
k2                      = args.k2
#====================================================================================================
add_bc_dummynode = True 
batchtrainsize = 20
numoftile_dim  = 100
numoftile_x        = numoftile_dim
numoftile_y        = numoftile_dim
#====================================================================================================
# tileidx = torch.tensor([5,6,7,8,9])
# tileidx = torch.tensor([3,4,5])
# tileidx = torch.tensor([3,4])
# tileidx     = torch.arange(start=60, end=65)
tileidx     = torch.arange(start=10, end=13)
tileidx_dir = torch.tensor([1]*len(tileidx)).tolist()
#====================================================================================================
edgeiddict = dict()
edgeiddict[0]=(0,0)
edgeiddict[1]=(0,1)
edgeiddict[2]=(1,0)
edgeiddict[3]=(1,1)




def normalize(t, min_=None, max_=None):
    if min_ is not None and max_ is not None:
        return (t-min_)/(max_-min_)
    return (t-t.min())/(t.max()-t.min())

def comparepredict_golden(predict, golden):
    result = np.absolute(np.array(predict) - np.array(golden))
    return result, None

def plot_im(plot_data, title, save_name,show=True,range01= False,vmin=None, vmax=None):
    fig,ax = plt.subplots()
    # im = ax.imshow(plot_data ,vmin=298, cmap = 'jet')
    if range01:
        im = ax.imshow(plot_data ,vmin=0,vmax=1, cmap = 'jet')
    else :
        if vmin is None or vmax is None:
            im = ax.imshow(plot_data , cmap = 'jet')
        else:
            im = ax.imshow(plot_data , vmin=vmin, vmax=vmax, cmap = 'jet')
             


    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax.set_title(title, pad=20)
    # fig.colorbar(im)
    fig.savefig(save_name, bbox_inches='tight',dpi=100)
    if show:
        plt.show()
    else:
        plt.close()

def generate_power_map_fortrain(numoftrain=10,resx=20, resy=20,complex_=None):
    power_map_l = []
    grid        = max(resx,resy)
    #====================================================================================================
    for i in range(numoftrain):

        if complex_ is None:
            complex_ = random.randint(5, 6)
        power_map_np = gr.gaussian_random_field(alpha = complex_, size = grid*2)
        max_power = power_map_np.max()
        min_power = power_map_np.min()
        power_map_np= (power_map_np-min_power)/(max_power-min_power)
        assert power_map_np.max()<=1 and power_map_np.min()>=0
        power_map_np = power_map_np*2-1
        #====================================================================================================
        x = random.randint(0, grid*2-resx)
        y = random.randint(0, grid*2-resy)
        power_map_np = power_map_np[x:x+resx,y:y+resy]
        max_power = power_map_np.max()
        min_power = power_map_np.min()
        power_map_np= (power_map_np-min_power)/(max_power-min_power)
        power_map = torch.tensor(power_map_np).double().to(device) #grid, grid
        power_map_l.append(power_map)
    #====================================================================================================
    return power_map_l

def analysis_chip(chipinfo):

    chipres   = chipinfo['chipres']
    chipshape = chipinfo['chipshape']
    #====================================================================================================
    totalresx = totalresy = 0
    for colchipidx in range(len(chipres[0])):
        resx_singlechip, resy_singlechip = chipres[0, colchipidx]
        totalresy+= resy_singlechip
    for rowchipidx in range(len(chipres)):
        resx_singlechip, resy_singlechip = chipres[rowchipidx, 0]
        totalresx+= resx_singlechip
    totalresx = (int)(totalresx)
    totalresy = (int)(totalresy)
    totalnumnodes = totalresx * totalresy
    #====================================================================================================
    chipinfo['totalnumofpixel'] = totalnumnodes
    chipinfo['numofpixel_x']    = totalresx
    chipinfo['numofpixel_y']    = totalresy
    #====================================================================================================
    #====================================================================================================
    blockassignment = 0

    tile_dim  = chipinfo['tile_dim']
    kmap_total = []
    numoftile_x = chipinfo['numoftile_x']
    numoftile_y = chipinfo['numoftile_y']

    for x in range(numoftile_x):
        kmap_l  = []
        for y in range(numoftile_y):
            ##====================================================================================================
            kmap_np      = np.ones((tile_dim,tile_dim))
            ##====================================================================================================
            blockid_np   = np.ones((tile_dim,tile_dim))*blockassignment
            blockassignment+=1
            #====================================================================================================
            kmap_l.append(kmap_np)
        kmap_total.append(np.hstack(kmap_l))
        #====================================================================================================
    #====================================================================================================
    kmap_total  = np.vstack(kmap_total).reshape((1, totalresx,totalresy))
    nodeidxs    = np.arange(totalnumnodes).reshape((totalresx, totalresy))
    #====================================================================================================
    edges_l = []
    #====================================================================================================
    # create vertical line
    #====================================================================================================
    start_xidx = 0
    for m in range(numoftile_x):
        start_yidx = tile_dim-1
        for n in range(numoftile_y-1):
            edge_l = []
            for row_idx in range(start_xidx, start_xidx+tile_dim):
                edge_node_id0 = nodeidxs[row_idx,start_yidx].item()
                edge_node_id1 = nodeidxs[row_idx,start_yidx+1].item() 
                edge_l.append([edge_node_id0, edge_node_id1])
                
            start_yidx += tile_dim
            edges_l.append(edge_l)
        start_xidx+= tile_dim
    #====================================================================================================
    # create horizental line
    #====================================================================================================
    start_yidx = 0
    for m in range(numoftile_y):
        start_xidx = tile_dim-1
        for n in range(numoftile_x-1):
            edge_l = []

            for col_idx in range(start_yidx, start_yidx+tile_dim):

                edge_node_id0 = nodeidxs[start_xidx,  col_idx].item()
                edge_node_id1 = nodeidxs[start_xidx+1,col_idx].item() 
                edge_l.append([edge_node_id0, edge_node_id1])

            start_xidx += tile_dim
            edges_l.append(edge_l)
        start_yidx+= tile_dim
    #====================================================================================================
    edgemap = dict()
    for edge_idx, edge in enumerate(edges_l):
        for edge_loc_idx, pair in enumerate(edge):
            edgemap[(pair[0],pair[1])] = (edge_idx,  edge_loc_idx)
    #====================================================================================================
    chipinfo['edges']     = (edges_l, edgemap)
    chipinfo['kmap']      = kmap_total
    #====================================================================================================

    # kplot = kmap_total[0,:50,:50]
    # plot_im(plot_data = kplot,      title="", save_name='golden power map5 kmap.png',show=False, range01=True)



    return

def analysis_chip_batch(chipinfo, numoftile_perbatch =10):
    #====================================================================================================
    totalnumnodes = chipinfo['totalnumofpixel']
    totalresx     = chipinfo['numofpixel_x']
    totalresy     = chipinfo['numofpixel_y']
    #====================================================================================================
    #====================================================================================================
    nodeidxs = np.arange(totalnumnodes).reshape((totalresx, totalresy))
    #====================================================================================================
    numoftile_x = chipinfo['numoftile_x']
    numoftile_y = chipinfo['numoftile_y']

    assert numoftile_x%numoftile_perbatch ==0
    assert numoftile_y%numoftile_perbatch ==0

    chipinfo['numofbatch_x']    = (int)(numoftile_x//numoftile_perbatch)
    chipinfo['numofbatch_y']    = (int)(numoftile_y//numoftile_perbatch)
    numofxlayer = chipinfo['numofbatch_x']
    numofylayer = chipinfo['numofbatch_y']
    chipinfo['totalnumofbatch']    = numofxlayer*numofylayer
    #====================================================================================================
    edges_l = []
    #====================================================================================================
    # create vertical line
    #====================================================================================================
    start_xidx = 0
    tile_dim  = chipinfo['tile_dim']


    for m in range(numofxlayer):
        start_yidx = tile_dim*numoftile_perbatch-1
        for n in range(numofylayer-1):
            edge_l = []
            for row_idx in range(start_xidx, start_xidx+tile_dim*numoftile_perbatch):
                edge_node_id0 = nodeidxs[row_idx,start_yidx].item()
                edge_node_id1 = nodeidxs[row_idx,start_yidx+1].item() 
                edge_l.append([edge_node_id0, edge_node_id1])
                
            start_yidx += tile_dim*numoftile_perbatch
            edges_l.append(edge_l)
        start_xidx+= tile_dim*numoftile_perbatch
    #====================================================================================================
    # create horizental line
    #====================================================================================================
    start_yidx = 0
    for m in range(numofylayer):
        start_xidx = tile_dim*numoftile_perbatch-1
        for n in range(numofxlayer-1):
            edge_l = []
            for col_idx in range(start_yidx, start_yidx+tile_dim*numoftile_perbatch):
                edge_node_id0 = nodeidxs[start_xidx,  col_idx].item()
                edge_node_id1 = nodeidxs[start_xidx+1,col_idx].item() 
                edge_l.append([edge_node_id0, edge_node_id1])
            start_xidx += tile_dim*numoftile_perbatch
            edges_l.append(edge_l)
        start_yidx+= tile_dim*numoftile_perbatch
    #====================================================================================================
    edgemap = dict()
    for edge_idx, edge in enumerate(edges_l):
        for edge_loc_idx, pair in enumerate(edge):
            edgemap[(pair[0],pair[1])] = (edge_idx,  edge_loc_idx)

    chipinfo['edges_batch'] = (edges_l, edgemap)
    #====================================================================================================
    #====================================================================================================
    tile_id = torch.arange(numoftile_x*numoftile_y).reshape(numoftile_x, numoftile_y)
    batch_id = torch.ones((numoftile_x , numoftile_y))
    id_=0
    for x in range(chipinfo['numofbatch_x']):
        for y in range(chipinfo['numofbatch_y']):
            batch_id[x*numoftile_perbatch: (x+1)*numoftile_perbatch,  y*numoftile_perbatch: (y+1)*numoftile_perbatch ] = id_
            id_+=1

    chipinfo['tile_id']= tile_id
    chipinfo['batch_id']= batch_id
    #====================================================================================================
    # generate per batch edge id
    #====================================================================================================
    
    # print(torch.arange(numoftile_perbatch*numoftile_perbatch).reshape(numoftile_perbatch,numoftile_perbatch))

    #====================================================================================================
    kmap_total = chipinfo['kmap'].squeeze()


    tile_id_pixellevel_row=[]
    batch_id_pixellevel_row=[]
    for x in range(numoftile_x):
        tile_id_pixellevel_col=[]
        batch_id_pixellevel_col=[]
        for y in range(numoftile_y):
            tile_id_tilelevel           = tile_id[x,y].item()
            batch_id_tilelevel          = batch_id[x,y].item()
            tile_id_pixellevel_pertile  = torch.ones((tile_dim,tile_dim))*tile_id_tilelevel
            batch_id_pixellevel_pertile = torch.ones((tile_dim,tile_dim))*batch_id_tilelevel
            tile_id_pixellevel_col.append(tile_id_pixellevel_pertile)
            batch_id_pixellevel_col.append(batch_id_pixellevel_pertile)

        tile_tmp= torch.hstack(tile_id_pixellevel_col)
        tile_id_pixellevel_row.append(tile_tmp)

        batch_tmp= torch.hstack(batch_id_pixellevel_col)
        batch_id_pixellevel_row.append(batch_tmp)

    tile_id_pixellevel  = torch.vstack(tile_id_pixellevel_row)
    batch_id_pixellevel = torch.vstack(batch_id_pixellevel_row)

    assert tile_id_pixellevel.shape == kmap_total.shape
    assert batch_id_pixellevel.shape == kmap_total.shape
    #====================================================================================================
    chipinfo['tile_id_pixellevel']  = tile_id_pixellevel
    chipinfo['batch_id_pixellevel'] = batch_id_pixellevel
    #====================================================================================================
    return

def generate_random_badedge(chipinfo):

    pixel_level_shape   = chipinfo['tile_id_pixellevel'].shape
    tile_id_pixellevel  = chipinfo['tile_id_pixellevel'].flatten()
    batch_id_pixellevel = chipinfo['batch_id_pixellevel'].flatten()
    numoftile_perbatch = chipinfo['numoftile_perbatch']
    tile_dim           = chipinfo['tile_dim']



    tile_id = chipinfo['tile_id'].flatten()
    batch_id = chipinfo['batch_id'].flatten()
    idx = torch.argwhere(batch_id==0).squeeze()


    tile_id_batch0_flatten = tile_id[idx]
    tile_id_batch0         = tile_id[idx].reshape(numoftile_perbatch, numoftile_perbatch)


    tile_id_foredge_singlebatch = torch.arange(len(tile_id_batch0_flatten)).reshape(tile_id_batch0.shape)
    # print(tile_id_batch0)
    print(tile_id_foredge_singlebatch)


    totalnumofpixel = chipinfo['totalnumofpixel']
    numofpixel_x    = chipinfo['numofpixel_x']
    numofpixel_y    = chipinfo['numofpixel_y']
    pixel_id_pixellevel = torch.arange(totalnumofpixel)


    pixel_idx                  = torch.argwhere(batch_id_pixellevel==0).squeeze()
    tile_id_pixellevel_batch0  = tile_id_pixellevel[pixel_idx]
    pixel_id_pixellevel_batch0 = pixel_id_pixellevel[pixel_idx]



    numofpixel_perbatch_x = tile_dim*numoftile_perbatch
    numofpixel_perbatch_y = tile_dim*numoftile_perbatch

    tile_id_pixellevel_batch0_flatten = tile_id_pixellevel_batch0.reshape(numofpixel_perbatch_x,numofpixel_perbatch_y )
    # print(tile_id_pixellevel_batch0_flatten)
    pixel_id_pixellevel_batch0_flatten = pixel_id_pixellevel_batch0.reshape(numofpixel_perbatch_x,numofpixel_perbatch_y )
    # print(pixel_id_pixellevel_batch0_flatten)
    pixel_id_pixellevel_flatten = pixel_id_pixellevel.reshape(numofpixel_x,numofpixel_y )
    # print(pixel_id_pixellevel_flatten)




    numofpixels_batch= numoftile_perbatch*numoftile_perbatch*0.2
    # numofpixels_batch= numoftile_perbatch*numoftile_perbatch
    numofpixels_batch= (int)(numofpixels_batch)

    # tileidx = torch.randperm(numoftile_perbatch*numoftile_perbatch)[:numofpixels_batch]
    # tileidx = torch.tensor([5,6,7,8,9])
    # tileidx = torch.tensor([3,4,5])
    # print(tileidx)

    badedge = set()
    badedge_l = []
    badedge_inbatch = set()

    for i, tmp_idx in enumerate(tileidx.tolist()):
        tile_id_tilelevel=tile_id_batch0_flatten[tmp_idx].item()



        tmp  = torch.argwhere(tile_id_batch0==tile_id_tilelevel).squeeze()
        x = tmp[0].item()
        y = tmp[1].item()

        
        neigh_tile = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
        newx       = None
        newy       = None
        random_dir = None
        while True:
            random_dir = random.randint(0,3)
            random_dir = tileidx_dir[i]
            newx, newy = neigh_tile[random_dir]
            if newx< 0 or newx>=numoftile_perbatch or newy<0 or newy>=numoftile_perbatch:
                continue
            break

        #====================================================================================================
        # for edge generation
        #====================================================================================================
        tile_id_foredge       = tile_id_foredge_singlebatch[x,y].item()
        tile_id_foredge_neigh = tile_id_foredge_singlebatch[newx,newy].item()
        print('tile id for edge: ', tile_id_foredge,tile_id_foredge_neigh)
        smallid = min(tile_id_foredge, tile_id_foredge_neigh)
        largeid = max(tile_id_foredge, tile_id_foredge_neigh)
        badedge_inbatch.add((smallid, largeid))
        #====================================================================================================
        # for fem edge weight 
        #====================================================================================================

        neigh_id_tilelevel = tile_id_batch0[newx,newy].item()

        idxs_center = torch.argwhere(tile_id_pixellevel_batch0==tile_id_tilelevel).flatten()
        idxs_neigh  = torch.argwhere(tile_id_pixellevel_batch0==neigh_id_tilelevel).flatten()

        pixel_id_center = pixel_id_pixellevel_batch0[idxs_center].reshape(tile_dim,tile_dim)
        pixel_id_neigh  = pixel_id_pixellevel_batch0[idxs_neigh].reshape(tile_dim,tile_dim)
        if random_dir==0:  #center shang 
            pixel_id_center_ = pixel_id_center[0]
            pixel_id_neigh_  = pixel_id_neigh[-1]

        elif random_dir==1:  #center xia 
            pixel_id_center_ = pixel_id_center[-1]
            pixel_id_neigh_  = pixel_id_neigh[0]

        elif random_dir==2:  #center zuo
            pixel_id_center = pixel_id_center.T
            pixel_id_neigh = pixel_id_neigh.T
            pixel_id_center_ = pixel_id_center[0]
            pixel_id_neigh_  = pixel_id_neigh[-1]

        elif random_dir==3:  #center you
            pixel_id_center = pixel_id_center.T
            pixel_id_neigh = pixel_id_neigh.T
            pixel_id_center_ = pixel_id_center[-1]
            pixel_id_neigh_  = pixel_id_neigh[0]


        for i in range(tile_dim):
            pixel_id_0 = pixel_id_center_[i].item()
            pixel_id_1 = pixel_id_neigh_[i].item()

            smallid = min(pixel_id_0, pixel_id_1)
            largeid = max(pixel_id_0, pixel_id_1)

            
            
            # print(chipinfo['numofbatch_y'])

            for x in range(chipinfo['numofbatch_x']):
                for y in range(chipinfo['numofbatch_y']):
                    smallid_ = smallid+y*numoftile_perbatch*tile_dim+x*numoftile_perbatch*tile_dim*chipinfo['numofbatch_y']*numoftile_perbatch*tile_dim
                    largeid_ = largeid+y*numoftile_perbatch*tile_dim+x*numoftile_perbatch*tile_dim*chipinfo['numofbatch_y']*numoftile_perbatch*tile_dim
                    badedge.add((smallid_,largeid_))
                    badedge_l.append((smallid_,largeid_))




    # print("bad node pair for fem :",badedge_l)
    # print()
    print('bad node pair for edge generation:', badedge_inbatch)
    chipinfo["badedge"]         = badedge
    chipinfo["badedge_inbatch"] = badedge_inbatch
    return

def reorgnize_graph(input_np,numoftile_x, numoftile_y, chipinfo):
    
    if torch.is_tensor(input_np):
        input_np = input_np.clone().detach().cpu().numpy()
    #====================================================================================================
    assert len(input_np)==numoftile_x*numoftile_y
    #====================================================================================================
    tile_dim    = chipinfo['tile_dim']
    #====================================================================================================
    output_np = np.zeros((numoftile_x*tile_dim, numoftile_y*tile_dim))

    currx=curry=0
    node_id = 0
    for x in range(numoftile_x):
        for y in range(numoftile_y):
            if len(input_np.shape)==2:
                node_info = input_np[node_id,:].reshape(tile_dim,tile_dim )
            else :
                node_info = input_np[node_id,:,:].reshape(tile_dim,tile_dim )
            output_np[currx:currx+tile_dim, curry:curry+tile_dim] = node_info
            curry+=tile_dim
            node_id+=1

        currx+=tile_dim
        curry=0
    return output_np

def duplicate_edge( edge, edge_feat, num, numoftiles_perbatch, numofdummynode=0 ):

    max_node_id     = edge.max().item()
    max_possible_id = (max_node_id+1)*num


    dummy_node_id=dict()
    #====================================================================================================
    if add_bc_dummynode:
        numofdummynode+=1
        id_=-1
    else:
        id_=-2
    #====================================================================================================
         
    for i in range(numofdummynode):
        dummy_node_id[id_] = max_possible_id
        id_-=1
        max_possible_id+=1

    # print(edge.shape)

    idxs        = []
    replacepair = []
    for i in range(len(edge[0])):
        pair = edge[:,i]
        if pair[0]<0 and pair[1]<0:
            print('error')
            exit()

        if pair[0]<0 :
            newid=dummy_node_id[pair[0].item()]
            idxs.append(i)
            replacepair.append((newid,   pair[1]))

        elif pair[1]<0 :
            newid=dummy_node_id[pair[1].item()]
            idxs.append(i)
            replacepair.append((pair[0],   newid))

            


    edge_batch_l =[]
    edge_batch_id_l =[]
    for sample_idx in range(num):

        edge_currbatch = edge+sample_idx*numoftiles_perbatch

        for i,idx in enumerate(idxs):
            if edge[0,idx]<0:
                edge_currbatch[0,idx]= replacepair[i][0]
            if edge[1,idx]<0:
                edge_currbatch[1,idx]= replacepair[i][1]





        edge_batch_l.append(edge_currbatch)
        edge_batch_id_l.append(edge_feat)

    edge_tensor    = torch.hstack(edge_batch_l).to(device)
    edge_tensor_id = torch.vstack(edge_batch_id_l).to(device)

    assert edge_tensor.min().item()>=0

    # print(edge_tensor.shape)
    # print(edge_tensor)
    # print(edge_tensor_id.shape)
    # print(edge_tensor.max())




    return edge_tensor, edge_tensor_id

def generate_batch_edge(chipinfo):
    numoftile_perbatch = chipinfo['numoftile_perbatch']

    if 'badedge_inbatch' in chipinfo.keys():
        badedge_inbatch    = chipinfo['badedge_inbatch']
    else:
        badedge_inbatch=set()
         
    dummy_node_id      = numoftile_perbatch*numoftile_perbatch
    # dummy_node_id      = 1000


    bc_id = -1
    dummy_node_id   = -2
    dummy_node_dict = dict()
    numofdummynode  = 0




    edge_l    = []
    edge_id_l = []
    node_id   = 0

    for x in range(numoftile_perbatch):
        for y in range(numoftile_perbatch):

            
            #====================================================================================================
            #bc remove
            if add_bc_dummynode:
                if x==0: # add top
                    edge_l.append([node_id, bc_id])
                    # edge_l.append([bc_id, node_id])
                    edge_id_l.append(edgeiddict[0])
                    # edge_id_l.append(edgeiddict[1])
                if y==0: # add left
                    edge_l.append([node_id, bc_id])
                    # edge_l.append([bc_id, node_id])
                    edge_id_l.append(edgeiddict[2])
                    # edge_id_l.append(edgeiddict[3])
                if x==numoftile_perbatch-1: # add bot
                    edge_l.append([node_id, bc_id])
                    # edge_l.append([bc_id, node_id])
                    edge_id_l.append(edgeiddict[1])
                    # edge_id_l.append(edgeiddict[0])
                if y==numoftile_perbatch-1: # add right
                    edge_l.append([node_id, bc_id])
                    # edge_l.append([bc_id, node_id])
                    edge_id_l.append(edgeiddict[3])
                    # edge_id_l.append(edgeiddict[2])

            #====================================================================================================
            for edgeidx, (newx, newy) in enumerate([(x-1,y), (x+1,y), (x,y-1), (x,y+1)]):
                if newx< 0 or newx>= numoftile_perbatch or newy<0 or newy>=numoftile_perbatch:
                    continue
                neigh_id = newx* numoftile_perbatch+ newy
                #====================================================================================================
                smallid = min(node_id, neigh_id)
                largeid = max(node_id, neigh_id)
                if (smallid, largeid) in badedge_inbatch:
                    if (smallid, largeid) in dummy_node_dict:
                        dummy_node_id_ = dummy_node_dict[(smallid,largeid)]
                    else:
                        dummy_node_id_                      = dummy_node_id
                        dummy_node_dict[(smallid,largeid)]  = dummy_node_id_
                        dummy_node_id                      -= 1
                        numofdummynode                     += 1

                    edge_l.append([node_id, dummy_node_id_])
                    edge_l.append([dummy_node_id_, node_id])

                    edge_id_l.append(edgeiddict[edgeidx])
                    if   edgeidx==0: new_edgeidx=1
                    elif edgeidx==1: new_edgeidx=0
                    elif edgeidx==2: new_edgeidx=3
                    elif edgeidx==3: new_edgeidx=2
                    edge_id_l.append(edgeiddict[new_edgeidx])

                else:
                    #====================================================================================================
                    edge_l.append([node_id, neigh_id])
                    edge_id_l.append(edgeiddict[edgeidx])
            #====================================================================================================
            node_id+=1
        curry=0

    edge_tensor    = torch.tensor(edge_l).to(device).T
    edge_id_tensor = torch.tensor(edge_id_l).to(device)
    # print(edge_tensor)
    # print(numofdummynode)
    # print(edge_id_tensor)
    # exit()
    chipinfo['numofdummynode']=numofdummynode
    return edge_tensor, edge_id_tensor







def seperate_wholemap(input_np_flatten, golden_np_flatten, powermap_np_flatten,kmap_np_flatten,chipinfo):
    #====================================================================================================
    chipres                      = chipinfo['chipres']
    chipshape                    = chipinfo['chipshape']
    edges_l, edgemap             = chipinfo['edges']
    edges_batch_l, edgemap_batch = chipinfo['edges_batch']
    kmap_total                   = chipinfo['kmap']
    #====================================================================================================
    numofpixel_x       = chipinfo['numofpixel_x']
    numofpixel_y       = chipinfo['numofpixel_y']
    tile_dim       = chipinfo['tile_dim']
    #====================================================================================================
    input_np  = input_np_flatten.reshape((numofpixel_x, numofpixel_y))
    golden_np = golden_np_flatten.reshape((numofpixel_x, numofpixel_y))
    power_np  = powermap_np_flatten.reshape((numofpixel_x, numofpixel_y))
    k_np  = kmap_np_flatten.reshape((numofpixel_x, numofpixel_y))
    #====================================================================================================
    input_l   = []
    inputp_l  = []
    inputk_l  = []
    output_l  = []
    edge_l    = []
    edge_id_l = []
    corenode_idx_set = set()

    #====================================================================================================
    currx=curry=0
    node_id = 0
    for x in range(len(chipshape)):
        for y in range(len(chipshape[0])):
            #====================================================================================================
            input_single_node_np = input_np[currx:currx+tile_dim, curry:curry+tile_dim]
            power_single_node_np = power_np[currx:currx+tile_dim, curry:curry+tile_dim]
            golden_single_node_np = golden_np[currx:currx+tile_dim, curry:curry+tile_dim]

            k_single_node_np = k_np[currx:currx+tile_dim, curry:curry+tile_dim]
            curr_k = k_single_node_np[0,0].item()
            if curr_k == k1:
                curr_k = 1
            else:
                curr_k = 0.1
            #====================================================================================================
            numoffeature = tile_dim*tile_dim
            node_feature = np.hstack([input_single_node_np.reshape((1,numoffeature))])
            node_power   = np.hstack([power_single_node_np.reshape((1,numoffeature))])
            node_k       = np.hstack([np.array(curr_k).reshape((1,1))])
            golden_feature = golden_single_node_np.reshape((1, numoffeature))
            #====================================================================================================
            node_feature_tensor = torch.tensor(node_feature).double().to(device)
            node_power_tensor   = torch.tensor(node_power).double().to(device)
            node_k_tensor       = torch.tensor(node_k).double().to(device)
            golden_feature_tensor = torch.tensor(golden_feature).double().to(device)
            #====================================================================================================
            input_l.append(node_feature_tensor)
            inputp_l.append(node_power_tensor)
            inputk_l.append(node_k_tensor)
            output_l.append(golden_feature_tensor)
            #====================================================================================================

            if x==0 or x==len(chipshape)-1 or y==0 or y==len(chipshape[0])-1:
                pass
            else:
                corenode_idx_set.add(node_id)

            #====================================================================================================
            for edgeidx, (newx, newy) in enumerate([(x-1,y), (x+1,y), (x,y-1), (x,y+1)]):
                if newx< 0 or newx>= len(chipshape) or newy<0 or newy>=len(chipshape[0]):
                    continue
                neigh_id = newx* len(chipshape[0])+ newy
                edge_l.append([node_id, neigh_id])
                edge_id_l.append(edgeiddict[edgeidx])
            #====================================================================================================
            curry+=tile_dim
            node_id+=1
        currx+=tile_dim
        curry=0
    #====================================================================================================
    edge_tensor   = torch.tensor(edge_l).to(device).T
    input_tensor  = torch.vstack(input_l)
    inputp_tensor  = torch.vstack(inputp_l)
    inputk_tensor  = torch.vstack(inputk_l)
    output_tensor = torch.vstack(output_l)
    corenode_idx_tensor = torch.tensor(list(corenode_idx_set))
    edge_id_tensor = torch.tensor(edge_id_l)

    return input_tensor, inputp_tensor, inputk_tensor, output_tensor, edge_tensor, corenode_idx_tensor,edge_id_tensor






def generate_edge_training_samples( chipinfo, load_old=False,skipededgepair=None,adddummynode=False,powermap=None,kmap=None ):

    #====================================================================================================
    edges_l, edgemap             = chipinfo['edges']
    edges_batch_l, edgemap_batch = chipinfo['edges_batch']
    kmap_total                   = chipinfo['kmap']
    numoftile_perbatch           = chipinfo['numoftile_perbatch']
    #====================================================================================================
    numoftile_x = chipinfo['numoftile_x']
    numoftile_y = chipinfo['numoftile_y']
    if 'badedge' in chipinfo:
        badedge_set = chipinfo['badedge']
    else:
        badedge_set = set() 
    #====================================================================================================
    tile_dim        = chipinfo['tile_dim']
    totalnumofpixel = chipinfo['totalnumofpixel']
    numofpixel_x       = chipinfo['numofpixel_x']
    numofpixel_y       = chipinfo['numofpixel_y']
    #====================================================================================================
    node_dim = tile_dim*tile_dim*2+1  # feat + power + k
    dummy_node_feat = torch.rand((1, node_dim)).double().to(device)
    #====================================================================================================
    #====================================================================================================
    input_l  = []
    inputp_l = []
    inputk_l = []
    output_l = []
    edge_l   = []
    #====================================================================================================
    goldengen_merge = golden(numofpixel_x,numofpixel_y)
    goldengen_merge.k = kmap_total
    #====================================================================================================
    if powermap is None:
        
        pmap_total  = []
        kmap_total  = []
        for x in range(numoftile_x):
            pmap_l  = []
            kmap_l  = []
            for y in range(numoftile_y):
                ##====================================================================================================
                power_map_tensor             = generate_power_map_fortrain(numoftrain=1,resx=tile_dim,resy=tile_dim)[0]
                ##====================================================================================================
                power_map_np = power_map_tensor.cpu().numpy()
                k_map_np     = np.ones(power_map_np.shape)*k1
                if random.uniform(0, 10)>6:
                    k_map_np=np.ones(power_map_np.shape)*k2

                # if random.uniform(0, 10)>-1:
                if random.uniform(0, 10)>4:
                    pmap_l.append(power_map_np)
                else:
                    pmap_l.append(np.ones(power_map_np.shape)*0.0000001)

                kmap_l.append(k_map_np)
                     
            pmap_total.append(np.hstack(pmap_l))
            kmap_total.append(np.hstack(kmap_l))

        pmap_total        = np.vstack(pmap_total).reshape((1, numofpixel_x,numofpixel_y))
        kmap_total        = np.vstack(kmap_total).reshape((1, numofpixel_x,numofpixel_y))

        goldengen_merge.p = pmap_total
        goldengen_merge.k = kmap_total
    else:
        goldengen_merge.p = powermap
        goldengen_merge.k = kmap
        pmap_total        = powermap
        kmap_total        = kmap


    #====================================================================================================
    # pplot = pmap_total[0,:50,:50]
    # kplot = kmap_total[0,:50,:50]
    # plot_im(plot_data = pplot,      title="", save_name='golden power map6 pmap.png',show=False, range01=True)
    # plot_im(plot_data = kplot,      title="", save_name='golden power map5 kmap.png',show=False, range01=True)
    # goldengen_merge.p = pmap_total
    #====================================================================================================
    edgeinfo_batch   = [np.zeros((tile_dim*numoftile_perbatch))]*len(edges_batch_l)
    edgeinfo_single  = [np.zeros((tile_dim))]*len(edges_l)
    #====================================================================================================
    goldengen_merge.edge_weight = edgeinfo_batch
    goldengen_merge.badedge     = badedge_set
    goldengen_merge.bc_pair     = edgemap_batch
    # goldengen_merge.bc_pair     = None 
    golden_out_merge_np         = goldengen_merge.gen(edges_l=edges_batch_l, edgemap=edgemap_batch, skipededgepair=skipededgepair).copy().reshape(kmap_total.shape).squeeze()
    print('Done generating the golden') 

    goldengen_merge.edge_weight = edgeinfo_single
    goldengen_merge.badedge     = None
    goldengen_merge.bc_pair     = edgemap_batch
    golden_out_stich_np         = goldengen_merge.gen(edges_l=edges_l, edgemap=edgemap).reshape(kmap_total.shape).squeeze()

    out_stich_total_np  = golden_out_stich_np.reshape((1, totalnumofpixel))
    golden_out_merge_np = golden_out_merge_np.reshape((1, totalnumofpixel))
    # exit()

    #====================================================================================================
    # plot input, output result
    #====================================================================================================
    # out_stich_total_np  = normalize(out_stich_total_np.reshape((numoftile_x*tile_dim,numoftile_y*tile_dim)))
    # golden_out_merge_np = normalize(golden_out_merge_np.reshape((numoftile_x*tile_dim,numoftile_y*tile_dim)))

    # plot_im(plot_data=pmap_total[0,:5,:5],        title="", save_name='golden power map0 pinput.png',show=False, vmin=vmin,vmax=vmax,range01=False)
    # plot_im(plot_data=normalize(out_stich_total_np.reshape((500,50))),      title="", save_name='golden power map1 stich.png',show=False, vmin=vmin,vmax=vmax,range01=False)
    # plot_im(plot_data=normalize(golden_out_merge_np.reshape((500,50))),      title="", save_name='golden power map2 merge.png',show=False, vmin=vmin,vmax=vmax,range01=False)

    # plot_im(plot_data=normalize(golden_out_stich_np)[:numoftile_perbatch*cell_dim*2,:numoftile_perbatch*cell_dim*2],      title="", save_name='golden power map1 merge.png',show=False, range01=False)
    # plot_im(plot_data=normalize(golden_out_merge_np)[:numoftile_perbatch*cell_dim*2,:numoftile_perbatch*cell_dim*2],      title="", save_name='golden power map2 merge.png',show=False, range01=False)
    # exit()


    # idx=2
    # plot_im(plot_data=out_stich_total_np[numoftile_perbatch*tile_dim*idx:numoftile_perbatch*tile_dim*(idx+1),:numoftile_perbatch*tile_dim],title="", save_name='golden power map1 stich.png',show=False, vmin=0,vmax=1,range01=False)
    # plot_im(plot_data=golden_out_merge_np[numoftile_perbatch*tile_dim*idx:numoftile_perbatch*tile_dim*(idx+1),:numoftile_perbatch*tile_dim],      title="", save_name='golden power map2 merge.png',show=False, vmin=0,vmax=1,range01=False)
    # exit()
    #====================================================================================================
    # seperate the stich map and golden out to single tile
    #====================================================================================================
    input_feat_tensor, inputp_tensor, inputk_tensor, golden_output_tensor,  edge_t, corenode_idx_tensor, edge_id_tensor = seperate_wholemap(out_stich_total_np, golden_out_merge_np, pmap_total, kmap_total,chipinfo)
    #====================================================================================================
    # print(chipinfo['tile_id'])
    # print(chipinfo['batch_id'])
    # exit()
    #====================================================================================================
    tile_id         = chipinfo['tile_id'].flatten()
    batch_id        = chipinfo['batch_id'].flatten()
    totalnumofbatch = chipinfo['totalnumofbatch']
    #====================================================================================================
    input_feat_tensor    = normalize(input_feat_tensor)
    golden_output_tensor = normalize(golden_output_tensor)
    # plot_im(plot_data=reorgnize_graph(input_feat_tensor,numoftile_x, numoftile_y,chipinfo),title="", save_name='./tmp/0stich.png',show=False, vmin=0,vmax=1,range01=False)
    # plot_im(plot_data=reorgnize_graph(golden_output_tensor,numoftile_x, numoftile_y,chipinfo),title="", save_name='./tmp/0merge.png',show=False, vmin=0,vmax=1,range01=False)
    # exit()
    #====================================================================================================
    input_plot_batch_l  = []
    input_tensor_batch_l  = []
    output_tensor_batch_l = []
    #====================================================================================================
    
    for batchid in range(totalnumofbatch):
        batch_idx_tilelevel = torch.argwhere(batch_id==batchid).squeeze()
        tile_idx_tilelevel  = tile_id[batch_idx_tilelevel].to(device)

        input_feat_ = torch.index_select(input=input_feat_tensor,    index=tile_idx_tilelevel, dim=0)
        input_p_    = torch.index_select(input=inputp_tensor,        index=tile_idx_tilelevel, dim=0)
        input_k_    = torch.index_select(input=inputk_tensor,        index=tile_idx_tilelevel, dim=0)
        output_t    = torch.index_select(input=golden_output_tensor, index=tile_idx_tilelevel, dim=0)

        input_plot_batch_l.append([input_feat_, input_p_, input_k_])
        input_feat_ = torch.cat([input_feat_, input_p_, input_k_],dim=1)
        # if adddummynode:
        #     input_feat_ = torch.vstack([input_feat_, dummy_node_feat])

        input_tensor_batch_l.append(input_feat_)
        output_tensor_batch_l.append(output_t)
    #====================================================================================================
    # plot_im(plot_data=reorgnize_graph(input_plot_batch_l[0][0],numoftile_perbatch, numoftile_perbatch,chipinfo),title="", save_name='./tmp/{}stich.png'.format(0),show=False, vmin=0,vmax=1,range01=False)
    # plot_im(plot_data=reorgnize_graph(output_tensor_batch_l[0],numoftile_perbatch, numoftile_perbatch,chipinfo), title="", save_name='./tmp/{}merge.png'.format(0),show=False, vmin=0,vmax=1,range01=False)
    # exit()
    #====================================================================================================
    #====================================================================================================
    
    input_tensor_batch = torch.stack(input_tensor_batch_l)
    output_tensor_batch = torch.stack(output_tensor_batch_l)


    return input_tensor_batch,  output_tensor_batch, pmap_total,kmap_total




















def train_gnn(model, data, numofepoch, chipinfo, name ='' ):

    input_batch_tensor, output_batch_tensor,edge_idx_perbatch, edge_feat_tensor_perbatch, _,_ = data 
    totalnumofbatch = chipinfo['totalnumofbatch']
    numoftile_perbatch = chipinfo['numoftile_perbatch']
    numofnodepergraph = numoftile_perbatch*numoftile_perbatch


    numofbatch_train = (int)(totalnumofbatch*0.8)
    numofbatch_val   = (int)(totalnumofbatch*0.15)
    numofbatch_test  = totalnumofbatch-numofbatch_train-numofbatch_val
    # numofbatch_test  = min(numofbatch_test,(int)(numofbatch_train/0.8*0.05) )



    input_train_tensor  = input_batch_tensor[:numofbatch_train,:,:].double().to(device)
    output_train_tensor = output_batch_tensor[:numofbatch_train,:,:].double().to(device)
    input_val_tensor    = input_batch_tensor[numofbatch_train:numofbatch_train+numofbatch_val,:,:].double().to(device)
    output_val_tensor   = output_batch_tensor[numofbatch_train:numofbatch_train+numofbatch_val,:,:].double().to(device)
    input_test_tensor   = input_batch_tensor[numofbatch_train+numofbatch_val:numofbatch_train+numofbatch_val+numofbatch_test,:,:].double().to(device)
    output_test_tensor  = output_batch_tensor[numofbatch_train+numofbatch_val:numofbatch_train+numofbatch_val+numofbatch_test,:,:].double().to(device)

    #====================================================================================================


    #====================================================================================================
    edge_idx_val,   edge_feat_val   = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, numofbatch_val, numoftile_perbatch*numoftile_perbatch )
    edge_idx_test,  edge_feat_test  = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, numofbatch_test, numoftile_perbatch*numoftile_perbatch)
    edge_idx_train, edge_feat_train = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, batchtrainsize, numoftile_perbatch*numoftile_perbatch )
    #====================================================================================================



    tile_dim = chipinfo['tile_dim']
    input_dim = tile_dim*tile_dim*2+1
    if add_bc_dummynode:
        bc_dummynode = torch.zeros((1, input_dim)).double().to(device)
    else:
        bc_dummynode = None
    #====================================================================================================



    #====================================================================================================
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=inilr, weight_decay=5e-8)
    # optimizer = torch.optim.SGD(model.parameters(), lr=inilr, weight_decay=5e-10)
    mse = nn.MSELoss()
    bestloss=2000000
    # numofnodepergraph = len(corenode_idx_tensor)
    #====================================================================================================

    loss_log = []
    try:
        for epoch in range(numofepoch):  

            trainsamples =numofbatch_train
            train_idx_shuffle=np.random.permutation(np.arange(trainsamples))

            loss_l =[]
            for train_idx in range(trainsamples//batchtrainsize) :

                #====================================================================================================
                optimizer.zero_grad()
                #====================================================================================================
                startidx       = train_idx*batchtrainsize
                endidx         = (train_idx+1)*batchtrainsize
                curr_train_idx = train_idx_shuffle[startidx:endidx].tolist()

                #====================================================================================================
                curr_train_idx_t = torch.tensor(curr_train_idx).to(device)
                input_tensor     = torch.index_select(input=input_train_tensor, index=curr_train_idx_t, dim=0)
                output_tensor    = torch.index_select(input=output_train_tensor, index=curr_train_idx_t, dim=0)
                
                #====================================================================================================
                pred_tensor_core = model(input_tensor,edge_idx_train, edge_feat_train,bc_dummynode).reshape(batchtrainsize, numofnodepergraph, -1)
                #====================================================================================================
                loss = mse(pred_tensor_core, output_tensor)
                loss.backward()
                optimizer.step()
                loss_l.append(loss.item())




            model.eval()
            pred_tensor_val_core = model(input_val_tensor,edge_idx_val, edge_feat_val,bc_dummynode).reshape(numofbatch_val, numofnodepergraph, -1)
            loss_val = mse(pred_tensor_val_core, output_val_tensor).item()


            #====================================================================================================
            idx = random.randint(0, len(input_test_tensor)-1)
            input_test_tensor_tmp = input_test_tensor[idx:idx+1,:,:]
            output_test_tensor_tmp = output_test_tensor[idx:idx+1,:,:]
            edge_idx_test_tmp,  edge_feat_test_tmp  = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, 1, numoftile_perbatch*numoftile_perbatch)
            pred_tensor_test_core = model(input_test_tensor_tmp,edge_idx_test_tmp, edge_feat_test_tmp,bc_dummynode).reshape(1, numofnodepergraph, -1)
            # loss_test = mse(pred_tensor_test_core, output_test_tensor_tmp).item()
            #====================================================================================================



            input_                  = reorgnize_graph(input_test_tensor_tmp[:,:,:cell_dim*cell_dim].squeeze(),numoftile_perbatch,numoftile_perbatch,chipinfo)
            pred_tensor_test_core   = reorgnize_graph(pred_tensor_test_core.squeeze(),numoftile_perbatch,numoftile_perbatch,chipinfo)
            golden_tensor_test_core = reorgnize_graph(output_test_tensor_tmp.squeeze(),numoftile_perbatch,numoftile_perbatch,chipinfo)

            golden_tensor_test_core= normalize(golden_tensor_test_core)
            pred_tensor_test_core= normalize(pred_tensor_test_core)
            abs_err , _             = comparepredict_golden(pred_tensor_test_core, golden_tensor_test_core)
            # print('Abs err  :', np.mean(abs_err).item())

            # plot_im(plot_data = input_,    title="",  save_name='./epoch/{}test map0 input.png'.format(epoch),show=False, range01=False)
            # plot_im(plot_data = golden_tensor_test_core,    title="",  save_name='./epoch/{}test map1 golden.png'.format(epoch),show=False, range01=False)
            # plot_im(plot_data = pred_tensor_test_core,      title="",  save_name='./epoch/{}test map2 pred.png'.format(epoch)  ,show=False, range01=False)
            # plot_im(plot_data = abs_err,      title="",  save_name='./epoch/{}test map3 err.png'.format(epoch)  ,show=False, range01=False)







            pred_tensor_test_core = model(input_test_tensor,edge_idx_test, edge_feat_test,bc_dummynode).reshape(numofbatch_test, numofnodepergraph, -1)
            loss_test = mse(pred_tensor_test_core, output_test_tensor).item()

            model.train()









            if epoch>=4 and loss_val<bestloss:
                bestloss = loss_val
                if name !='':
                    torch.save(model.state_dict(), './pickles/{}.pt'.format("best_trained_model_"+name))
                else :
                    torch.save(model.state_dict(), './pickles/{}.pt'.format("best_trained_model"))

            print(name, '   Epoch={}, train_loss={},\t val_loss={},\t  test_loss={},'.format(epoch, round(np.mean(loss_l).item(),5) , round(loss_val,5), round(loss_test,5)))
            loss_log.append((epoch, '  ',   round(np.mean(loss_l).item(),5),'  ',    round(loss_val,5),'  ',    round(loss_test,5)  ))




    except Exception:

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        with open('log.txt', 'a') as f:
            f.write('\n')
            f.write('\n')
            f.write('\n')
            f.write('='*100)
            f.write('\n')
            f.write(dt_string)
            f.write('\n')
            f.write('='*100)
            f.write('\n')

            for l in loss_log:
                str_ = 'epoch '+str(l[0])+l[1]+str(l[2])
                print(str_)
                f.write(str_)
                f.write('\n')

        traceback.print_exc()
        exit()















def train_defect_node(model, data, numofepoch, chipinfo, name ='' ):


    #====================================================================================================
    model.load_state_dict(torch.load('./pickles/{}.pt'.format(name)))
    model.eval()
    #====================================================================================================
    numofdummynode= chipinfo['numofdummynode']
    #====================================================================================================
    tile_dim = chipinfo['tile_dim']
    input_dim = tile_dim*tile_dim*2+1
    dummy_node_feat = torch.rand((numofdummynode, input_dim)).double().to(device)
    dummy_node_feat.requires_grad = True


    #====================================================================================================
    input_batch_tensor, output_batch_tensor,edge_idx_perbatch, edge_feat_tensor_perbatch,_,_ = data 
    totalnumofbatch    = chipinfo['totalnumofbatch']
    numoftile_perbatch = chipinfo['numoftile_perbatch']
    numofnodepergraph  = numoftile_perbatch*numoftile_perbatch


    # numofbatch_train = (int)(totalnumofbatch*0.3)
    # numofbatch_val   = (int)(totalnumofbatch*0.3)
    # numofbatch_test  = totalnumofbatch-numofbatch_train-numofbatch_val
    # numofbatch_test  = min(numofbatch_test,(int)(numofbatch_train/0.3*0.4) )

    numofbatch_train = (int)(totalnumofbatch*0.8)
    numofbatch_val   = (int)(totalnumofbatch*0.15)
    numofbatch_test  = totalnumofbatch-numofbatch_train-numofbatch_val



    input_train_tensor  = input_batch_tensor[:numofbatch_train,:,:].double().to(device)
    output_train_tensor = output_batch_tensor[:numofbatch_train,:,:].double().to(device)
    input_val_tensor    = input_batch_tensor[numofbatch_train:numofbatch_train+numofbatch_val,:,:].double().to(device)
    output_val_tensor   = output_batch_tensor[numofbatch_train:numofbatch_train+numofbatch_val,:,:].double().to(device)
    input_test_tensor   = input_batch_tensor[numofbatch_train+numofbatch_val:numofbatch_train+numofbatch_val+numofbatch_test,:,:].double().to(device)
    output_test_tensor  = output_batch_tensor[numofbatch_train+numofbatch_val:numofbatch_train+numofbatch_val+numofbatch_test,:,:].double().to(device)



    traincount=50
    input_train_tensor  = input_train_tensor[:traincount,:,:]
    output_train_tensor = output_train_tensor[:traincount,:,:]
    numofbatch_train =traincount 
    #====================================================================================================
    # print(numofbatch_train)
    # print(numofbatch_test)
    # print(numofbatch_val)
    #====================================================================================================


    batchtrainsize=1
    edge_idx_val,   edge_feat_val   = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, numofbatch_val, numoftile_perbatch*numoftile_perbatch, numofdummynode)
    # edge_idx_val,   edge_feat_val   = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, 1, numoftile_perbatch*numoftile_perbatch, numofdummynode)
    edge_idx_test,  edge_feat_test  = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, numofbatch_test, numoftile_perbatch*numoftile_perbatch,numofdummynode)
    edge_idx_train, edge_feat_train = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, batchtrainsize, numoftile_perbatch*numoftile_perbatch,numofdummynode)
    #====================================================================================================
    # print(edge_idx_val)
    # exit()






    #====================================================================================================
    if add_bc_dummynode:
        bc_dummynode = torch.zeros((1, input_dim)).double().to(device)
    else:
        bc_dummynode = None
    #====================================================================================================
    # optimizer = torch.optim.Adam(model.parameters(), lr=inilr, weight_decay=5e-8)
    optimizer = torch.optim.Adam([dummy_node_feat], lr=inilr*100, weight_decay=5e-8)
    mse = nn.MSELoss()
    bestloss=2000000
    # numofnodepergraph = len(corenode_idx_tensor)
    #====================================================================================================

    try:
        for epoch in range(numofepoch):  

            trainsamples =numofbatch_train
            train_idx_shuffle=np.random.permutation(np.arange(trainsamples))

            loss_l =[]
            # batchtrainsize = min(batchtrainsize , trainsample)
            for train_idx in range(trainsamples//batchtrainsize) :

                #====================================================================================================
                optimizer.zero_grad()
                #====================================================================================================
                startidx       = train_idx*batchtrainsize
                endidx         = (train_idx+1)*batchtrainsize
                curr_train_idx = train_idx_shuffle[startidx:endidx].tolist()

                #====================================================================================================
                curr_train_idx_t = torch.tensor(curr_train_idx).to(device)
                input_tensor     = torch.index_select(input=input_train_tensor, index=curr_train_idx_t, dim=0)
                output_tensor    = torch.index_select(input=output_train_tensor, index=curr_train_idx_t, dim=0)


                #====================================================================================================
                pred_tensor_core = model(input_tensor,edge_idx_train, edge_feat_train, bc_dummynode, dummy_node_feat ).reshape(batchtrainsize, numofnodepergraph, -1)
                #====================================================================================================
                loss = mse(pred_tensor_core, output_tensor)
                loss.backward()
                optimizer.step()
                loss_l.append(loss.item())




            pred_tensor_val_core = model(input_val_tensor,edge_idx_val, edge_feat_val, bc_dummynode, dummy_node_feat).reshape(numofbatch_val, numofnodepergraph, -1)
            loss_val = mse(pred_tensor_val_core, output_val_tensor).item()


            #====================================================================================================
            # idx = random.randint(0, len(input_test_tensor)-1)
            # input_test_tensor_tmp = input_test_tensor[idx:idx+1,:,:]
            # output_test_tensor_tmp = output_test_tensor[idx:idx+1,:,:]
            # edge_idx_test_tmp,  edge_feat_test_tmp  = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, 1, numoftile_perbatch*numoftile_perbatch,numofdummynode)
            # exit()
            # pred_tensor_test_core = model(input_test_tensor_tmp,edge_idx_test_tmp, edge_feat_test_tmp).reshape(1, numofnodepergraph, -1)
            # loss_test = mse(pred_tensor_test_core, output_test_tensor_tmp).item()
            #====================================================================================================



            # input_                  = reorgnize_graph(input_test_tensor_tmp[:,:,:cell_dim*cell_dim].squeeze(),numoftile_perbatch,numoftile_perbatch,chipinfo)
            # pred_tensor_test_core   = reorgnize_graph(pred_tensor_test_core.squeeze(),numoftile_perbatch,numoftile_perbatch,chipinfo)
            # golden_tensor_test_core = reorgnize_graph(output_test_tensor_tmp.squeeze(),numoftile_perbatch,numoftile_perbatch,chipinfo)

            # golden_tensor_test_core= normalize(golden_tensor_test_core)
            # pred_tensor_test_core= normalize(pred_tensor_test_core)
            # abs_err , _             = comparepredict_golden(pred_tensor_test_core, golden_tensor_test_core)
            # print('Abs err  :', np.mean(abs_err).item())

            # plot_im(plot_data = input_,    title="",  save_name='./epoch/{}test map0 input.png'.format(epoch),show=False, range01=False)
            # plot_im(plot_data = golden_tensor_test_core,    title="",  save_name='./epoch/{}test map1 golden.png'.format(epoch),show=False, range01=False)
            # plot_im(plot_data = pred_tensor_test_core,      title="",  save_name='./epoch/{}test map2 pred.png'.format(epoch)  ,show=False, range01=False)
            # plot_im(plot_data = abs_err,      title="",  save_name='./epoch/{}test map3 err.png'.format(epoch)  ,show=False, range01=False)







            pred_tensor_test_core = model(input_test_tensor,edge_idx_test, edge_feat_test, bc_dummynode,dummy_node_feat).reshape(numofbatch_test, numofnodepergraph, -1)
            loss_test = mse(pred_tensor_test_core, output_test_tensor).item()

            # model.train()


            if epoch>=4 and loss_val<bestloss:
                bestloss = loss_val
                with open('./pickles/dummynode_best.pickle', "wb") as f: pickle.dump((dummy_node_feat), f,protocol=pickle.HIGHEST_PROTOCOL)

            print(name, '   Epoch={}, train_loss={},\t val_loss={},\t  test_loss={},'.format(epoch, round(np.mean(loss_l).item(),5) , round(loss_val,5), round(loss_test,5)))




    except Exception:

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        with open('log.txt', 'a') as f:
            f.write('\n')
            f.write('\n')
            f.write('\n')
            f.write('='*100)
            f.write('\n')
            f.write(dt_string)
            f.write('\n')
            f.write('='*100)
            f.write('\n')

            for l in loss_log:
                str_ = 'epoch '+str(l[0])+l[1]+str(l[2])
                print(str_)
                f.write(str_)
                f.write('\n')

        traceback.print_exc()
        exit()










def test_gnn(model, data, chipinfo, show=False, idx=None, name='',dummy_node=None ):


    model.load_state_dict(torch.load('./pickles/{}.pt'.format(name)))

    input_batch_tensor, output_batch_tensor,edge_idx_perbatch, edge_feat_tensor_perbatch,_,kmap = data
    totalnumofbatch    = chipinfo['totalnumofbatch']
    numoftile_perbatch = chipinfo['numoftile_perbatch']
    numofnodepergraph  = numoftile_perbatch*numoftile_perbatch


    numofbatch_train = (int)(totalnumofbatch*0.8)
    numofbatch_val   = (int)(totalnumofbatch*0.15)
    numofbatch_test  = totalnumofbatch-numofbatch_train-numofbatch_val


    #====================================================================================================
    tile_dim      = chipinfo['tile_dim']
    singlerow     = (numofbatch_train+numofbatch_val+idx)//(numoftile_x//numoftile_perbatch)
    singlerow_mod = (numofbatch_train+numofbatch_val+idx)%(numoftile_x//numoftile_perbatch)
    kmap=kmap.squeeze()
    kmap_curr = kmap[singlerow*tile_dim*numoftile_perbatch:(singlerow+1)*tile_dim*numoftile_perbatch, singlerow_mod*tile_dim*numoftile_perbatch:(singlerow_mod+1)*tile_dim*numoftile_perbatch, ]
    #====================================================================================================



    input_test_tensor   = input_batch_tensor[numofbatch_train+numofbatch_val:numofbatch_train+numofbatch_val+numofbatch_test,:,:].double().to(device)
    output_test_tensor  = output_batch_tensor[numofbatch_train+numofbatch_val:numofbatch_train+numofbatch_val+numofbatch_test,:,:].double().to(device)




    if 'numofdummynode' in chipinfo.keys():
        numofdummynode= chipinfo['numofdummynode']
    else:
        numofdummynode=0


    #====================================================================================================
    tile_dim = chipinfo['tile_dim']
    input_dim = tile_dim*tile_dim*2+1
    if add_bc_dummynode:
        bc_dummynode = torch.zeros((1, input_dim)).double().to(device)
    else:
        bc_dummynode = None
    #====================================================================================================
    edge_idx_test,  edge_feat_test  = duplicate_edge( edge_idx_perbatch, edge_feat_tensor_perbatch, 1, numoftile_perbatch*numoftile_perbatch,numofdummynode)
    #====================================================================================================
    model = model.to(device)
    model.eval()

    #====================================================================================================
    if idx is None:
        idx = random.randint(0, len(input_test_tensor)-1)

    input_batch_tensor_feat = input_test_tensor[idx:idx+1,  :,:cell_dim*cell_dim].double().to(device)
    input_batch_tensor      = input_test_tensor[idx:idx+1,  :,:].double().to(device)
    output_batch_tensor     = output_test_tensor[idx:idx+1, :,:].double().to(device)


    pred_np = model(input_batch_tensor, edge_idx_test, edge_feat_test, bc_dummynode, dummy_node ).clone().detach().cpu().numpy().reshape(output_batch_tensor.shape).squeeze()

    input_np_reorg  = reorgnize_graph(input_batch_tensor_feat.squeeze(),numoftile_perbatch, numoftile_perbatch,chipinfo)
    output_np_reorg = reorgnize_graph(output_batch_tensor.squeeze(),numoftile_perbatch, numoftile_perbatch,chipinfo)
    pred_np_reorg   = reorgnize_graph(pred_np,numoftile_perbatch, numoftile_perbatch,chipinfo)

    pred_np_reorg   = gaussian_filter(pred_np_reorg, sigma=1)

    output_np_reorg = normalize(output_np_reorg)
    pred_np_reorg   = normalize(pred_np_reorg)


    abs_err , _= comparepredict_golden(pred_np_reorg, output_np_reorg)


    # if os.path.exists('./result/testcase{}'.format(idx)):
        # os.system('rm -rf ./result/testcase{}'.format(idx))
    os.system('mkdir -p ./result/testcase{}'.format(idx))
    if dummy_node is not None:
        print('Bad edge Abs err  :', np.mean(abs_err).item())
        print('Bad edge Loss :', mean_squared_error(pred_np_reorg, output_np_reorg).item())
        plot_im(plot_data = kmap_curr,            title="", save_name='./result/testcase{}/0_kmap_bad.png'.format(idx),show=False, range01=False)
        plot_im(plot_data = input_np_reorg,       title="", save_name='./result/testcase{}/1_input_bad.png'.format(idx),show=False, range01=False)
        plot_im(plot_data = output_np_reorg,      title="", save_name='./result/testcase{}/2_golden_bad.png'.format(idx),show=False, range01=False)
        plot_im(plot_data = pred_np_reorg,        title="", save_name='./result/testcase{}/3_prediction_bad.png'.format(idx),show=False, range01=False)
        plot_im(plot_data = abs_err,              title="", save_name='./result/testcase{}/4_err_bad.png'.format(idx),show=False, range01=True)
    else:
        print('Abs err  :', np.mean(abs_err).item())
        print('Loss :', mean_squared_error(pred_np_reorg, output_np_reorg).item())
        plot_im(plot_data = kmap_curr,            title="", save_name='./result/testcase{}/0_kmap.png'.format(idx),show=False, range01=False)
        plot_im(plot_data = input_np_reorg,       title="", save_name='./result/testcase{}/1_input.png'.format(idx),show=False, range01=False)
        plot_im(plot_data = output_np_reorg,      title="", save_name='./result/testcase{}/2_golden.png'.format(idx),show=False, range01=False)
        plot_im(plot_data = pred_np_reorg,        title="", save_name='./result/testcase{}/3_prediction.png'.format(idx),show=False, range01=False)
        plot_im(plot_data = abs_err,              title="", save_name='./result/testcase{}/4_err.png'.format(idx),show=False, range01=True)





    






#====================================================================================================
# main_
#====================================================================================================
input_dim   = cell_dim*cell_dim*2+1
out_dim     = cell_dim*cell_dim
model_graph = gnn_top(c_in =input_dim,  c_out = out_dim ).double().to(device)
#====================================================================================================


#====================================================================================================
chipshape = np.zeros((numoftile_x,numoftile_y))
#====================================================================================================
chipres = np.ones((numoftile_x, numoftile_y, 2))*cell_dim
chipres = chipres.astype(int)
#====================================================================================================
chipinfo                       = dict()
chipinfo['chipshape']          = chipshape
chipinfo['chipres']            = chipres
chipinfo['numoftile_x']        = numoftile_x
chipinfo['numoftile_y']        = numoftile_y
chipinfo['numoftile_perbatch'] = numoftile_perbatch
chipinfo['tile_dim']           = cell_dim
analysis_chip(chipinfo)
analysis_chip_batch(chipinfo, numoftile_perbatch =numoftile_perbatch)

#====================================================================================================
if generate_new_sample:
    print('INFO: Generating samples for :GNN...')
    input_batch_tensor, output_batch_tensor ,powermap_total, kmap_total   = generate_edge_training_samples(chipinfo = chipinfo, load_old = False)
    edge_tensor, edge_feat_tensor = generate_batch_edge(chipinfo)
    data = (input_batch_tensor, output_batch_tensor,edge_tensor, edge_feat_tensor, powermap_total ,kmap_total)
    with open('./pickles/newbatchdata{}x{}.pickle'.format(numoftile_x,numoftile_y), "wb") as f: pickle.dump((data), f,protocol=pickle.HIGHEST_PROTOCOL)


#====================================================================================================
with open('./pickles/newbatchdata{}x{}.pickle'.format(numoftile_x,numoftile_y), "rb") as f: data=pickle.load(f)
if train_newgnn:
    train_gnn(model_graph, data, numofepoch, chipinfo, name ='mid')
    torch.save(model_graph.state_dict(), './{}.pt'.format("trained_model"))


print('INFO: GNN training is done')
#====================================================================================================
# testing
#====================================================================================================

for i in range(3):
    test_gnn(model_graph, data, chipinfo, show=True, idx=i, name ='best_trained_model_mid' )

#====================================================================================================
# training dummy node
#====================================================================================================
print('INFO: Inserting defects to chip')
generate_random_badedge(chipinfo)
edge_tensor, edge_feat_tensor   = generate_batch_edge(chipinfo)

if generate_sample_badedge:
    print('INFO: Generating samples with defects...')
    _,_ ,_, _, powermap,kmap = data 
    input_batch_tensor, output_batch_tensor,_ ,_= generate_edge_training_samples(chipinfo = chipinfo, load_old = False, adddummynode=False,powermap=powermap,kmap=kmap )
    data_bad = (input_batch_tensor, output_batch_tensor,edge_tensor, edge_feat_tensor,powermap,kmap )
    with open('./pickles/newbatchdata_badedge{}x{}.pickle'.format(numoftile_x,numoftile_y), "wb") as f: pickle.dump((data_bad), f,protocol=pickle.HIGHEST_PROTOCOL)
#====================================================================================================
with open('./pickles/newbatchdata_badedge{}x{}.pickle'.format(numoftile_x,numoftile_y), "rb") as f: data_bad=pickle.load(f)
if train_defectnode:
    train_defect_node(model_graph, data_bad, numofepoch, chipinfo, name ='best_trained_model_mid')

#====================================================================================================
with open('./pickles/dummynode_best.pickle', "rb") as f: dummynode_feat=pickle.load(f)
for i in range(3):
    test_gnn(model_graph, data_bad, chipinfo, show=True, idx=i, name ='best_trained_model_mid', dummy_node= dummynode_feat )

