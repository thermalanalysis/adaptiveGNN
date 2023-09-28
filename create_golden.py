import re
import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
import sys
import torch
import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
from util import *
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=50_000)

class golden():
    def __init__(self,xgrid,ygrid,thermalbc=False, bc_locidx = None):
        self.xnumofnode = xgrid
        self.ynumofnode = ygrid


        self.numoflayer = 1
        # self.numoflayer = 3
        self.imc_size               = 1/self.xnumofnode
        self.p = np.zeros((self.numoflayer,self.xnumofnode,self.ynumofnode))
        self.edge_pair= None
        self.badedge= None
        self.edge_weight= None
        self.thermalbc = thermalbc
        self.bc_locidx = bc_locidx
        self.bc_pair = None









# def plot_im(plot_data, title, save_name, vmin, vmax):
#     fig,ax = plt.subplots()
#     width_plot = 100
#     im = ax.imshow(plot_data , cmap = 'jet',vmin=vmin, vmax=vmax)
#     ax.set_title(title, pad=20)
#     fig.colorbar(im)
#     # fig.set_size_inches(len_plot*0.09,width_plot*0.09) # convert to inches, 100->4 inches
#     fig.savefig(save_name, bbox_inches='tight',dpi=100)
#     plt.close()
#     # plt.show()

#====================================================================================================
#====================================================================================================
    def checkneighbours(self,idxx,idxy,idxz,numx,numy,numz):
        if idxx<0 or idxx>=numx:
            return False
        if idxy<0 or idxy>=numy:
            return False
        if idxz<0 or idxz>=numz:
            return False
        return True
    def id_(self,idxx, idxy, idxz, numx, numy, numz):
        return idxz*(numx*numy)+ idxx*(numy)+ idxy
    def load_conductivity(self):
        return self.k

        #====================================================================================================
    def load_map(self):
        #====================================================================================================
        chipmap =np.array([['imc']*self.ynumofnode]*self.xnumofnode).reshape(self.numoflayer, self.xnumofnode, self.ynumofnode)
        return chipmap



    def get_conductance_G(self,k,chipmap,skipededgepair, edge_pair=None,edge_weight=None):
        

        k_tobc = 1
        #====================================================================================================
        edges_l, edgemap, badedge = self.edges_l, self.edgemap, self.badedge
        #====================================================================================================

        xnumofnode = self.xnumofnode
        ynumofnode = self.ynumofnode
        numoflayer = self.numoflayer
        imc_size               = 1/self.xnumofnode
        dict_size = dict()
        dict_size["imc"]       = (self.imc_size,self.imc_size)
        dict_size["imc_power"] = (self.imc_size,self.imc_size)
        #====================================================================================================
        if self.thermalbc:
            numtotalnode = (xnumofnode+1)*ynumofnode*numoflayer              # +1 for edge boundary thermal
        else :
            if self.bc_pair is not None:
                numtotalnode = xnumofnode*ynumofnode*numoflayer+1              # +1 for edge boundary thermal
            else:
                numtotalnode = xnumofnode*ynumofnode*numoflayer
                 

        bc_nodeid = numtotalnode-1
        G_sparse = sparse_mat.dok_matrix((numtotalnode, numtotalnode))
        #====================================================================================================
        direction_map = dict()
        direction_map[0]=(0,1,0)  #east,0
        direction_map[1]=(0,-1,0) #west,1
        direction_map[2]=(1,0,0)  #north,2
        direction_map[3]=(-1,0,0) #south,3
        direction_map[4]=(0,0,1)  #top,4
        direction_map[5]=(0,0,-1) #bottom,5
        #====================================================================================================
        nodeid = 0
        edgecount=0
        for z_idx in range(numoflayer):
            for x_idx in range(xnumofnode):
                for y_idx in range(ynumofnode):

                    center_nodetype = chipmap[z_idx,x_idx,y_idx]
                    centernode_length, centernode_width = dict_size[center_nodetype]
                    centerlayer_height = imc_size



                    centernodeid = self.id_(x_idx, y_idx, z_idx, xnumofnode, ynumofnode,numoflayer )
                    assert centernodeid==nodeid
                    G_sparse[centernodeid, centernodeid]=0
                    #====================================================================================================
                    # check all neighbours and calculate conductance 
                    #====================================================================================================
                    center_k = k[z_idx, x_idx, y_idx]
                    # assert center_k==0.1


                    #====================================================================================================
                    # add input thermal
                    if self.thermalbc and x_idx==self.bc_locidx:
                        boundary_id = xnumofnode*ynumofnode + y_idx
                        G_sparse[centernodeid, boundary_id]=1
                        G_sparse[boundary_id, centernodeid]=1
                        # G_sparse[centernodeid, centernodeid]=1
                        
                    #====================================================================================================
                    #0919
                    #====================================================================================================
                    if self.bc_pair is not None and (x_idx==0 or x_idx==xnumofnode-1 or y_idx==0 or y_idx== ynumofnode-1):

                        G_sparse[centernodeid, centernodeid]+=k_tobc
                        G_sparse[centernodeid, bc_nodeid]   =-k_tobc
                        G_sparse[bc_nodeid, centernodeid]   =-k_tobc

                    #====================================================================================================

                    for direction in range(6):
                        neighbour_x_idx= (direction_map[direction][0]+x_idx)
                        neighbour_y_idx= (direction_map[direction][1]+y_idx)
                        neighbour_z_idx= (direction_map[direction][2]+z_idx)
                        if self.checkneighbours(neighbour_x_idx,neighbour_y_idx,neighbour_z_idx,xnumofnode, ynumofnode,numoflayer):
                            neighbour_id = self.id_(neighbour_x_idx, neighbour_y_idx,neighbour_z_idx, xnumofnode, ynumofnode,numoflayer)

                            smallid = min(centernodeid,  neighbour_id)
                            largeid = max(centernodeid,  neighbour_id)

                            myedge_weight = 1
                            #====================================================================================================
                            if edge_weight is not None and (smallid, largeid) in edgemap:
                                edge_idx, edge_loc_idx = edgemap[(smallid,largeid)]
                                myedge_weight          = edge_weight[edge_idx][edge_loc_idx].item()

                            if badedge is not None and (smallid, largeid) in badedge:
                                # print(smallid, largeid)
                                myedge_weight = 0

                            #====================================================================================================
                            #0919
                            if (self.bc_pair is not None and (smallid,largeid) in self.bc_pair) :

                                # add connection to the bc node
                                G_sparse[centernodeid, centernodeid]+=k_tobc
                                G_sparse[centernodeid, bc_nodeid]   =-k_tobc
                                G_sparse[bc_nodeid, centernodeid]   =-k_tobc






                            #====================================================================================================
                            # calcualte the avg k
                            #====================================================================================================
                            neighbour_k                                 = k[neighbour_z_idx,neighbour_x_idx,neighbour_y_idx]
                            # assert neighbour_k==0.1

                            neighbour_nodetype                          = chipmap[neighbour_z_idx,neighbour_x_idx,neighbour_y_idx]
                            neighbour_node_length, neighbour_node_width = dict_size[neighbour_nodetype]
                            #====================================================================================================
                            if direction==0 or direction==1:
                                assert centernode_width==neighbour_node_width
                                d1 = centernode_length/2
                                d2 = neighbour_node_length/2
                                A = centernode_width*centerlayer_height
                            elif direction==2 or direction==3:
                                assert centernode_length==neighbour_node_length
                                d1 = centernode_width/2
                                d2 = neighbour_node_width/2
                                A = centernode_length*centerlayer_height
                            else:
                                assert centernode_width==neighbour_node_width
                                assert centernode_length==neighbour_node_length
                                if neighbour_z_idx==3:
                                    neighbourlayer_height = 0.0001
                                else:
                                    neighbourlayer_height = imc_size
                                d1 = centerlayer_height/2
                                d2 = neighbourlayer_height/2
                                A = centernode_width*centernode_length

                            dd = d1+d2
                            #====================================================================================================
                            k_avg = dd/(d1/center_k+ d2/neighbour_k)
                            # G_ = round((k_avg*A)/dd,6)
                            # G_ = (k_avg*A)/dd
                            G_ = (k_avg*A)/dd * myedge_weight

                            G_sparse[centernodeid, centernodeid]+=G_
                            G_sparse[centernodeid, neighbour_id]=-G_


                    nodeid+=1
        #====================================================================================================
        return G_sparse




    def gen(self,edges_l, edgemap, skipededgepair=None,edge_thermal=None):

        map_        = self.load_map()
        k           = self.load_conductivity()
        edge_pair   = self.edge_pair
        edge_weight = self.edge_weight
        badedge     = self.badedge
        self.edges_l = edges_l
        self.edgemap = edgemap



        G_sparse = self.get_conductance_G(k=k,chipmap=map_,edge_pair=edge_pair,edge_weight=edge_weight,skipededgepair=skipededgepair)
        G_sparse   = G_sparse.tocsc()
        p = self.p.reshape((self.numoflayer*self.xnumofnode*self.ynumofnode,1))



        if self.bc_pair is not None:

            newp = np.ones((self.numoflayer*self.xnumofnode*self.ynumofnode+1,1))
            newp[:self.numoflayer*self.xnumofnode*self.ynumofnode,:]= p
            newp[self.numoflayer*self.xnumofnode*self.ynumofnode:,:]= 298
            p=newp


        #====================================================================================================
        if self.thermalbc:
            assert edge_thermal is not None
            newp = np.ones((self.numoflayer*(self.xnumofnode+1)*self.ynumofnode,1))
            newp[:self.numoflayer*self.xnumofnode*self.ynumofnode,:]= p
            newp[self.numoflayer*self.xnumofnode*self.ynumofnode:,:]= edge_thermal
            p=newp
        #====================================================================================================





        p = sparse_mat.csc_matrix(p)




        I = sparse_mat.identity(G_sparse.shape[0]) * 1e-4
        # I = sparse_mat.identity(self.numoflayer*self.xnumofnode*self.ynumofnode) * 1e-4
        # for i in range(self.numoflayer*self.xnumofnode*self.ynumofnode):
        #     G_sparse[i,i]+= 1e-4



        G_sparse = G_sparse + I
        G_sparse[-1,-1]=0
        # G_sparse = G_sparse 
        t = sparse_algebra.spsolve(G_sparse,p,permc_spec=None,use_umfpack=True)
        if self.bc_pair is not None:
            t = t[:-1]

        if self.thermalbc:
            t = t.reshape(self.numoflayer,self.xnumofnode+1,self.ynumofnode)
        else :
            t = t.reshape(self.numoflayer,self.xnumofnode,self.ynumofnode)
        # print(t.min(),t.max())
        # print(t)
        # exit()
       
        return t


