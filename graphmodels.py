import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean



#====================================================================================================
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
# from ..inits import glorot, zeros
#====================================================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _weights_init_4(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_uniform_(m.weight)
        # nn.init.constant_(m.bias, 0)






class mlp(nn.Module):
    # def __init__(self, input_dim ,out_dim, numofhiddenlayer = 4, dim_l=[64,64,64,64], num_heads=1):

    def __init__(self, input_dim ,out_dim,  dim=1000, num_heads=1):
        super(mlp, self ).__init__()
        self.fc1 = nn.Linear(808,     dim*num_heads)
        self.fc2 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc3 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc4 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc5 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc6 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc7 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc8 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc9 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc10 = nn.Linear(dim*num_heads, 800)
        self.apply(_weights_init_4)    


    # x represents our data
    def forward(self, x, edge, activate= True):
        # print(x.shape)
        # exit()
        # act = nn.ReLU()
        # act = nn.LeakyReLU()

        act = nn.Tanh()
        # act = nn.SELU()
        x = x.reshape((len(x), 808))

        x = act(self.fc1(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        x = act(self.fc4(x))
        x = act(self.fc5(x))
        x = act(self.fc6(x))
        x = act(self.fc7(x))
        x = act(self.fc8(x))
        x = act(self.fc9(x))
        output = self.fc10(x)
        return output








class f_v(nn.Module):
    # def __init__(self, input_dim ,out_dim, numofhiddenlayer = 4, dim_l=[64,64,64,64], num_heads=1):

    def __init__(self, input_dim ,out_dim,  dim=64, num_heads=1):
        super(f_v, self ).__init__()


        #self.layers   = [nn.Linear(input_dim,  dim_l[0])]
        #self.bnlayers = [nn.BatchNorm1d(dim_l[0])]
        #self.lnlayers = [nn.LayerNorm(dim_l[0])]
        #assert len(dim_l)==numofhiddenlayer
        #for i in range(1,numofhiddenlayer):
        #    self.layers.append(nn.Linear(dim_l[i-1], dim_l[i]))
        #    self.bnlayers.append([nn.BatchNorm1d(dim_l[i])])
        #    self.lnlayers.append([nn.LayerNorm(dim_l[i])])


        #self.layers.append(nn.Linear(dim_l[-1], out_dim))
        ##====================================================================================================
        #for fclayer in self.layers:
        #    if isinstance(fclayer, nn.Linear):
        #        nn.init.xavier_normal_(fclayer.weight)
        #        # nn.init.xavier_uniform_(m.weight)
        #        nn.init.constant_(fclayer.bias, 0)
        #====================================================================================================



        self.fc1 = nn.Linear(input_dim,     dim*num_heads)
        self.fc2 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc3 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc4 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc5 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc6 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc7 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc8 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc9 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc10 = nn.Linear(dim*num_heads, out_dim)
        self.apply(_weights_init_4)    


        self.bn1 = nn.BatchNorm1d(dim*num_heads)
        self.bn2 = nn.BatchNorm1d(dim*num_heads)
        self.bn3 = nn.BatchNorm1d(dim*num_heads)
        self.bn4 = nn.BatchNorm1d(dim*num_heads)
        self.bn5 = nn.BatchNorm1d(dim*num_heads)
        self.bn6 = nn.BatchNorm1d(dim*num_heads)
        self.bn7 = nn.BatchNorm1d(dim*num_heads)
        self.bn8 = nn.BatchNorm1d(dim*num_heads)
        self.bn9 = nn.BatchNorm1d(dim*num_heads)
        self.bn10 = nn.BatchNorm1d(out_dim)

        self.ln1 = nn.LayerNorm(dim*num_heads)
        self.ln2 = nn.LayerNorm(dim*num_heads)
        self.ln3 = nn.LayerNorm(dim*num_heads)
        self.ln4 = nn.LayerNorm(dim*num_heads)
        self.ln5 = nn.LayerNorm(dim*num_heads)
        self.ln6 = nn.LayerNorm(dim*num_heads)
        self.ln7 = nn.LayerNorm(dim*num_heads)
        self.ln8 = nn.LayerNorm(dim*num_heads)
        self.ln9 = nn.LayerNorm(dim*num_heads)
        self.ln10 = nn.LayerNorm(out_dim)



    # x represents our data
    def forward(self, x, activate= True):
        # act = nn.ReLU()
        act = nn.LeakyReLU()

        drop = nn.Dropout(p=0.2)
        # act = nn.Tanh()
        # act = nn.SELU()



        # for idx, layer in enumerate(self.layers):
        #     if idx==len(self.layers)-1:
        #         break
        #     x=F.leaky_relu(layer(x)) 

        # out = self.layers[-1](x)
        # return out

        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))
        # x = F.tanh(self.fc4(x))
        # x = F.tanh(self.fc5(x))
        # x = F.tanh(self.fc6(x))
        # x = F.tanh(self.fc7(x))
        # x = F.tanh(self.fc8(x))
        # x = F.tanh(self.fc9(x))

        # x = act(self.fc1(x))
        # x = act(self.fc2(x))
        # x = act(self.fc3(x))
        # x = act(self.fc4(x))
        # x = act(self.fc5(x))
        # x = act(self.fc6(x))
        # x = act(self.fc7(x))
        # x = act(self.fc8(x))
        # x = act(self.fc9(x))


        # x = F.leaky_relu(self.ln1(self.fc1(x)))
        # x = F.leaky_relu(self.ln2(self.fc2(x)))
        # x = F.leaky_relu(self.ln3(self.fc3(x)))
        # x = F.leaky_relu(self.ln4(self.fc4(x)))
        # x = F.leaky_relu(self.ln5(self.fc5(x)))
        # x = F.leaky_relu(self.ln6(self.fc6(x)))
        # x = F.leaky_relu(self.ln7(self.fc7(x)))




        # x = F.leaky_relu(self.bn8(self.fc8(x)))
        # x = F.leaky_relu(self.bn9(self.fc9(x)))



        x = F.tanh(self.ln1(self.fc1(x)))
        x = F.tanh(self.ln2(self.fc2(x)))
        x = F.tanh(self.ln3(self.fc3(x)))
        # x = F.tanh(self.ln4(self.fc4(x)))
        # x = drop(x)
        # x = F.tanh(self.ln5(self.fc5(x)))
        # # x = drop(x)
        # x = F.tanh(self.ln6(self.fc6(x)))
        # x = drop(x)
        # x = F.tanh(self.ln7(self.fc7(x)))
        # x = F.tanh(self.ln8(self.fc8(x)))
        # x = F.selu(self.ln9(self.fc9(x)))


        if activate:
            # output = act(self.fc10(x))
            # output = F.tanh(self.fc10(x))
            output = F.leaky_relu(self.bn10(self.fc10(x)))
            # output = F.selu(self.ln10(self.fc10(x)))
            # output = F.tanh(self.ln10(self.fc10(x)))
        else :
            output = self.fc10(x)
        return output





class f_e(nn.Module):

    def __init__(self, input_dim ,out_dim,  dim=64, num_heads=1):
        super(f_e, self ).__init__()

        self.fc1 = nn.Linear(input_dim,     dim*num_heads)
        self.fc2 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc3 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc4 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc5 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc6 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc7 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc8 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc9 = nn.Linear(dim*num_heads, dim*num_heads)
        self.fc10 = nn.Linear(dim*num_heads, out_dim)
        self.apply(_weights_init_4)    


        self.bn1 = nn.BatchNorm1d(dim*num_heads)
        self.bn2 = nn.BatchNorm1d(dim*num_heads)
        self.bn3 = nn.BatchNorm1d(dim*num_heads)
        self.bn4 = nn.BatchNorm1d(dim*num_heads)
        self.bn5 = nn.BatchNorm1d(dim*num_heads)
        self.bn6 = nn.BatchNorm1d(dim*num_heads)
        self.bn7 = nn.BatchNorm1d(dim*num_heads)
        self.bn8 = nn.BatchNorm1d(dim*num_heads)
        self.bn9 = nn.BatchNorm1d(dim*num_heads)
        self.bn10 = nn.BatchNorm1d(out_dim)

        self.ln1 = nn.LayerNorm(dim*num_heads)
        self.ln2 = nn.LayerNorm(dim*num_heads)
        self.ln3 = nn.LayerNorm(dim*num_heads)
        self.ln4 = nn.LayerNorm(dim*num_heads)
        self.ln5 = nn.LayerNorm(dim*num_heads)
        self.ln6 = nn.LayerNorm(dim*num_heads)
        self.ln7 = nn.LayerNorm(dim*num_heads)
        self.ln8 = nn.LayerNorm(dim*num_heads)
        self.ln9 = nn.LayerNorm(dim*num_heads)
        self.ln10 = nn.LayerNorm(out_dim)



    # x represents our data
    def forward(self, x, activate= True):
        # act = nn.ReLU()
        act = nn.LeakyReLU()
        # act = nn.Tanh()
        # act = nn.SELU()
        drop = nn.Dropout(p=0.2)


        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))
        # x = F.tanh(self.fc4(x))
        # x = F.tanh(self.fc5(x))
        # x = F.tanh(self.fc6(x))
        # x = F.tanh(self.fc7(x))
        # x = F.tanh(self.fc8(x))
        # x = F.tanh(self.fc9(x))

        # x = act(self.fc1(x))
        # x = act(self.fc2(x))
        # x = act(self.fc3(x))
        # x = act(self.fc4(x))
        # x = act(self.fc5(x))
        # x = act(self.fc6(x))
        # x = act(self.fc7(x))
        # x = act(self.fc8(x))
        # x = act(self.fc9(x))


        # x = F.leaky_relu(self.ln1(self.fc1(x)))
        # x = F.leaky_relu(self.ln2(self.fc2(x)))
        # x = F.leaky_relu(self.ln3(self.fc3(x)))
        # x = F.leaky_relu(self.ln4(self.fc4(x)))
        # x = F.leaky_relu(self.ln5(self.fc5(x)))
        # x = F.leaky_relu(self.ln6(self.fc6(x)))
        # x = F.leaky_relu(self.bn7(self.fc7(x)))
        # x = F.leaky_relu(self.bn8(self.fc8(x)))
        # x = F.leaky_relu(self.bn9(self.fc9(x)))



        x = F.tanh(self.ln1(self.fc1(x)))
        x = F.tanh(self.ln2(self.fc2(x)))
        x = F.tanh(self.ln3(self.fc3(x)))
        # x = F.tanh(self.ln4(self.fc4(x)))
        # x = drop(x)
        # x = F.tanh(self.ln5(self.fc5(x)))
        # x = drop(x)
        # x = F.tanh(self.ln6(self.fc6(x)))
        # x = drop(x)
        # x = F.tanh(self.ln7(self.fc7(x)))
        # x = F.tanh(self.ln8(self.fc8(x)))
        # x = F.selu(self.ln9(self.fc9(x)))


        if activate:
            # output = act(self.fc10(x))
            # output = F.tanh(self.fc10(x))
            output = F.leaky_relu(self.bn10(self.fc10(x)))
            # output = F.selu(self.ln10(self.fc10(x)))
            # output = F.tanh(self.ln10(self.fc10(x)))
        else :
            output = self.fc10(x)
        return output











class gnn_top(nn.Module):
    def __init__(self, c_in, c_out, num_heads=1):
        super().__init__()
        # self.gnn_0=gnn(c_in, c_out)
        interdim =512 
        self.gnn_0=gnn(c_in, interdim)
        self.gnn_1=gnn(interdim, interdim)
        self.gnn_2=gnn(interdim, c_out)
        # self.gnn_3=gnn(interdim, interdim)
        # self.gnn_4=gnn(interdim, c_out)
        # self.gnn_5=gnn(interdim, interdim)
        # self.gnn_6=gnn(interdim, interdim)
        # self.decoder = f_e(interdim*2 ,c_out, dim=128 )


        # self.gnn_1=gnn(interdim, interdim)
        # self.gnn_2=gnn(interdim, c_out)
        # self.gnn_3=gnn(interdim, c_out)
        

        
    def forward(self,node_feats,edges,  edge_id, bc_dummy_node=None, dummy_node_feat=None):

        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        x = node_feats.reshape(batch_size * num_nodes,  -1)

        num_normalnode = len(x)


        #====================================================================================================
        if bc_dummy_node is not None:
            x = torch.vstack([x, bc_dummy_node])


        if dummy_node_feat is not None:
            x = torch.vstack([x, dummy_node_feat])
        #====================================================================================================


        x=self.gnn_0(x,edges,edge_id)
        x=self.gnn_1(x,edges,edge_id)
        x=self.gnn_2(x,edges,edge_id)
        # x=self.gnn_3(x,edges,edge_id)
        # x=self.gnn_4(x,edges,edge_id)
        x=x[:num_normalnode,:]
        return x



        edge_indices_row = edges[0,:]
        edge_indices_col = edges[1,:]
        #====================================================================================================
        # decoder part
        #====================================================================================================
        concat_input = torch.cat(
            [
                torch.index_select(input=x, index=edge_indices_row, dim=0),
                torch.index_select(input=x, index=edge_indices_col, dim=0),
            ],
            dim=-1,
        )  

        edge_feat = self.decoder(concat_input, False)
        x = scatter_add(src=edge_feat,  dim=0,index=edge_indices_row )         # shape [batch*num_nodes, embed_size]
        return x



        # x=self.gnn_2(x,edges)
        # x=self.gnn_3(x,edges)
        # return x




















class gnn(nn.Module):
    def __init__(self, c_in, c_out, num_heads=1):
        super().__init__()
        self.num_heads = num_heads


        self.f_e = f_e(c_in*2+2,  256,  dim=512   )
        self.f_v = f_v(c_in+256, c_out ,dim=512 )


        # self.f_e = f_e(c_in*2+2,  128,  dim=128   )
        # self.f_v = f_v(c_in+128, c_out ,dim=128 )


        # self.f_v = f_e(c_in+c_in, 128 ,dim=128 )
        # self.f_final = f_e(c_out , c_out  )
        # self.f_ini   = f_e(c_in ,c_in, dim=128 )
        # self.apply(_weights_init_4)    








    # def forward(self, node_feats, adj_matrix, print_attn_probs=False):
    def forward(self, node_feats, edges,edge_id):

        node_feats_flat = node_feats
        # node_feats_flat = self.f_ini(node_feats_flat, False ) 

        edge_indices_row = edges[0,:]
        edge_indices_col = edges[1,:]

        concat_input = torch.cat(
            [
                torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
                torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0),
                edge_id
            ],
            dim=-1,
        )  
        # print(edge_indices_row[:10])
        # print(edge_indices_col[:10])

        # concat_input shape :   batch*numofnodes_perbatch,   len(feat) + len(feat)
        edge_encode = self.f_e(concat_input, False) # size should be     batch*numofnodes_perbatch, length of output_edge encoding 
        # print(edge_encode[:10,:5])
        #====================================================================================================
        # aggr
        #====================================================================================================
        # tmp = torch.index_select(input=edge_encode, dim=0,index=edge_indices_col) # shape [batch*num_edges, embed_size ]

        # print(tmp[:10,:5])
        # assert torch.equal(edge_encode, tmp)
        # print(edge_encode.shape)
        # print(tmp.shape)
        # print(edge_encode[5000,:])
        # print(tmp[5000,:])
        # exit()


        aggr_encode = scatter_add(src=edge_encode,          dim=0,index=edge_indices_row )         # shape [batch*num_nodes, embed_size]


        # aggr_encode_core = torch.index_select(input = aggr_encode,     index =corenode_idx ,dim=0)
        # node_feats_core  = torch.index_select(input = node_feats_flat, index =corenode_idx ,dim=0)
        




        # aggr_encode = scatter_mean(src=tmp, index=edge_indices_col, dim=0)         # shape [batch*num_nodes, embed_size]
        # print(aggr_encode[:10,:5])
        # exit()
        # assert len(aggr_encode)==len(node_feats_flat)



        # print(node_feats_flat.shape)
        # print(aggr_encode.shape)
        delta = len(node_feats_flat)-len(aggr_encode)
        aggr_encode = torch.vstack([aggr_encode, torch.zeros(delta, aggr_encode.shape[1]).double().to(device) ])
        # print(aggr_encode.shape)
        # exit()
        #====================================================================================================
        node_feats_next = torch.hstack([node_feats_flat, aggr_encode])   # batch*num_nodes_per_batch,   embed size
        # node_feats_next = torch.hstack([node_feats_core, aggr_encode_core])   # batch*num_nodes_per_batch,   embed size
        #====================================================================================================
        # out = self.f_v(node_feats_next, True)  # out shape :  batch*num_nodes_per_batch,  out_feat 
        out = self.f_v(node_feats_next, False)  # out shape :  batch*num_nodes_per_batch,  out_feat 

        # out = self.f_final(out,False)
        return out



