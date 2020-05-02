import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb

def knn(points, queries, K):
    """
    Args:
        points ( B x N x 3 tensor )
        query  ( B x M x 3 tensor )  M < N
        K      (constant) num of neighbors
    Outputs:
        knn    (B x M x K x 3 tensor) sorted K nearest neighbor
        indice (B x M x K tensor) knn indices   
    """
    value = None
    indices = None
    num_batch = points.shape[0]
    for i in range(num_batch):
        point = points[i]
        query = queries[i]
        dist  = torch.cdist(point, query)
        idxs  = dist.topk(K, dim=0, largest=False, sorted=True).indices
        idxs  = idxs.transpose(0,1)
        nn    = point[idxs].unsqueeze(0)
        value = nn if value is None else torch.cat((value, nn))

        idxs  = idxs.unsqueeze(0)
        indices = idxs if indices is None else torch.cat((indices, idxs))
        
    return value, indices

def gather_feature(features, indices):
    """
    Args:
        features ( B x N x F tensor) -- feature from previous layer
        indices  ( B x M x K tensor) --  represents queries' k nearest neighbor
    Output:
        features ( B x M x K x F tensor) -- knn features from previous layer 
    """
    res = None
    num_batch = features.shape[0]
    #pdb.set_trace()
    for B in range(num_batch):
        knn_features = features[B][indices[B]].unsqueeze(0)
        res = knn_features if res is None else torch.cat((res, knn_features))
    return res


class Lift(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(Lift, self).__init__()    
        """
        Args:
            input ( B x M x K x 3  tensor ) -- subtraction vectors 
                from query to its k nearest neighbor
        Output: 
            local point feature ( B x M x K x 64 tensor ) 
        """
        self.is_batchnorm = is_batchnorm

        self.bn1 = nn.BatchNorm2d(3)
        self.linear1 = nn.Linear(in_size, out_size // 2)
        self.bn2 = nn.BatchNorm2d(out_size // 2)
        self.linear2 = nn.Linear(out_size // 2, out_size)


    def forward(self, inputs):

        if self.is_batchnorm == True:
            outputs = self.bn1(inputs.transpose(1,3)).transpose(1,3)
            outputs = F.relu(self.linear1(outputs))
            outputs = self.bn2(outputs.transpose(1,3)).transpose(1,3)
            outputs = F.relu(self.linear2(outputs))
            return outputs

        else:
            outputs = F.relu(self.linear1(inputs))
            outputs = F.relu(self.linear2(outputs))
            return outputs

class ShellConv(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division, 
            is_batchnorm=True):
        super(ShellConv, self).__init__() 
        """
        out_features  (int) num of output feature (dim = -1)
        prev_features (int) num of prev feature (dim = -1)
        neighbor      (int) num of nearest neighbor in knn
        division      (int) num of division
        """

        self.K = neighbor
        self.S = int(self.K/division) # num of feaure per shell
        self.point_feature = 64
        self.neighbor = neighbor
        in_channel = self.point_feature + prev_features
        out_channel = out_features

        self.lift = Lift(3, self.point_feature)
        self.maxpool = nn.MaxPool2d((1, self.S), stride = (1, self.S))
        if is_batchnorm == True:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, out_channel, (1, division))
            )
        else:
            self.conv = nn.Conv2d(in_channel, out_channel, (1, division))
        
    def forward(self, points, queries, feat_prev):
        """
        Args:
            points      (B x N x 3 tensor)
            query       (B x M x 3 tensor) -- note that M < N
            feat_prev   (B x N x F1 tensor)
        Outputs:
            feat        (B x M x F2 tensor)
        """

        nn_pts, idxs = knn(points, queries, self.K)
        nn_center    = queries.unsqueeze(2)
        nn_points_local = nn_center - nn_pts

        nn_feat_local = self.lift(nn_points_local)

        # shape: B x M x K x F
        if feat_prev is not None:
            feat_prev = gather_feature(feat_prev, idxs)
            feat_cat  = torch.cat((nn_feat_local, feat_prev), dim = -1)
        else:
            feat_cat  = nn_feat_local 

        feat_cat = feat_cat.permute(0,3,1,2) # BMKF -> BFMK        
        feat_max = self.maxpool(feat_cat)
        output   = self.conv(feat_max).permute(0,2,3,1)

        return output.squeeze(2)

