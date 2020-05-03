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
    for B in range(num_batch):
        knn_features = features[B][indices[B]].unsqueeze(0)
        res = knn_features if res is None else torch.cat((res, knn_features))
    return res

def random_sample(points, num_sample):
    """
    Args:
        points ( B x N x 3 tensor )
        num_sample (constant)
    Outputs:
        sampled_points (B x num_sample x 3 tensor)
    """    
    perm = torch.randperm(points.shape[1])
    return points[:, perm[:num_sample]].clone()

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

        self.batchnorm1 = nn.BatchNorm2d(3)
        self.linear1 = nn.Sequential(
            nn.Linear(in_size, out_size // 2),
            nn.ReLU()
        )
        self.batchnorm2 = nn.BatchNorm2d(out_size // 2)
        self.linear2 = nn.Sequential(
            nn.Linear(out_size // 2, out_size),
            nn.ReLU()
        )
        
    def forward(self, inputs):

        if self.is_batchnorm == True:
            outputs = self.batchnorm1(inputs.transpose(1,3)).transpose(1,3)
            outputs = self.linear1(outputs)
            outputs = self.batchnorm2(outputs.transpose(1,3)).transpose(1,3)
            outputs = self.linear2(outputs)
            return outputs

        else:
            outputs = self.linear1(inputs)
            outputs = self.linear2(outputs)
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
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )
        
    def forward(self, points, queries, feat_prev):
        """
        Args:
            points      (B x N x 3 tensor)
            query       (B x M x 3 tensor) -- note that M < N
            feat_prev   (B x N x F1 tensor)
        Outputs:
            feat        (B x M x F2 tensor)
        """

        knn_pts, idxs = knn(points, queries, self.K)
        knn_center    = queries.unsqueeze(2)
        knn_points_local = knn_center - knn_pts

        knn_feat_local = self.lift(knn_points_local)

        # shape: B x M x K x F
        if feat_prev is not None:
            knn_feat_prev = gather_feature(feat_prev, idxs)
            knn_feat_cat  = torch.cat((knn_feat_local, knn_feat_prev), dim=-1)
        else:
            knn_feat_cat  = knn_feat_local 

        knn_feat_cat = knn_feat_cat.permute(0,3,1,2) # BMKF -> BFMK        
        knn_feat_max = self.maxpool(knn_feat_cat)
        output   = self.conv(knn_feat_max).permute(0,2,3,1)

        print("knn_feat_local.shape = {}, feat_cat shape = {}, feat max shape = {}".format(
            knn_feat_local.shape, knn_feat_cat.shape, knn_feat_max.shape))

        return output.squeeze(2)

class ShellUp(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division, 
            is_batchnorm=True):
        super(ShellUp, self).__init__()
        self.is_batchnorm = is_batchnorm
        self.sconv = ShellConv(out_features, prev_features, neighbor,
            division, is_batchnorm)
        self.batchnorm = nn.BatchNorm1d(2 * out_features)
        self.linear    = nn.Sequential(
            nn.Linear(2 * out_features, out_features),
            nn.ReLU()
        )
    def forward(self, points, queries, feat_prev, feat_skip_connect):
        sconv = self.sconv(points, queries, feat_prev)
        feat_cat = torch.cat((sconv, feat_skip_connect), dim=-1)
        print("shell up feat_cat.shape = ",feat_cat.shape)
        if self.is_batchnorm == True:
            outputs = self.batchnorm(feat_cat.transpose(1,2)).transpose(1,2)
            outputs = self.linear(outputs)
        else:
            outputs = self.linear(feat_cat)
        return outputs


class ShellNet(nn.Module):
    def __init__(self, num_class, num_points, is_batchnorm=True):
        super(ShellNet, self).__init__()
        self.num_points = num_points
        self.filter_scale = 1
        self.feature_scale = 1

        filters = [64, 128, 256, 512]
        filters = [int(x / self.filter_scale) for x in filters]

        self.shellconv1 = ShellConv(filters[1],   0, 32, 4, is_batchnorm)
        self.shellconv2 = ShellConv(filters[2], filters[1], 16, 2, is_batchnorm)
        self.shellconv3 = ShellConv(filters[3], filters[2],  8, 1, is_batchnorm)        
        self.shellup3   = ShellUp(  filters[2], filters[3],  8, 1, is_batchnorm)
        self.shellup2   = ShellUp(  filters[1], filters[2], 16, 2, is_batchnorm)
        self.shellup1   = ShellConv(filters[0], filters[1], 32, 4, is_batchnorm)

        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.Dropout(0),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, num_class),
            nn.ReLU()
        )

    def forward(self, inputs):
        
        query1 = random_sample(inputs, self.num_points // 2)
        sconv1 = self.shellconv1(inputs, query1, None)
        print("sconv1.shape = ", sconv1.shape)

        
        query2 = random_sample(query1, self.num_points // 4)
        sconv2 = self.shellconv2(query1, query2, sconv1)
        print("sconv2.shape = ", sconv2.shape)

        query3 = random_sample(query2, self.num_points // 8)
        sconv3 = self.shellconv3(query2, query3, sconv2)
        print("sconv3.shape = ", sconv3.shape)        
        
        
        up3    = self.shellup3(query3, query2, sconv3, sconv2)
        print("up3.shape = ", up3.shape)

        up2    = self.shellup2(query2, query1, up3   , sconv1)
        print("up2.shape = ", up2.shape)

        up1    = self.shellup1(query1, inputs, up2)
        print("up1.shape = ", up1.shape)

        output = self.fc(up1)
        print("output.shape = ", output.shape)

        return output

B, M, K = 1, 1024, 32
# Create random Tensors to hold inputs and outputs
p = torch.randn(B, M, 3)
q = torch.randn(B, M//2, 3)
f = torch.randn(B, M, 128)
y = torch.randn(B, M//2, 128)

nn_pts, idxs = knn(p, q, 32)
nn_center    = q.unsqueeze(2)
nn_points_local = nn_center - nn_pts


model = ShellNet(2, 1024)
print(model(q).shape)