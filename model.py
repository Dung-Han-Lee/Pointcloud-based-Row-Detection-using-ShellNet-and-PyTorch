import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb

# TODO 
# experiment with small example to test 
# batchnorm3d
# maxpool2d
# conv2d

def knn_one_batch(points, queries, K):
    """
    Args:
        points: L x N x 3 tensor
        query : L x M x 3 tensor, M < N
        K     : num of neighbors (constant)
    Outputs:
        nn one batch : L x M x K x 3 tensor (sorted K nearest neighbor)
    """
    value = None
    indices = None
    for i in range(points.shape[0]):
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

def knn(points, queries, K):
    """
    Args:
        points: B x L x N x 3 tensor
        query : B x L x M x 3 tensor, M < N
        K     : num of neighbors (constant)
    Outputs:
        knn   : B x L x M x K x 3 tensor (sorted K nearest neighbor)
    """
    num_batch = points.shape[0]
    value = None
    indices = None
    for i in range(num_batch):
        point_batch = points[i]
        query_batch = queries[i]
        knn_batch, idx_batch = knn_one_batch(point_batch, query_batch, K)
        knn_batch = knn_batch.unsqueeze(0)
        idx_batch = idx_batch.unsqueeze(0)
        value   = knn_batch if value is None else torch.cat((value, knn_batch))
        indices = idx_batch if indices is None else torch.cat((indices, idx_batch))
    return value, indices

def gather_feature(feat, indices):
    """
    This function uses the indices generated in knn to gather associated
    feature in the previous layer. This implementation is an inefficient
    workaround.

    Args:
        input  : B x L x N x K x F tensor
        indices: B x L x N x K tensor
    Output:

    """

    res = None
    for B, point2d in enumerate(test_input):
        one_batch = None
        for L, point1d in enumerate(point2d):
            pt = point1d[indices[B][L]].unsqueeze(0)
            one_batch = pt if one_batch is None else torch.cat((one_batch, pt))
        one_batch = one_batch.unsqueeze(0)
        res = one_batch if res is None else torch.cat((res, one_batch))
    return res


class Lift(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(Lift, self).__init__()    

        self.is_batchnorm = is_batchnorm

        self.bn1 = nn.BatchNorm3d(3)
        self.linear1 = nn.Linear(in_size, out_size // 2)
        self.bn2 = nn.BatchNorm3d(out_size // 2)
        self.linear2 = nn.Linear(out_size // 2, out_size)


    def forward(self, inputs):

        if self.is_batchnorm == True:
            outputs = self.bn1(inputs.transpose(1,4)).transpose(1,4)
            outputs = F.relu(self.linear1(outputs))
            outputs = self.bn2(outputs.transpose(1,4)).transpose(1,4)
            outputs = F.relu(self.linear2(outputs))
            return outputs

        else:
            outputs = F.relu(self.linear1(inputs))
            outputs = F.relu(self.linear2(outputs))
            return outputs


class ShellConv(nn.Module):
    def __init__(self, num_feature, num_neighbor, num_division, is_batchnorm=True):
        super(Lift, self).__init__() 

        s = int(num_neighbor/num_division)
        self.K = num_neighbor
        self.C = num_feature
        self.lift = Lift(3, 64)
        self.maxpool = nn.Maxpool3d((1, s, 1), stride = (1,s,1))

    def forward(self, points, queries, feat_prev):
        
        nn_pts, idxs = knn(points, queries, self.K)
        nn_center    = queries.unsqueeze(3)
        nn_points_local = nn_center - nn_pts

        nn_feat_local = self.lift(nn_points_local)

        if feat_prev is not None:
            feat_prev = gather_feature(feat_prev, idxs)
            feat_cat  = torch.cat((nn_feat_local, feat_prev), dim = -1)
        else
            feat_cat  = nn_feat_local

        feat_max = self.maxpool(feat_cat)
        return feat_max


B , N, M = 1, 32, 1024
# Create random Tensors to hold inputs and outputs
p = torch.randn(B, N, M, 3)
q = torch.randn(B, N, M//2, 3)

# Construct our model by instantiating the class defined above

test_input = torch.zeros(2,3,4)
test_input[0] = torch.Tensor([[[0,2,3,3],
                               [0,0,0,3],
                               [0,0,0,0]]])
test_input[1] = torch.Tensor([[[3,5,6,6],
                               [3,0,0,6],
                               [3,0,0,6]]])
queries = torch.zeros(2,3,1)
queries[0] = torch.Tensor([[1], [0], [0]])
queries[1] = torch.Tensor([[3], [3], [3]])

test_input = test_input.permute(0,2,1).unsqueeze(0) 
queries = queries.permute(0,2,1).unsqueeze(0)

test_input = p
value, indices = knn(test_input, q, 32)

#print(value.shape)
#print(indices.shape)
#res = gather_feature(test_input, indices)
#print(torch.all(torch.eq(value, res)))
#print(knn(test_input, queries, 2).shape)


model = Lift(3, 64, True)
#print(model(test_input, queries).shape)

nn_pts, indices = knn(p, q, 32)
nn_center = q.unsqueeze(3)
nn_pts_local = nn_center - nn_pts

print(model(nn_pts_local).shape)
y = torch.randn(1,N, M//2, 32, 64)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for t in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(nn_pts_local)
    # Compute and print loss
    loss = criterion(y_pred, y)
    #if t % 100 == 99:
    print(t, loss.item())
    # Zero gradients, perform a backward pass, and update the weights.
    #optimizer.zero_grad()
    loss.backward()

    #for param in model.parameters():
    #    if param.grad is not None:
    #        print(param.grad)

    optimizer.step()
