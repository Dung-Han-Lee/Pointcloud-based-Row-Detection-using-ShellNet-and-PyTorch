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
    res = None
    num_batch = points.shape[0]
    for i in range(num_batch):
        point = points[i]
        query = queries[i]
        dist  = torch.cdist(point, query)
        idxs  = dist.topk(K, dim=0, largest=False, sorted=True).indices
        nn    = point[idxs].transpose(0,1).unsqueeze(0)
        res   = nn if res is None else torch.cat((res, nn))
    return res 

def knn(points, queries, K):
    """
    Args:
        points: B x L x 3 x N tensor
        query : B x L x 3 x M tensor, M < N
        K     : num of neighbors (constant)
    Outputs:
        knn   : B x L x 3 x KM tensor (sorted K nearest neighbor)
    """
    num_batch = points.shape[0]
    res = None
    for i in range(num_batch):
        point_batch = points[i]
        query_batch = queries[i]
        knn_batch = knn_one_batch(point_batch, query_batch, K)
        knn_batch = knn_batch.unsqueeze(0)
        res = knn_batch if res is None else torch.cat(res, knn_batch)
    return res


class Lift(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(Lift, self).__init__()    

        self.is_batchnorm = is_batchnorm
        self.linear1 = nn.Linear(in_size, out_size // 2)
        self.bn1 = nn.BatchNorm3d(out_size // 2)
        self.linear2 = nn.Linear(out_size // 2, out_size)
        self.bn2 = nn.BatchNorm3d(out_size)

    def forward(self, inputs):

        if self.is_batchnorm == True:
            outputs = self.linear1(inputs)
            outputs = F.relu(self.bn1(outputs.transpose(1,4))).transpose(1,4)
            outputs = self.linear2(outputs)
            outputs = F.relu(self.bn2(outputs.transpose(1,4))).transpose(1,4)
            return outputs

        else:
            outputs = F.relu(self.linear1(inputs))
            outputs = F.relu(self.linear2(outputs))
            return outputs

class ShellConv(nn.Module):
    def __init__(self, num_feature, num_neighbor, num_shell, is_batchnorm=True):
        super(Lift, self).__init__() 

        self.K = num_neighbor
        self.D = num_shell
        self.C = num_feature
        self.lift = Lift(3, 64)

    def forward(self, points, queries, feat_prev):
        nn_pts       = knn(points, queries, self.K)
        nn_center    = queries.unsqueeze(3)
        nn_points_local = nn_center - nn_pts
        nn_feat_local = self.lift(nn_points_local)


N, M = 32, 1024
# Create random Tensors to hold inputs and outputs
p = torch.randn(1, N, M, 3)
q = torch.randn(1, N, M//2, 3)

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

model = Lift(3, 64, True)
#print(model(test_input, queries).shape)

nn_pts = knn(p, q, 32)
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
