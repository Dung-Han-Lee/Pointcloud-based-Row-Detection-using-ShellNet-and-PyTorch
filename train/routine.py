
import time
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# User define module
import config
from faci_training import unique_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, data_loader, test_loader, criterion, optimizer):

    msg = "lr = " + str(config.lr) + " batch sz = " + str(config.batch_size) + " weights = " + str(config.weights)
    identifier = config.summary_prefix + "%02d" % unique_id(msg)

    model.train()
    writer = SummaryWriter("./runs/")
    start_time = time.time()

    for epoch in range(config.num_epoch):
        avg_loss = 0.0

        for batch_num, (pointcloud, labels) in enumerate(data_loader):
            (pointcloud, labels) = map(lambda x : x.to(device), (pointcloud, labels))

            optimizer.zero_grad()

            outputs = model(pointcloud.float())
            
            loss = criterion(outputs, labels.long(), config.weights.to(device))
            loss.backward()
                
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 10 == 9:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/10))
                avg_loss = 0.0   
                
            torch.cuda.empty_cache()
            del pointcloud
            del labels
            del loss

        val_loss   = test_classify(model, test_loader, criterion, optimizer)
        train_loss = test_classify(model, data_loader, criterion, optimizer)
        print('Train Loss: {:.4f} \tVal Loss: {:.4f}'.format(train_loss, val_loss))
        
        end_time = time.time()
        print('Time: ',end_time - start_time, 's')
        
        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)

        torch.save( model.state_dict(), "../weights/"+ identifier + "/" + str(epoch)+".pth") 

def test_classify(model, test_loader, criterion, optimizer):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (pointcloud, labels) in enumerate(test_loader):
        (pointcloud, labels) = map(lambda x : x.to(device), (pointcloud, labels))
        outputs = model(pointcloud.float())
        
        loss = criterion(inputs=outputs, target=labels.long(), weight=config.weights.to(device))
        
        test_loss.extend([loss.item()]*pointcloud.size()[0])
        del pointcloud
        del labels

    model.train()
    return np.mean(test_loss)
