from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from numpy.random import beta
from load_dataset import load_NTUdataset
from os.path import join
import h5py
##########################################################################################
class SFNNet(nn.Module):
    def __init__(self,indim,hinddim,laysers,dp=.0):
        super(SFNNet, self).__init__() 
        self.Enlstm = nn.LSTM(indim, hinddim, laysers,batch_first=True,dropout=dp)
        self.Delstm = nn.LSTM(hinddim, indim, laysers, batch_first=True,dropout=dp)
    
    def forward(self, inputs):
        encoder,hidden = self.Enlstm(inputs)
        out,_ = self.Delstm(encoder)
        return out
#####################
class HARNet(nn.Module):
    def __init__(self,indim,outdim,layers,Output_classnum,dp=.0):
        super(HARNet, self).__init__()
        self.lstm = nn.LSTM(indim, outdim, layers, batch_first=True,dropout=dp)
        self.linear = nn.Sequential(nn.Linear(outdim, Output_classnum),nn.ELU()) 
    def forward(self, inputs):
        features,_ = self.lstm(inputs)
        out = self.linear(features[:,-1,:])
        return out
##########################################################################################

def load_data(datadir):
    f = h5py.File(datadir,'r')
    train_data = f['train_data'][:]
    test_data = f['test_data'][:]
    train_labels = f['train_label'][:]
    test_labels = f['test_label'][:]             
    return train_data, train_labels, test_data, test_labels
##########################################################################################


def train(args,model_HAR, model_SFN, device, train_dataset, optimizer, epoch):
    model_HAR.train()
    model_SFN.train()
    ##################################################################
    beta_param = 1.0  - min(0.1*(int(epoch//(args.LR_STEP//10))),0.9)
    train_loader =  torch.utils.data.DataLoader(dataset=train_dataset, 
                          num_workers=2,batch_size= args.batch_size, shuffle=True)

    for batch_idx, (data_a, target_a) in enumerate(train_loader):
        ######################################################################
        data_a, target_a = data_a.to(device), target_a.to(device)
                
        B1,B2,B3 = beta(beta_param,beta_param),beta(beta_param,beta_param),beta(beta_param,beta_param)

        Indx_b = torch.randperm(len(target_a)).to(device)

        Indx_c = torch.randperm(len(target_a)).to(device)
        Indx_d = torch.randperm(len(target_a)).to(device)
        
        input_org       = B1*data_a         + (1-B1)*data_a[Indx_b]
        input_SFN_befor = B2*data_a[Indx_c] + (1-B2)*data_a[Indx_d] 
              
        input_SFN_after = model_SFN(input_SFN_befor)

        input_fusion = B3*input_org + (1-B3)*input_SFN_after
        ##############
        optimizer.zero_grad()
        output = model_HAR(input_fusion)

        loss_a = F.cross_entropy(output, target_a)        
        loss_b = F.cross_entropy(output, target_a[Indx_b])
        loss_c = F.cross_entropy(output, target_a[Indx_c])
        loss_d = F.cross_entropy(output, target_a[Indx_d])

        loss_AE  = F.mse_loss(input_SFN_after,input_SFN_befor)/1.414
        loss = B3*(B1*loss_a + (1-B1)*loss_b) + (1-B3)*(B2*loss_c + (1-B2)*loss_d + args.lama*loss_AE)

        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_a), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args,model_HAR, model_SFN, device, test_dsets):
    model_HAR.eval()
    model_SFN.eval()
    test_loss = 0
    correct_05,correct_1,correct_0 = 0,0,0
    test_loader =  torch.utils.data.DataLoader(dataset=test_dsets, 
                          num_workers=2,batch_size= args.testbatch_size, shuffle=False)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_org = model_HAR(data)
            output_SFN = model_HAR(model_SFN(data))

            #SFN_0.5
            output = 0.5*output_org + 0.5*output_SFN
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct_05 += pred.eq(target.view_as(pred)).sum().item()

            #SFN_1
            test_loss += F.cross_entropy(output_SFN, target, size_average=False).item() 
            pred = output_SFN.max(1, keepdim=True)[1] 
            correct_1 += pred.eq(target.view_as(pred)).sum().item()

            #SFN_0
            test_loss += F.cross_entropy(output_org, target, size_average=False).item() 
            pred = output_org.max(1, keepdim=True)[1] 
            correct_0 += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: SFN_05: {}/{} {:.2f}%, SFN_0: {:.2f}%, SFN_1: {:.2f}%)\n'.format(
        correct_05, len(test_loader.dataset), 100. * correct_05/ len(test_loader.dataset), 100. * correct_0 / len(test_loader.dataset)
        , 100. * correct_1 / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SFN Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--testbatch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='NTU_CV.hdf5', metavar='DS',
                        help='dataset (default: NTU_CV.hdf5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--LR_STEP', type=int, default=800, metavar='LS',
                        help='LR_STEP (default: 800)')
    parser.add_argument('--lama', type=float, default=0.1, metavar='LM',
                        help='lr-step (default: 0.1)')
    parser.add_argument('--Scale', type=float, default=0.5, metavar='S',
                        help='AE Scale (default: 0.5)')
    parser.add_argument('--Output_dim', type=int, default=100, metavar='OD',
                        help='Output_dim (default: 100)')
    parser.add_argument('--Output_classnum', type=int, default=60, metavar='CN',
                        help='Output_classnum (default: 60)')
    parser.add_argument('--SFN_layers', type=int, default=1, metavar='SLs',
                        help='SFN_layers (default: 1)')
    parser.add_argument('--HAR_layers', type=int, default=3, metavar='HLs',
                        help='HAR_layers (default: 3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_devices', type=int, default=0,
                        help='useing cuda devices id')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    ####################################################################################

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(args.cuda_devices)
    ####################################################################################
    data_files = join('./data',args.dataset)     
    train_data, train_labels, test_data, test_labels = load_data(data_files) 
    train_dsets = torch.utils.data.TensorDataset(
                  torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long()
                            )    
    test_dsets  = torch.utils.data.TensorDataset(
                  torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long()
                            )
    ####################################################################################
    input_dim = train_data.shape[-1]
    hind_dim = int(args.Scale*input_dim)

    print('hind_dim {} , lama {} , epochs {} , SFN_layers {} and  HAR_layers {}'.format(
           hind_dim,args.lama,args.epochs,args.SFN_layers,args.HAR_layers))
    ####################################################################################

    model_SFN = SFNNet(input_dim,hind_dim,args.SFN_layers).cuda()
    model_HAR = HARNet(input_dim,args.Output_dim,args.HAR_layers,args.Output_classnum).cuda()

    optimizer = optim.Adam(list(model_SFN.parameters())+list(model_HAR.parameters()), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.LR_STEP, 0.1)

    for epoch in range(1, args.epochs + 1):
        lr_scheduler.step()
        train(args, model_HAR, model_SFN,device, train_dsets, optimizer, epoch)
        if epoch%10==0:
            test(args, model_HAR, model_SFN,device, test_dsets)


if __name__ == '__main__':
    main()