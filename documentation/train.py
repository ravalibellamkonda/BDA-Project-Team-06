'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse

import numpy as np

#from models import *
from utils import progress_bar
from efficientnet_pytorch import EfficientNet


title = 'possmilevsnegsmile'
data_dir = '/home/srinivas/classificon/data/'+title
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch >250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1
    return lr / (10**optim_factor)#math.pow(10, (optim_factor))
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1., 1., 1.)),
    
])

transform_test = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1., 1., 1.)),
    
])

trainset = datasets.ImageFolder(data_dir+'/train',  transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.ImageFolder(data_dir+'/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
#print('==> Building model..')
# net = VGG('VGG19')
#net = ResNet18()
#net = PreActResNet18()
# net = GoogLeNet()
#net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
net = EfficientNet.from_pretrained('efficientnet-b7',num_classes=2)

model = 'EfficientNetB7'

Train_acc = []
Test_acc = []
Train_loss = []
Test_loss = []

print(net)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
start_epoch=0
'''
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/rose_ckpt_Res18_new_17March_90Epoch.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']+1
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d best_acc: %f' %(epoch,best_acc))
    optimizer  = optim.SGD(net.parameters(), lr = lr_schedule(args.lr, epoch), momentum = 0.9, weight_decay = 5e-4)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    Train_loss.append(train_loss/(batch_idx+1))
    Train_acc.append(100.*correct/total)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        Test_loss.append(test_loss/(batch_idx+1))
        Test_acc.append(100.*correct/total)


    # Save checkpoint.
    acc = 100.*correct/total
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            }
        
    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/best_'+title+'_'+model+'.pth') 
        best_acc = acc
    torch.save(state, './checkpoint/'+title+'_'+model+'.pth') 


for epoch in range(start_epoch, 350):
    print("data: ",title, "model :", model)
    train(epoch)
    #if (epoch+1)%10 == 0:
    test(epoch)
    if not os.path.isdir('./checkpoint/'+title+'_'+model):
        os.mkdir('./checkpoint/'+title+'_'+model)

    np.save('./checkpoint/'+title+'_'+model+'/train_loss.npy',Train_loss)
    np.save('./checkpoint/'+title+'_'+model+'/test_loss.npy',Test_loss)
    np.save('./checkpoint/'+title+'_'+model+'/train_acc.npy',Train_acc)
    np.save('./checkpoint/'+title+'_'+model+'/test_acc.npy',Test_acc)
