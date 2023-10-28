#dataset:https://www.crcv.ucf.edu/data/UCF101.php

from sys import api_version
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from dataloaders.dataset import VideoDataset
from network import C3D_model


NUM_CLASS = 44
CLIP_LEN = 40

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 30  
resume_epoch = 0  
useTest = True 
nTestInterval = 2 
snapshot = 30 
lr = 1e-5 


epochs_list = []
epochs_list_test = []
epochs_list_test.append(0)
for i in range(1, nEpochs + 1):
        epochs_list.append(i)


dataset = 'Auslan'

if dataset == 'Auslan':
    num_classes = NUM_CLASS
else:
    print('We only implemented Auslan datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset




def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    train_loss_records = []
    train_accuracy_records = []
    vali_loss_records = []
    vali_accuracy_records = []
    test_loss_records = []
    test_accuracy_records = []
    test_loss_records.append(5)
    test_accuracy_records.append(0)

    model = C3D_model.Linear3DModel(num_classes=num_classes, pretrained=False)
    train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                    {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    # retrain from beginning
    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
        model.to(device)
        criterion.to(device)

    # retraining 
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        criterion.to(device)
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))


    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=CLIP_LEN, preprocess=True), batch_size=6, shuffle=True, num_workers=0)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=CLIP_LEN, preprocess=True), batch_size=6, num_workers=0)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=CLIP_LEN, preprocess=True), batch_size=6, num_workers=0)

    # train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=CLIP_LEN), batch_size=6, shuffle=True, num_workers=0)
    # val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=CLIP_LEN), batch_size=6, num_workers=0)
    # test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=CLIP_LEN), batch_size=6, num_workers=0)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
                train_loss_records.append(epoch_loss)
                train_accuracy_records.append(epoch_acc.item())
                

            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
                vali_loss_records.append(epoch_loss)
                vali_accuracy_records.append(epoch_acc.item())



            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            


        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

            test_loss_records.append(epoch_loss)
            test_accuracy_records.append(epoch_acc.item())
            epochs_list_test.append(epoch+1)

    writer.close()


    train_loss_records = np.array(train_loss_records)
    train_accuracy_records = np.array(train_accuracy_records)
    vali_loss_records = np.array(vali_loss_records)
    vali_accuracy_records = np.array(vali_accuracy_records)

    plt.figure(1)
    plt.plot(epochs_list, train_loss_records)
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.title('epoch num VS training loss')
    plt.savefig(f'{save_dir}\\epoch num VS training loss.png')

    plt.figure(2)
    plt.plot(epochs_list, train_accuracy_records)
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.title('epoch num VS training accuracy')
    plt.savefig(f'{save_dir}\\epoch num VS training accuracy.png')

    plt.figure(3)
    plt.plot(epochs_list, vali_loss_records)
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.title('epoch num VS validation loss')
    plt.savefig(f'{save_dir}\\epoch num VS validation loss.png')
    
    plt.figure(4)
    plt.plot(epochs_list, vali_accuracy_records)
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.title('epoch num VS validation accuracy')
    plt.savefig(f'{save_dir}\\epoch num VS validation accuracy.png')

    plt.figure(5)
    plt.plot(epochs_list_test, test_loss_records)
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.title('epoch num VS test loss')
    plt.savefig(f'{save_dir}\\epoch num VS test loss.png')

    plt.figure(6)
    plt.plot(epochs_list_test, test_accuracy_records)
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.title('epoch num VS test accuracy')
    plt.savefig(f'{save_dir}\\epoch num VS test accuracy.png')
    
    # OVERALL 
    plt.figure(7)
    plt.plot(epochs_list, train_accuracy_records, color='blue')
    plt.plot(epochs_list, vali_accuracy_records, color='red')
    plt.plot(epochs_list_test, test_accuracy_records, color='green')
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.legend(labels=['train', 'validation','test'])
    plt.title('epoch num VS accuracy')
    plt.savefig(f'{save_dir}\\epoch num VS accuracy.png')

    plt.figure(8)
    plt.plot(epochs_list, train_loss_records, color='blue')
    plt.plot(epochs_list, vali_loss_records, color='red')
    plt.plot(epochs_list_test, test_loss_records, color='green')
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.legend(labels=['train', 'validation','test'])
    plt.title('epoch num VS loss')
    plt.savefig(f'{save_dir}\\epoch num VS loss.png')

    plt.show()



if __name__ == "__main__":
    train_model()