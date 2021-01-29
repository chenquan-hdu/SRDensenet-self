from __future__ import print_function
import argparse, os
import torch
import random
import math
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import get_training_set
import torch.optim as optim
from model import Net

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # use the chosen gpu

def total_gradient(parameters):
    """Computes a gradient clipping coefficient based on gradient norm."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = 0
    for p in parameters:
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = totalnorm ** (1./2)
    return totalnorm

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(epoch):
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        out = model(input)
        loss = criterion(out, target)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),opt.clip)
        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), epoch_loss / len(training_data_loader)))
            print("total gradient", total_gradient(model.parameters()))

    print("===> Epoch [{}] Complete: Avg.Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))



#def test():
    #avg_psnr = 0
    #with torch.no_grad():
        #for iteration, batch in enumerate(testing_data_loader,1):
            #input, target = Variable(batch[0]), Variable(batch[1])

            #if opt.cuda:
                #input = input.cuda()
                #target = target.cuda()

            #mse = criterion(model(input), target)
            #cacl_psnr = 10 * math.log10(1/mse.item())
            #avg_psnr = cacl_psnr

    #avg_psnr = avg_psnr /len(testing_data_loader)
    #print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))

def save_checkpoint(epoch):
    model_folder = "checkpoint/"
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


#Training settings
parser = argparse.ArgumentParser(description="PyTorch DenseNet")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--testbatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=60, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=30, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="D:\project\SRDenseNet-self\checkpoint\model_epoch_11.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
opt = parser.parse_args()
print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True

print("===> Loading datasets")
train_set = get_training_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True, pin_memory=True)
#test_set = get_test_set()
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchSize)

print("===> Building model")
model = Net(16,8)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model.cuda()

criterion = nn.MSELoss()

print("===> Setting GPU")
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

# optionally copy weights from a checkpoint
if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
#optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

print("===> Training")
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(epoch)
    #test()
    save_checkpoint(epoch)




