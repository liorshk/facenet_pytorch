from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
from model import FaceModel,FaceModelCenter
from eval_metrics import evaluate
from logger import Logger
from LFWDataset import LFWDataset
from PIL import Image
from utils import PairwiseDistance,display_triplet_distance,display_triplet_distance_test
import collections

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Face Recognition')
# Model options
parser.add_argument('--dataroot', type=str, default='/media/lior/LinuxHDD/datasets/MSCeleb-cleaned',#default='/media/lior/LinuxHDD/datasets/vgg_face_dataset/aligned'
                    help='path to dataset')
parser.add_argument('--lfw-dir', type=str, default='/media/lior/LinuxHDD/datasets/lfw-aligned-mtcnn',
                    help='path to dataset')
parser.add_argument('--lfw-pairs-path', type=str, default='lfw_pairs.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='/media/lior/LinuxHDD/pytorch_face_logs',
                    help='folder to output model checkpoints')

parser.add_argument('--resume',
                    default='/media/lior/LinuxHDD/pytorch_face_logs/run-optim_adam-lr0.001-wd0.0-embeddings512-center0.5-MSCeleb/checkpoint_11.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
# parser.add_argument('--embedding-size', type=int, default=256, metavar='ES',
#                     help='Dimensionality of the embedding')

parser.add_argument('--center_loss_weight', type=float, default=0.5, help='weight for center loss')
parser.add_argument('--alpha', type=float, default=0.5, help='learning rate of the centers')
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adam', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True

LOG_DIR = args.log_dir + '/run-optim_{}-lr{}-wd{}-embeddings{}-center{}-MSCeleb'.format(args.optimizer, args.lr, args.wd,args.embedding_size,args.center_loss_weight)

# create logger
logger = Logger(LOG_DIR)

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
l2_dist = PairwiseDistance(2)

transform = transforms.Compose([
                         transforms.Scale(96),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                               std = [ 0.5, 0.5, 0.5 ])
                     ])

train_dir = ImageFolder(args.dataroot,transform=transform)
train_loader = torch.utils.data.DataLoader(train_dir,
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    LFWDataset(dir=args.lfw_dir,pairs_path=args.lfw_pairs_path,
                     transform=transform),
    batch_size=args.batch_size, shuffle=False, **kwargs)



def main():
    test_display_triplet_distance= True
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Classes:\n{}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
        else:
            checkpoint = None
            print('=> no checkpoint found at {}'.format(args.resume))

    model = FaceModelCenter(embedding_size=args.embedding_size,num_classes=len(train_dir.classes)
                            ,checkpoint=checkpoint)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    start = args.start_epoch
    end = start + args.epochs

    for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch)
        if test_display_triplet_distance:
            display_triplet_distance_test(model,test_loader,LOG_DIR+"/test_{}".format(epoch))




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))

    top1 = AverageMeter()

    for batch_idx, (data, label) in pbar:

        data_v = Variable(data.cuda())
        target_var = Variable(label)

        # compute output
        prediction = model.forward_classifier(data_v)

        center_loss, model.centers = model.get_center_loss(target_var, args.alpha)

        criterion = nn.CrossEntropyLoss()

        cross_entropy_loss = criterion(prediction.cuda(),target_var.cuda())

        loss = args.center_loss_weight*center_loss + cross_entropy_loss

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)

        # log loss value
        # logger.log_value('cross_entropy_loss', cross_entropy_loss.data[0]).step()
        logger.log_value('total_loss', loss.data[0]).step()

        prec = accuracy(prediction.data, label.cuda(), topk=(1,))
        top1.update(prec[0], data_v.size(0))

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'
                'Train Prec@1 {:.2f} ({:.2f})'.format(
                    epoch, batch_idx * len(data_v), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0],float(top1.val[0]), float(top1.avg[0])))



    logger.log_value('Train Prec@1 ',float(top1.avg[0]))

    # do checkpointing
    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'centers': model.centers},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist[0] for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far = evaluate(distances,labels)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    logger.log_value('Test Accuracy', np.mean(accuracy))

    plot_roc(fpr,tpr,figure_name="roc_test_epoch_{}.png".format(epoch))

def plot_roc(fpr,tpr,figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(LOG_DIR,figure_name), dpi=fig.dpi)


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd, betas=(args.beta1, 0.999))
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

if __name__ == '__main__':
    main()