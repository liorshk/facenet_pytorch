from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
from model import FaceModel
from eval_metrics import evaluate
from logger import Logger
from FaceDataset import FaceDataset
from LFWDataset import LFWDataset
from PIL import Image
from utils import PairwiseDistance,display_triplet_distance
import collections

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Face Recognition')
# Model options
parser.add_argument('--dataroot', type=str, default='/media/lior/LinuxHDD/datasets/vgg_face_dataset/cropped',
                    help='path to dataset')
parser.add_argument('--lfw-dir', type=str, default='/media/lior/LinuxHDD/datasets/lfw-deepfunneled-aligned-jpg',
                    help='path to dataset')
parser.add_argument('--lfw-pairs-path', type=str, default='lfw_pairs.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='./logs',
                    help='folder to output model checkpoints')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--n-triplets', type=int, default=100000, metavar='N',
                    help='how many triplets will generate from the dataset')

parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')

parser.add_argument('--lr', type=float, default=0.125, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adagrad', type=str,
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

LOG_DIR = args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}'\
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin)

# create logger
logger = Logger(LOG_DIR)


class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
l2_dist = PairwiseDistance(2)

transform = transforms.Compose([
                         Scale(96),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                               std = [ 0.5, 0.5, 0.5 ])
                     ])

train_dir = FaceDataset(dir=args.dataroot,n_triplets=args.n_triplets,transform=transform)
train_loader = torch.utils.data.DataLoader(train_dir,
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    LFWDataset(dir=args.lfw_dir,pairs_path=args.lfw_pairs_path,
                     transform=transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)



def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    test_display_triplet_distance = False

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    # instantiate model and initialize weights
    model = FaceModel(embedding_size=args.embedding_size,
                      num_classes=len(train_dir.classes),
                      pretrained=False)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs

    if test_display_triplet_distance:
        display_triplet_distance(model,train_loader)
    else:
        for epoch in range(start, end):
            train(train_loader, model, optimizer, epoch)
            test(test_loader, model, epoch)


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))
    labels, distances = [], []

    for batch_idx, (data_a, data_p, data_n,label_p,label_n) in pbar:

        data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), \
                                 Variable(data_n)

        # compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
        triplet_loss = TripletMarginLoss(args.margin).forward(out_a, out_p, out_n)

        cls_a = model.forward_classifier(data_a)
        cls_p = model.forward_classifier(data_p)
        cls_n = model.forward_classifier(data_n)

        criterion = nn.CrossEntropyLoss()
        predicted_labels = torch.cat([cls_a,cls_p,cls_n])
        true_labels = torch.cat([Variable(label_p.cuda()),Variable(label_p.cuda()),Variable(label_n.cuda())])

        cross_entropy_loss = criterion(predicted_labels.cuda(),true_labels.cuda())

        # compute gradient and update weights
        optimizer.zero_grad()
        triplet_loss.backward()
        cross_entropy_loss.backward()
        optimizer.step()

        loss = triplet_loss.data[0]+cross_entropy_loss.data[0]

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)

        # log loss value
        logger.log_value('triplet_loss', triplet_loss.data[0]).step()
        logger.log_value('cross_entropy_loss', cross_entropy_loss.data[0]).step()
        logger.log_value('total_loss', loss).step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss))


        dists = l2_dist.forward(out_a,out_n) #torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(np.zeros(len(distances)))


        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(np.ones(len(distances)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist[0] for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far = evaluate(distances,labels)
    print('\33[91mTrain set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    logger.log_value('Train Accuracy', np.mean(accuracy))

    plot_roc(fpr,tpr,figure_name="roc_train_epoch_{}.png".format(epoch))

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
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
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

if __name__ == '__main__':
    main()