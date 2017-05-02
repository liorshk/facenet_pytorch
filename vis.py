from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from model import FaceModel
from torchvision.datasets import ImageFolder
from TripletFaceDataset import FaceDataset
import math
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

parser = argparse.ArgumentParser(description='PyTorch face recognition Example')
parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--root', type=str,
        help='path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='/media/lior/LinuxHDD/datasets/vgg_face_dataset/aligned')
parser.add_argument('--resume', type=str,
        help='model path to the resume training',
        default='/home/lior/dev/workspace/face_recognition_seminar/facenet_pytorch/logs/run-optim_adagrad-n1000000-lr0.125-wd0.0-m0.5/checkpoint_1.pth')

def visual_feature_space(features, labels, num_classes, name_dict):
    num = len(labels)

    title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':'20'}

    # draw
    palette = np.array(sns.color_palette("hls", num_classes))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:,0], features[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    # ax.axis('off')
    # ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(features[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, name_dict[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    ax.set_xlabel('Activation of the 1st neuron', **axis_font)
    ax.set_ylabel('Activation of the 2nd neuron', **axis_font)
    ax.set_title('softmax_loss + center_loss', **title_font)
    ax.set_axis_bgcolor('grey')
    f.savefig('center_loss.png')
    plt.show()
    return f, ax, sc, txts

def validation_iterator(dataLoader):
    for data, target in dataLoader:
        yield data, target

def main():
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # 1. dataset
    root = args.root
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    test_transforms = transforms.Compose([transforms.Scale(96),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    test_dataset = ImageFolder(root, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False, **kwargs)

    val_iterator = validation_iterator(test_loader)

    # 2. model
    #train_dir = FaceDataset(dir='/media/lior/LinuxHDD/datasets/MSCeleb-cleaned',n_triplets=10)

    print('construct model')
    model = FaceModel(embedding_size=128,
                      num_classes=3367,
                      pretrained=False)

    model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # extract feature
    print('extracting feature')
    embeds = []
    labels = []
    for data, target in val_iterator:
        if cuda:
            data, target = data.cuda(), target.cuda(async=True)
        data_var = Variable(data, volatile=True)
        # compute output
        output = model(data_var)

        embeds.append( output.data.cpu().numpy() )
        labels.append( target.cpu().numpy() )


    embeds = np.vstack(embeds)
    labels = np.hstack(labels)

    print('embeds shape is ', embeds.shape)
    print('labels shape is ', labels.shape)

    # prepare dict for display
    namedict = dict()
    for i in range(10):
        namedict[i]=str(i)

    visual_feature_space(embeds, labels, len(test_dataset.classes), namedict)

if __name__ == '__main__':
    main()
