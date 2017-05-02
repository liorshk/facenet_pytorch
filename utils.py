import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable,Function


class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)

def denormalize(tens):
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]

    img_1 = tens.clone()
    for t, m, s in zip(img_1, mean, std):
        t.mul_(s).add_(m)
    img_1 = img_1.numpy().transpose(1,2,0)
    return img_1

def display_triplet_distance(model,train_loader,name):
    f, axarr = plt.subplots(3,figsize=(10,10))
    f.tight_layout()
    l2_dist = PairwiseDistance(2)

    for batch_idx, (data_a, data_p, data_n,c1,c2) in enumerate(train_loader):

        try:
            data_a_c, data_p_c,data_n_c = data_a.cuda(), data_p.cuda(), data_n.cuda()
            data_a_v, data_p_v, data_n_v = Variable(data_a_c, volatile=True), \
                                    Variable(data_p_c, volatile=True), \
                                    Variable(data_n_c, volatile=True)

            out_a, out_p, out_n = model(data_a_v), model(data_p_v), model(data_n_v)
        except Exception as ex:
            print(ex)
            print("ERROR at: {}".format(batch_idx))
            break

        print("Distance (anchor-positive): {}".format(l2_dist.forward(out_a,out_p).data[0][0]))
        print("Distance (anchor-negative): {}".format(l2_dist.forward(out_a,out_n).data[0][0]))


        axarr[0].imshow(denormalize(data_a[0]))
        axarr[1].imshow(denormalize(data_p[0]))
        axarr[2].imshow(denormalize(data_n[0]))
        axarr[0].set_title("Distance (anchor-positive): {}".format(l2_dist.forward(out_a,out_p).data[0][0]))
        axarr[2].set_title("Distance (anchor-negative): {}".format(l2_dist.forward(out_a,out_n).data[0][0]))

        break
    f.savefig("{}.png".format(name))
    #plt.show()

from sklearn.decomposition import PCA
import numpy as np

def display_triplet_distance_test(model,test_loader,name):
    f, axarr = plt.subplots(5,2,figsize=(10,10))
    f.tight_layout()
    l2_dist = PairwiseDistance(2)

    for batch_idx, (data_a, data_n,label) in enumerate(test_loader):

        if np.all(label.cpu().numpy()):
            continue

        try:
            data_a_c, data_n_c = data_a.cuda(), data_n.cuda()
            data_a_v, data_n_v = Variable(data_a_c, volatile=True), \
                                    Variable(data_n_c, volatile=True)

            out_a, out_n = model(data_a_v), model(data_n_v)

        except Exception as ex:
            print(ex)
            print("ERROR at: {}".format(batch_idx))
            break

        for i in range(5):
            rand_index = np.random.randint(0, label.size(0)-1)
            if i%2 == 0:
                for j in range(label.size(0)):
                    # Choose label == 0
                    rand_index = np.random.randint(0, label.size(0)-1)
                    if label[rand_index] == 0:
                        break

            distance = l2_dist.forward(out_a,out_n).data[rand_index][0]
            print("Distance: {}".format(distance))
            #distance_pca = l2_dist.forward(PCA(128).fit_transform(out_a.data[i].cpu().numpy()),PCA(128).fit_transform(out_n.data[i].cpu().numpy())).data[0]
            #print("Distance(PCA): {}".format(distance_pca))

            axarr[i][0].imshow(denormalize(data_a[rand_index]))
            axarr[i][1].imshow(denormalize(data_n[rand_index]))
            plt.figtext(0.5, i/5.0+0.1,"Distance : {}, Label: {}\n".format(distance,label[rand_index]), ha='center', va='center')


        break
    plt.subplots_adjust(hspace=0.5)

    f.savefig("{}.png".format(name))
    #plt.show()
