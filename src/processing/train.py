import sys
import torch
import visdom
import argparse
import numpy as np
import os

from torchvision import models
from torch.autograd import Variable
from torch.utils import data
from torch.nn import DataParallel

sys.path.append('./src')
from models import get_model
from utils.ccf_loader import CCFLoader
from utils.loss import cross_entropy2d
from utils.loss import focal_loss2d, bin_clsloss
from utils.metrics import scores

weights_per_class = torch.FloatTensor([1, 1, 1, 1, 1]).cuda()


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


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=15000, power=0.9,):
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr*(1 - iter*1.0/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print("iteration %d with learning rate: %f" % (iter, lr))


def adjust_learning_rate(optimizer, init_lr, epoch, step):
    lr = init_lr * (0.1 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(model, valloader, n_class):
    import pdb;pdb.set_trace()
    losses = AverageMeter()
    model.eval()
    gts, preds = [], []
    for i, (images, labels) in enumerate(valloader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)
    if(isinstance(outputs, tuple)):
        outputs = outputs[0]

        loss = cross_entropy2d(outputs, labels)
        losses.update(loss.data[0], images.size(0))

        gt = labels.data.cpu().numpy()
        pred = outputs.data.max(1)[1].cpu().numpy()
        #pred = outputs.data[:,1:,:,:].max(1)[1].cpu().numpy() + 1

        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)
    score = scores(gts, preds, n_class=n_class)

    return losses.avg, score


def train(args):
    # Setup train DataLoader
    trainloader = CCFLoader(args.traindir, split=args.split,
                            is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = trainloader.n_classes
    TrainDataLoader = data.DataLoader(
        trainloader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # Setup validate DataLoader
    valloader = CCFLoader(args.traindir, split='val', is_transform=True, img_size=(
        args.img_rows, args.img_cols))
    VALDataLoader = data.DataLoader(
        valloader, batch_size=4, num_workers=4, shuffle=False)

    # Setup visdom for visualization
    vis = visdom.Visdom()
    assert vis.check_connection()

    loss_window = vis.line(X=np.zeros((1,)),
                           Y=np.zeros((1)),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title=args.arch+' Training Loss',
                                     legend=['Loss']))
    valacc_window = vis.line(X=np.zeros((1,)),
                             Y=np.zeros((1)),
                             opts=dict(xlabel='minibatches',
                                       ylabel='ACC',
                                       title='Val ACC',
                                       legend=['ACC']))

    # Setup model
    if(args.snapshot == None):
        model = get_model(args.arch, n_classes)
        model = DataParallel(model.cuda(args.gpu[0]), device_ids=args.gpu)
        start_epoch = 0
    else:
        model = get_model(args.arch, n_classes)
        state_dict = torch.load(args.snapshot).state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in list(state_dict.items()):
            original_key = key[7:] # remove 'moudle.'
            new_state_dict[original_key] = value
        model.load_state_dict(new_state_dict)
        model = DataParallel(model.cuda(), device_ids=[i for i in range(len(args.gpu))])
        start_epoch = int(os.path.basename(args.snapshot).split('.')[0])

    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    print(model)

    # Start training
    for epoch in range(args.n_epoch):
        adjust_learning_rate(optimizer, args.l_rate, epoch, args.step)
        if(epoch < start_epoch):
            continue
        print("Epoch [%d/%d] learning rate: %f" % (epoch+1, args.n_epoch, optimizer.param_groups[0]['lr']))
        for i, (images, labels) in enumerate(TrainDataLoader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(args.gpu[0]))
                labels = Variable(labels.cuda(args.gpu[0]))
            else:
                images = Variable(images)
                labels = Variable(labels)

            iter = len(TrainDataLoader)*epoch + i
            #poly_lr_scheduler(optimizer, args.l_rate, iter)

            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            if(isinstance(outputs, tuple)):
                loss = cross_entropy2d(outputs[0], labels, weights_per_class) + args.clsloss_weight * bin_clsloss(outputs[1], labels)
            else:
                #loss = cross_entropy2d(outputs, labels)
                loss = cross_entropy2d(outputs, labels, weights_per_class)
                #loss = focal_loss2d(outputs, labels)

            loss.backward()
            optimizer.step()

            vis.line(
                X=torch.ones((1, 1)).cpu()*iter,
                Y=torch.Tensor([loss.data]).unsqueeze(0).cpu(),
                win=loss_window,
                update='append')

        print("Epoch [%d/%d] loss: %f" % (epoch+1, args.n_epoch, loss))

        # validation
        loss, score = validate(model, VALDataLoader, n_classes)
        for i in range(n_classes):
            print(i, score['Class Acc'][i])
        vis.line(
            X=torch.ones((1, 1)).cpu()*(epoch+1),
            Y=torch.ones((1, 1)).cpu()*score['Overall Acc'],
            win=valacc_window,
            update='append')

        if(not os.path.exists("snapshot/{}".format(args.arch))):
            os.mkdir("snapshot/{}".format(args.arch))
        torch.save(model, "snapshot/{}/{}.pkl".format(args.arch, epoch+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=64,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--gpu', nargs='*', type=int, default=0)
    parser.add_argument('--traindir', nargs='?', type=str, default=None)
    parser.add_argument('--snapshot', nargs='?', type=str, default=None)
    parser.add_argument('--clsloss_weight', nargs='?',
                        type=float, default=None)
    parser.add_argument('--split', nargs='?', type=str, default='train')
    parser.add_argument('--step', nargs='?', type=int, default=30)

    args = parser.parse_args()
    train(args)
