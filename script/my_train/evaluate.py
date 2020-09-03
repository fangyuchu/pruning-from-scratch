import torch
import time
import os
from datetime import datetime
from script.my_train import data_loader, measure_flops, config as conf
import torch.nn as nn
from script.my_train import storage
from copy import deepcopy
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

# def validate(val_loader, model, criterion):
def validate(val_loader, model,max_data_to_test=99999999,device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        model.eval()
        end = time.time()
        s_n=0
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            s_n+=input.shape[0]
            # compute output
            output = model(input)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            #losses.update(loss.data.item(), input.size(0))

            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if s_n>=max_data_to_test:
                break
        print('{} Acc@1 {top1.avg:} Acc@5 {top5.avg:}'
              .format(datetime.now(),top1=top1, top5=top5))

        return top1.avg, top5.avg



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
        res.append(correct_k.mul_(1 / batch_size))                  #each item is one k_accuracy
    return res


def evaluate_net(  net,
                   data_loader,
                   save_net,
                   net_name=None,
                   exp_name='',
                   checkpoint_path=None,
                   sample_num=0,
                   target_accuracy=1,
                   dataset_name='cifar10',
                   max_data_to_test=99999999,
                   top_acc=1,
                   device=None,
                   ):
    '''
    :param net: net of NN
    :param data_loader: data loader of test set
    :param save_net: Boolean. Whether or not to save the net.
    :param net_name: name of the network. eg:vgg16_bn
    :param checkpoint_path:
    :param highest_accuracy_path:
    :param sample_num_path:
    :param sample_num: sample num of the current trained net
    :param target_accuracy: save the net if its accuracy surpasses the target_accuracy
    :param max_data_to_test: use at most max_data_to_test images to evaluate the net
    :param top_acc: top 1 or top 5 accuracy
    '''
    if isinstance(net,nn.DataParallel):
        net=deepcopy(net.module)
    net.eval()
    if save_net:
        flop_num = measure_flops.measure_model(net=net, dataset_name=dataset_name, print_flop=False)
        if checkpoint_path is None :
            raise AttributeError('please input checkpoint path')

        lists=os.listdir(checkpoint_path)
        file_new = checkpoint_path
        if len(lists) > 0:
            lists.sort(key=lambda fn: os.path.getmtime(checkpoint_path + "/" + fn))  # 按时间排序
            file_new = os.path.join(checkpoint_path, lists[-1])  # 获取最新的文件保存到file_new

        if os.path.isfile(file_new):
            checkpoint=torch.load(file_new)
            highest_accuracy = checkpoint['highest_accuracy']
            flop_num_old=checkpoint['flop_num']
            if flop_num!=flop_num_old:
                highest_accuracy=0
        else:
            highest_accuracy=0

    print("{} Start Evaluation".format(datetime.now()))
    print("{} sample num = {}".format(datetime.now(), sample_num))

    top1_accuracy,top5_accuracy=validate(data_loader,net,max_data_to_test,device)
    if top_acc==1:
        accuracy=top1_accuracy
    elif top_acc==5:
        accuracy=top5_accuracy

    if save_net and (accuracy > highest_accuracy or accuracy>target_accuracy):
        # save net
        print("{} Saving net...".format(datetime.now()))
        checkpoint={'highest_accuracy':accuracy,
                    'sample_num':sample_num,
                    'flop_num':flop_num,
                    'exp_name':exp_name}
        checkpoint.update(storage.get_net_information(net,dataset_name,net_name))
        torch.save(checkpoint,'%s/flop=%d,accuracy=%.5f.tar' % (checkpoint_path, flop_num,accuracy))
        print("{} net saved at sample num = {}".format(datetime.now(), sample_num))

    return accuracy

#


    






if __name__ == "__main__":
    print()







