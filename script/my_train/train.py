import os,sys
sys.path.append('../')
sys.path.append('../../')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datetime import datetime
import math
import matplotlib.pyplot as plt
from script.my_train import data_loader, measure_flops, evaluate, config as conf
from math import ceil
from script.my_train import storage
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def exponential_decay_learning_rate(optimizer, sample_num, num_train,learning_rate_decay_epoch,learning_rate_decay_factor,batch_size):
    """Sets the learning rate to the initial LR decayed by learning_rate_decay_factor every decay_steps"""
    current_epoch = ceil(sample_num / num_train)
    if learning_rate_decay_factor > 1:
        learning_rate_decay_factor = 1 / learning_rate_decay_factor  # to prevent the mistake
    if current_epoch in learning_rate_decay_epoch and sample_num - (num_train * (current_epoch - 1)) <= batch_size:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * learning_rate_decay_factor
            lr = param_group['lr']
        print('{} learning rate at present is {}'.format(datetime.now(), lr))


def set_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_modules_no_grad(net, module_no_grad):
    '''
    :param net:
    :param module_no_grad: module name which doesn't need to be trained
    :return: a dict containing parameter names that don't need grad
    '''
    requires_grad_dict=dict()
    for name, _ in net.named_parameters():
        requires_grad_dict[name] = True
        if type(module_no_grad) is list:
            for mod_name in module_no_grad:
                if mod_name in name:
                    requires_grad_dict[name]=False
                    print(name)
        else:
            if module_no_grad in name:
                requires_grad_dict[name]=False
    requires_grad_dict['default']=True
    return requires_grad_dict

def look_up_hyperparameter(parameter_dict,parameter_name, hyperparameter_type):
    '''
    find the specified hyper-parameter for each parameter
    :param parameter_dict: a dict containing(key:partial name of the parameter, value: hyperparameter)
    :param parameter_name: name of the parameter in model
    :param hyperparameter_type: type of the hyperparameter. e.g 'learning rate', 'momentum'
    :return:
    '''
    if type(parameter_dict) is not dict:
        return parameter_dict
    if 'default' not in parameter_dict.keys():
        raise Exception('Default '+hyperparameter_type+' is not given.')
    for key in parameter_dict.keys():
        if key in parameter_name:
            return parameter_dict[key]
    return parameter_dict['default']

def prepare_optimizer(
        net,
        optimizer,
        momentum=conf.momentum,
        learning_rate=conf.learning_rate,
        weight_decay=conf.weight_decay,
        requires_grad=True,
        # **kwargs
):
    param_list=[]
    for name, value in net.named_parameters():
        value.requires_grad = look_up_hyperparameter(requires_grad,name,'requires_grad')
        if value.requires_grad is True:
            m=look_up_hyperparameter(momentum,name,'momentum')
            lr=look_up_hyperparameter(learning_rate,name,'learning rate')
            wd=look_up_hyperparameter(weight_decay,name,'weight decay')
            param_list+=[{'params':value,
                          'lr':lr,
                          'initial_lr':lr,
                          'weight_decay':wd,
                          'momentum':m}]

    optimizer=optimizer(param_list,lr=look_up_hyperparameter(learning_rate,'default','lr'))



    # if optimizer is optim.Adam:
    #     # optimizer = optimizer([{'params':filter(lambda p: p.requires_grad, net.parameters()),'initial_lr':learning_rate}],
    #     #                       lr=learning_rate,
    #     #                       weight_decay=weight_decay,**kwargs)
    # elif optimizer is optim.SGD:
    #     optimizer=optimizer([{'params':filter(lambda p: p.requires_grad, net.parameters()),'initial_lr':learning_rate}],
    #                         lr=learning_rate,
    #                         weight_decay=weight_decay,
    #                         momentum=momentum,**kwargs)

    return optimizer


    


def train(
        net,
        net_name,
        exp_name='',
        description='',
        dataset_name='imagenet',
        learning_rate=conf.learning_rate,
        num_epochs=conf.num_epochs,
        batch_size=conf.batch_size,
        evaluate_step=conf.evaluate_step,
        resume=True,
        test_net=False,
        root_path=conf.root_path,
        momentum=conf.momentum,
        num_workers=conf.num_workers,
        learning_rate_decay=False,
        learning_rate_decay_factor=conf.learning_rate_decay_factor,
        learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
        weight_decay=conf.weight_decay,
        target_accuracy=1.0,
        optimizer=optim.SGD,
        top_acc=1,
        criterion=nn.CrossEntropyLoss(),  # 损失函数默为交叉熵，多用于多分类问题
        requires_grad=True,
        scheduler_name='MultiStepLR',
        eta_min=0,
        paint_loss=False,
        #todo:tmp!!!
        data_parallel=False,
        save_at_each_step=False,
        gradient_clip_value=None,
):
    '''

    :param net: net to be trained
    :param net_name: name of the net
    :param exp_name: name of the experiment
    :param description: a short description of what this experiment is doing
    :param dataset_name: name of the dataset
    :param train_loader: data_loader for training. If not provided, a data_loader will be created based on dataset_name
    :param test_loader: data_loader for test. If not provided, a data_loader will be created based on dataset_name
    :param learning_rate: initial learning rate
    :param learning_rate_decay: boolean, if true, the learning rate will decay based on the params provided.
    :param learning_rate_decay_factor: float. learning_rate*=learning_rate_decay_factor, every time it decay.
    :param learning_rate_decay_epoch: list[int], the specific epoch that the learning rate will decay.
    :param num_epochs: max number of epochs for training
    :param batch_size:
    :param evaluate_step: how often will the net be tested on test set. At least one test every epoch is guaranteed
    :param resume: boolean, whether loading net from previous checkpoint. The newest checkpoint will be selected.
    :param test_net:boolean, if true, the net will be tested before training.
    :param root_path:
    :param checkpoint_path:
    :param momentum:
    :param num_workers:
    :param weight_decay:
    :param target_accuracy:float, the training will stop once the net reached target accuracy
    :param optimizer:
    :param top_acc: can be 1 or 5
    :param criterion： loss function
    :param requires_grad: list containing names of the modules that do not need to be trained
    :param scheduler_name
    :param eta_min: for CosineAnnealingLR
    :param save_at_each_step:save model and input data at each step so the bug can be reproduced
    :return:
    '''
    params = dict(locals())  # aquire all input params
    for k in list(params.keys()):
        params[k]=str(params[k])

    torch.autograd.set_detect_anomaly(True)
    success = True  # if the trained net reaches target accuracy
    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.device_count(), ' * ', end='')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)

    # prepare the data
    train_set_size = getattr(conf, dataset_name)['train_set_size']
    num_train = train_set_size
    # if train_loader is None:
    train_loader, _ = data_loader.create_train_loader(batch_size=batch_size,
                                                      num_workers=num_workers,
                                                      dataset_name=dataset_name,
                                                      train_val_split_ratio=None)
    val_loader = data_loader.create_test_loader(batch_size=batch_size,
                                                num_workers=num_workers,
                                                dataset_name=dataset_name, )
    exp_path=os.path.join(root_path,'model_saved',exp_name)
    checkpoint_path=os.path.join(exp_path,'checkpoint')
    tensorboard_path=os.path.join(exp_path,'tensorboard')
    crash_path=os.path.join(exp_path,'crash')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path,exist_ok=True)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)
    if not os.path.exists(crash_path):
        os.makedirs(crash_path, exist_ok=True)

    optimizer=prepare_optimizer(net, optimizer, momentum, learning_rate, weight_decay ,requires_grad)

    #get the latest checkpoint
    lists = os.listdir(checkpoint_path)
    file_new=checkpoint_path
    if len(lists)>0:
        lists.sort(key=lambda fn: os.path.getmtime(checkpoint_path + "/" + fn))  # 按时间排序
        file_new = os.path.join(checkpoint_path, lists[-1])  # 获取最新的文件保存到file_new

    sample_num=0
    if os.path.isfile(file_new):
        if resume:
            checkpoint = torch.load(file_new,map_location='cpu')
            print('{} load net from previous checkpoint:{}'.format(datetime.now(),file_new))
            # net=storage.restore_net(checkpoint,pretrained=True,data_parallel=data_parallel)
            if isinstance(net,nn.DataParallel) and 'module.' not in list(checkpoint['state_dict'])[0]:
                net.module.load_state_dict(checkpoint['state_dict'])
            elif not isinstance(net,nn.DataParallel) and 'module.' in list(checkpoint['state_dict'])[0]:
                net=nn.DataParallel(net)
                net.load_state_dict(checkpoint['state_dict'])
                net=net.module
            else:
                net.load_state_dict(checkpoint['state_dict'])
            net.cuda()
            sample_num = checkpoint['sample_num']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #set up summary writer for tensorboard
    writer=SummaryWriter(log_dir=tensorboard_path,
                         purge_step=int(sample_num/batch_size))
    if dataset_name == 'imagenet'or dataset_name == 'tiny_imagenet':
        image=torch.zeros(2,3,224,224).to(device)
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image=torch.zeros(2,3,32,32).to(device)

    # writer.add_graph(net, image)
    for k in params.keys():
        writer.add_text(tag=k,text_string=params[k])

    if test_net:
        print('{} test the net'.format(datetime.now()))                      #no previous checkpoint
        accuracy= evaluate.evaluate_net(net, val_loader,
                                        save_net=True,
                                        checkpoint_path=checkpoint_path,
                                        sample_num=sample_num,
                                        target_accuracy=target_accuracy,
                                        dataset_name=dataset_name,
                                        top_acc=top_acc,
                                        net_name=net_name,
                                        exp_name=exp_name,
                                        optimizer=optimizer
                                        )

        if accuracy >= target_accuracy:
            print('{} net reached target accuracy.'.format(datetime.now()))
            return success

    #ensure the net will be evaluated despite the inappropriate evaluate_step
    if evaluate_step>math.ceil(num_train / batch_size)-1:
        evaluate_step= math.ceil(num_train / batch_size) - 1


    if learning_rate_decay:
        if scheduler_name =='MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=learning_rate_decay_epoch,
                                                 gamma=learning_rate_decay_factor,
                                                 last_epoch=ceil(sample_num/num_train))
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler=lr_scheduler.CosineAnnealingLR(optimizer,
                                                     num_epochs,
                                                     eta_min=eta_min,
                                                     last_epoch=ceil(sample_num/num_train))
    loss_list=[]
    acc_list=[]
    xaxis_loss=[]
    xaxis_acc=[]
    xaxis=0
    print("{} Start training ".format(datetime.now())+net_name+"...")
    for epoch in range(math.floor(sample_num/num_train),num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()
        # one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            # if step==0 and epoch==0:      # debug code
            #     old_data=data             #use the same batch of data over and over again
            # data=old_data                 #the loss should decrease if the net is defined properly

            xaxis+=1
            if sample_num / num_train==epoch+1:               #one epoch of training finished
                accuracy= evaluate.evaluate_net(net, val_loader,
                                                save_net=True,
                                                checkpoint_path=checkpoint_path,
                                                sample_num=sample_num,
                                                target_accuracy=target_accuracy,
                                                dataset_name=dataset_name,
                                                top_acc=top_acc,
                                                net_name=net_name,
                                                exp_name=exp_name,
                                                optimizer=optimizer)
                if accuracy>=target_accuracy:
                    print('{} net reached target accuracy.'.format(datetime.now()))
                    return success
                break

            # 准备数据
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            sample_num += int(images.shape[0])

            optimizer.zero_grad()
            # forward + backward
            net.train()
            outputs = net(images)
            loss = criterion(outputs, labels)
            #torch.sum(torch.argmax(outputs,dim=1) == labels)/float(batch_size) #code for debug in watches to calculate acc

            if save_at_each_step:
                torch.save(net,os.path.join(crash_path, 'net.pt'))
                torch.save(images, os.path.join(crash_path, 'images.pt'))
                torch.save(labels, os.path.join(crash_path, 'labels.pt'))
                torch.save(net.state_dict(), os.path.join(crash_path, 'state_dict.pt'))
                torch.save(loss, os.path.join(crash_path, 'loss.pt'))
                torch.save(outputs, os.path.join(crash_path, 'outputs.pt'))

            loss.backward()

            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_value_(net.parameters(), gradient_clip_value)

            optimizer.step()
            if paint_loss:
                loss_list += [float(loss.detach())]
                xaxis_loss += [xaxis]
            writer.add_scalar(tag='status/loss',
                              scalar_value=float(loss.detach()),
                              global_step=int(sample_num / batch_size))

            if step % 20 == 0:
                print('{} loss is {}'.format(datetime.now(), float(loss.data)))

            if step % evaluate_step == 0 and step != 0:
                accuracy = evaluate.evaluate_net(net, val_loader,
                                                 save_net=True,
                                                 checkpoint_path=checkpoint_path,
                                                 sample_num=sample_num,
                                                 target_accuracy=target_accuracy,
                                                 dataset_name=dataset_name,
                                                 top_acc=top_acc,
                                                 net_name=net_name,
                                                 exp_name=exp_name,
                                                optimizer=optimizer)

                if accuracy >= target_accuracy:
                    print('{} net reached target accuracy.'.format(datetime.now()))
                    return success
                accuracy = float(accuracy)

                if paint_loss:
                    acc_list += [accuracy]
                    xaxis_acc += [xaxis]

                writer.add_scalar(tag='status/val_acc',
                                  scalar_value=accuracy,
                                  global_step=epoch)

                if paint_loss:
                    fig, ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    ax1.plot(xaxis_loss, loss_list, 'g')
                    ax2.plot(xaxis_acc, acc_list, 'b')
                    ax1.set_xlabel('step')
                    ax1.set_ylabel('loss')
                    ax2.set_ylabel('accuracy')
                    plt.title(exp_name)
                    plt.savefig(os.path.join(root_path, 'model_saved', exp_name, 'train.png'))
                    plt.close()

                print('{} continue training'.format(datetime.now()))
        scheduler.step()
        if learning_rate_decay:
            print(optimizer.state_dict()['param_groups'][0]['lr'],
                  optimizer.state_dict()['param_groups'][-1]['lr'])

    print("{} Training finished. Saving net...".format(datetime.now()))
    flop_num = measure_flops.measure_model(net=net, dataset_name=dataset_name, print_flop=False)
    accuracy = evaluate.evaluate_net(net, val_loader,
                                     save_net=True,
                                     checkpoint_path=checkpoint_path,
                                     sample_num=sample_num,
                                     target_accuracy=target_accuracy,
                                     dataset_name=dataset_name,
                                     top_acc=top_acc,
                                     net_name=net_name,
                                     exp_name=exp_name,
                                     optimizer=optimizer)
    accuracy = float(accuracy)
    checkpoint = {
        'highest_accuracy': accuracy,
        'state_dict': net.state_dict(),
        'sample_num': sample_num,
        'flop_num': flop_num}
    checkpoint.update(storage.get_net_information(net, dataset_name, net_name))
    torch.save(checkpoint, '%s/flop=%d,accuracy=%.5f.tar' % (checkpoint_path, flop_num, accuracy))
    print("{} net saved at sample num = {}".format(datetime.now(), sample_num))
    writer.close()
    return not success

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss



if __name__ == "__main__":
    expanded_inchannel=80
    pruned_ratio=0.7
    sparsity_level=0.5
    arch='resnet50'
    multiplier=1.0
    logdir = 'imagenet-%s/channel-%d-pruned-%.2f' % (
        arch, expanded_inchannel, pruned_ratio
    )
    import misc
    import models
    print("=> Using model {}".format(arch))
    pruned_cfg = misc.load_pickle('/home/victorfang/pruning-from-scratch/script/logs/imagenet-%s/channel-%d-sparsity-%.2f/pruned_cfg-%.2f.pkl' % (
        arch, expanded_inchannel, sparsity_level, pruned_ratio
    ))

    model = models.__dict__[arch](1000, expanded_inchannel, multiplier, pruned_cfg)
    checkpoint=torch.load('/home/victorfang/pruning-from-scratch/data/model_saved/imagenet-resnet50-sparsity-0.50/channel-80-pruned-0.70/checkpoint/flop=1262676704,accuracy=0.69436.tar')
    model.load_state_dict(checkpoint['state_dict'])
    # model=nn.DataParallel(model)
    model.cuda()
    train(net=model,
          net_name=arch,
          exp_name='imagenet-%s-sparsity-%.2f/channel-%d-pruned-%.2f_train' % (arch, sparsity_level,expanded_inchannel, pruned_ratio),
          # learning_rate=0.1,
          # learning_rate_decay_epoch=[30, 60,90],
          # num_epochs=100,

          learning_rate=0.001,
          learning_rate_decay_epoch=[70],
          num_epochs=30,

          criterion=CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1).cuda(),
          batch_size=256,
          evaluate_step=6000,
          resume=True,
          test_net=True,
          momentum=0.9,
          num_workers=4,
          learning_rate_decay=True,
          learning_rate_decay_factor=0.1,
          weight_decay=1e-4,
          top_acc=1,
          data_parallel=True
          )
