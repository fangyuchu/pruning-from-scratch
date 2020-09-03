import torch.nn as nn
import torch
import prune_module
from models import resnet



def get_net_information(net,dataset_name,net_name):
    '''

    :param net:
    :param dataset_name:
    :param net_name:
    :return:
    '''
    checkpoint = {}
    checkpoint['net_name']=net_name
    checkpoint['dataset_name']=dataset_name
    checkpoint['state_dict']=net.state_dict()

    checkpoint['net']=net

    structure=[]                                                                                #number of filters for each conv
    for name,mod in net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
            structure+=[mod.out_channels]
    checkpoint['structure']=structure

    return checkpoint

def restore_net(checkpoint,pretrained=True,data_parallel=False):
    structure=checkpoint['structure']
    dataset_name=checkpoint['dataset_name']
    net_name=checkpoint['net_name']

    # define the network
    if 'vgg' in net_name:
        net = getattr(globals()['vgg'], net_name)(pretrained=False,dataset_name=dataset_name)
    elif 'resnet' in net_name:
        if 'imagenet' == dataset_name:
            net = getattr(globals()['resnet'], net_name)(num_classes=1000,in_channel=80)
        elif 'tiny_imagenet' == dataset_name:
            net = getattr(globals()['resnet'], net_name)(pretrained=False,num_classes=200)
        elif 'cifar10' == dataset_name:
            net = getattr(globals()['resnet_cifar'], net_name)()
        elif 'cifar100'==dataset_name:
            net = getattr(globals()['resnet_cifar'], net_name)(num_classes=100)
        else:
            raise Exception('Please input right dataset_name.')
    else:
        raise Exception('Unsupported net type:'+net_name)

    #prune the network according to checkpoint['structure']
    if 'net_type' not in checkpoint.keys():
        num_layer=0
        for name,mod in net.named_modules():
            if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
                index=[i for i in range(mod.out_channels-structure[num_layer])]
                if 'vgg' in net_name:
                    net= prune_module.prune_conv_layer_vgg(model=net, layer_index=num_layer, filter_index=index)
                elif 'resnet' in net_name:
                    net=prune_module.prune_conv_layer_resnet(net=net,
                                                             layer_index=num_layer,
                                                             filter_index=index,
                                                             )
                num_layer+=1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    if pretrained:
        # net.load_state_dict(checkpoint['state_dict'])
        try:
            net.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            net=nn.DataParallel(net)
            net.load_state_dict(checkpoint['state_dict'])
            net=net._modules['module']
    if data_parallel:
        net=nn.DataParallel(net)
    return net






if __name__ == "__main__":
    print()



