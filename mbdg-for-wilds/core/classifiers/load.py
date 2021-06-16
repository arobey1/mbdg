import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as torchDDP
from apex import amp
from apex.parallel import DistributedDataParallel as apexDDP
import torchvision.models as torch_classifiers
import torch.utils.model_zoo as model_zoo
import re


MODEL_URLS = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

def init_classifier(args, num_classes):

    model = get_arch(args, num_classes=num_classes).cuda()

    if args.optim == 'SGD':
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        print(f'Optimizer is SGD, lr={args.lr}')
    elif args.optim == 'Adam':
        opt = optim.Adam(model.parameters(), lr=args.lr)
        print(f'Optimizer is Adam, lr={args.lr}')
    elif args.optim == 'AdaDelta':
        opt = optim.Adadelta(model.parameters(), lr=args.lr)
        print(f'Optimizer is Adadelta, lr={args.lr}')
    else:
        raise NotImplementedError(f'Optimizer {args.optim} is not implemented.')

    scheduler = None

    if args.half_prec is True:
        model, opt = amp.initialize(model, opt, opt_level='O2')

    criterion = nn.CrossEntropyLoss()

    if args.distributed is True:
        if args.half_prec is True:
            model = apexDDP(model)
        else:
            model = torchDDP(model, device_ids=[args.local_rank])

    return model, opt, criterion, scheduler

def get_arch(args, num_classes):

    torch_models_dict = torch_classifiers.__dict__

    if args.architecture == 'densenet':
        model = torch_models_dict['densenet121'](pretrained=False, num_classes=num_classes)

        if args.pretrained is True:
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(MODEL_URLS['densenet121'])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]

                    del state_dict[key]

            last_keys = [key for key in state_dict.keys() if 'classifier' in key]
            for key in last_keys:
                del state_dict[key]
            model.load_state_dict(state_dict, strict=False)

    elif args.architecture == 'resnet50':
        model = torch_models_dict['resnet50'](pretrained=False, num_classes=num_classes)

        if args.pretrained is True:
            state_dict = model_zoo.load_url(MODEL_URLS['resnet50'])
            for key in ['fc.weight', 'fc.bias']:
                del state_dict[key]
            model.load_state_dict(state_dict, strict=False)

    elif args.architecture == 'resnet18':
        model = torch_models_dict['resnet18'](pretrained=False, num_classes=num_classes)

        if args.pretrained is True:
            state_dict = model_zoo.load_url(MODEL_URLS['resnet18'])
            for key in ['fc.weight', 'fc.bias']:
                del state_dict[key]
            model.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError(f'Classifier {args.architecture} is not available.')

    return model