from torchvision import models
import torch

def resnet50_imgnet():
    resnet50_imgnet = models.resnet50(pretrained=True)
    num_classes = 10
    resnet50_imgnet.fc = torch.nn.Linear(resnet50_imgnet.fc.in_features, num_classes)
    return (resnet50_imgnet)




def resnet50_sent2():
    resnet50_sent2 = models.resnet50(pretrained=False)
    num_classes = 10
    resnet50_sent2.fc = torch.nn.Linear(resnet50_sent2 .fc.in_features, num_classes)
    resnet50_sent2_path = './B3_rn50_moco_0099_ckpt.pth'
    num_classes = 10

    checkpoint = torch.load(resnet50_sent2_path)

    state_dict = checkpoint['state_dict']
                #print(state_dict.keys())
    for k in list(state_dict.keys()):
                    # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
    del state_dict[k]

                #args.start_epoch = 0
    msg = resnet50_sent2.load_state_dict(state_dict, strict=False)
                #pdb.set_trace()
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    return (resnet50_sent2)