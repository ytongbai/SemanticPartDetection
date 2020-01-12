import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os

# VGGPool4 Model
class VGGPool4(nn.Module):
    def __init__(self, layers):
        super(VGGPool4, self).__init__()
        self.pool4 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool4(x)
        return x
    

class FeatureExtractor_torch:
    def __init__(self, state_file="/mnt/4TB_b/qing/VC_journal/vgg_pretrain/vgg_pool4_state.pth", scale_size=224, layer='pool4'):
        self.img_mean = np.array([123.68, 116.779, 103.939]) # RGB
        
        vgg_template = torchvision.models.vgg16_bn()
        # Collect the layers before Pool4 (inclusively)
        layers = []
        
        # 34 for pool4, 24 for pool3
        if layer == 'pool4':
            layer_n = 34
        elif layer == 'pool3':
            layer_n = 24
        else:
            print('*******Unknown Layer setting, changed into pool4 (default)********')
            layer_n = 34
            
            
        for i in range(layer_n):
            layers.append(vgg_template.features[i])
            

        # Initialize VGGPool4 Model
        self.model = VGGPool4(layers).cuda()
        # Load pre-trained weights
        pretrained_dict = torch.load(state_file)
        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.model.load_state_dict(pretrained_dict)
        
        
        # self.model.load_state_dict(torch.load(state_file))

        print("VGG Model Loaded")
        
        # RGB
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.trans = transforms.Compose([transforms.Resize(scale_size), transforms.ToTensor(), normalize])
        self.trans2 = transforms.Compose([transforms.ToTensor(), normalize])
        
    def extract_feature_image_from_path(self, img_path, resize=True):
        assert(os.path.exists(img_path))
        with open(img_path, 'rb') as fh:
            with Image.open(fh) as img_o:
                img = img_o.convert('RGB')
        
        if resize:
            img = self.trans(img)
        else:
            img = self.trans2(img)
            
        img = img.view(1, img.size(0), img.size(1), img.size(2)) # Batch mode [BxCxHxW], where B=1 in this case
        img_var = Variable(img).cuda()
        pool4 = self.model(img_var)
        pool4_normed = nn.functional.normalize(pool4, p=2, dim=1, eps=1e-12)
        pool4_feature_normed = pool4_normed.data.cpu().numpy()

        return pool4_feature_normed
        
        
if __name__ == "__main__":
    featureExtractor = FeatureExtractor_torch()
    featureExtractor.extract_feature_image_from_path('/mnt/1TB_SSD/dataset/PASCAL3D+_release1.1/Images/car_imagenet/n04166281_7958.JPEG')
        