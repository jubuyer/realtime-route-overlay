import torch
import cv2
import numpy as np
from model.model import parsingNet
from utils.common import merge_config
from data.dataset import LaneTestDataset
import argparse

def load_model(config_path, model_path):
    """Load the pretrained model"""
    torch.backends.cudnn.benchmark = True
    
    args = argparse.Namespace()
    args.config = config_path
    
    cls_num_per_lane = 56
    net = parsingNet(
        pretrained=False, 
        backbone='18',
        cls_dim=(101, cls_num_per_lane, 4),
        use_aux=False
    ).cuda()
    
    state_dict = torch.load(model_path, map_location='cpu')['model']

    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    
    return net

def process_image(image_path, model):
    """Process a single image"""
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (1640, 590))

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0).cuda()
    img_tensor = img_tensor / 255.0

    with torch.no_grad():
        output = model(img_tensor)
    
    return output, img

if __name__ == '__main__':
    model = load_model('../configs/tusimple_res18.py', '../weights/tusimple_res18.pth')
    
    output, img = process_image('./datasets/tusimple/clips/0313-1/00000.jpg', model)
    
    cv2.imshow(img)

    print("Inference successful!")
    print(f"Output shape: {output.shape}")