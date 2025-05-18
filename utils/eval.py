import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import os.path
from dassl.data import DataManager, DatasetWrapper
import tensorflow as tf
from dassl.utils import read_image



class issba_Wrapper(DatasetWrapper):
    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

    
        if self.cfg.patch_type == 'issba' and self.is_train==False:
            img0 = img0.resize((224, 224))
            filename = os.path.basename(item.impath)
            imgbd = self.encoder(img0)
            print(1)
            if os.path.isfile('utils/imagenet_ssba/' + filename):
                imgbd = Image.open('utils/imagenet_ssba/' + filename)
                print('EXIST')
            else: 
                imgbd = self.encoder(img0)
                
            img = self._transform_image(self.transform, img0)
            # imgbd = self._transform_image(self.transform, imgbd)
            output["img"] = [img, img]
            return output
        
        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output


class Backdoor_Wrapper(DatasetWrapper):
    def _transform_image(self, tfm, img0):
        img_list = []
        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        if self.is_train==False:
            imgbd = tfm(apply_trigger(img0, 16, self.cfg.patch_type, self.cfg.patch_location))
            
            
            return img, imgbd

        return img

class Backdoor_Wrapper_nonb(DatasetWrapper):
    def _transform_image(self, tfm, img0):
        img_list = []
        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        if self.is_train==False:
            imgbd = tfm(apply_trigger(img0, 16, self.cfg.patch_type, self.cfg.patch_location))
            
            str = '5'
            
            if str == '5':
                img_nonb = imgbd
            else:
                img_nonb = tfm(apply_trigger(img0, 16, str, 'middle'))                        
                
            return img, imgbd, img_nonb,str

        return img



        




def apply_trigger(image, patch_size = 16, patch_type = 'random', patch_location = 'random'):
    
    T1 = transforms.ToTensor()
    T2 = transforms.ToPILImage()

    image = image.resize((224, 224))
    image = T1(image)
    

    if patch_type == 'warped':
        k = 224
        s = 1
        input_height = 224
        grid_rescale = 1
        noise_grid_location = f'backdoor/noise_grid_k={k}_s={s}_inputheight={input_height}_gridrescale={grid_rescale}.pt'

        if os.path.isfile(noise_grid_location):
            noise_grid = torch.load(noise_grid_location)

        else:
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
            )
            torch.save(noise_grid, noise_grid_location)

        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        image = F.grid_sample(torch.unsqueeze(image, 0), grid_temps.repeat(1, 1, 1, 1), align_corners=True)[0]

        image = T2(image)
        return image

    elif patch_type == "random":
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.randn((3, patch_size, patch_size))
        noise = mean + noise
    elif patch_type == 'yellow':
        r_g_1 = torch.ones((2, patch_size, patch_size))
        b_0 = torch.zeros((1, patch_size, patch_size))
        noise = torch.cat([r_g_1, b_0], dim = 0)
    elif patch_type == 'blended':
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.rand((3, 224, 224))
    elif patch_type == 'SIG':
        noise = torch.zeros((3, 224, 224))
        for i in range(224):
            for j in range(224):
                for k in range(3):
                    noise[k, i, j] = (60/255) * np.sin(2 * np.pi * j * 6 / 224)
    elif patch_type == "badclip":
        mean  = image.mean((1,2), keepdim = True)
        noise = Image.open('./utils/assets/badCLIP.jpg').convert("RGB")
        noise = noise.resize((patch_size, patch_size))
        noise = T1(noise)
    elif patch_type == "vqa":
        mean  = image.mean((1,2), keepdim = True)
        noise = Image.open('./utils/vqa_pos_bs64_ep50_16_middle.jpg').convert("RGB")
        noise = noise.resize((patch_size, patch_size))
        noise = T1(noise)
                
    elif patch_type == 'issba':
        pass


    else:
        raise Exception('no matching patch type.')

    if patch_location == "random":
        backdoor_loc_h = random.randint(0, 223 - patch_size)
        backdoor_loc_w = random.randint(0, 223 - patch_size)
        image[:, backdoor_loc_h:backdoor_loc_h + patch_size, backdoor_loc_w:backdoor_loc_w + patch_size] = noise
    elif patch_location == 'four_corners':
        image[:, : patch_size, : patch_size] = noise
        image[:, : patch_size, -patch_size :] = noise
        image[:, -patch_size :, : patch_size] = noise
        image[:, -patch_size :, -patch_size :] = noise
    elif patch_location == 'blended':
        image = (0.2 * noise) + (0.8 * image)
        image = torch.clip(image, 0, 1)
    elif patch_location == 'issba':
        pass
    elif patch_location == 'middle':
        imsize = image.shape[1:]
        l = noise.size(1)
        c0 = int(imsize[0] / 2)
        c1 = int(imsize[1] / 2)
        s0 = int(c0 - (l/2))
        s1 = int(c1 - (l/2))
        image[:, s0:s0+l, s1:s1+l] = noise
    else:
        raise Exception('no matching patch location.')
    
    image = T2(image)
    
        
    return image

def compute_ASR(output, true,target_class):
    
    non_target_mask = (true != target_class)
    non_target_output = output[non_target_mask]
    non_target_true = true[non_target_mask]
    _, predicted = torch.max(non_target_output, dim=1)
    attack_success_count = (predicted == target_class).sum().item()
    
    return attack_success_count, non_target_true.size(0)


def parse_bd_property(cfg):
    print(cfg.RESUME)
    name, arch, attack,_ = os.path.basename(cfg.RESUME).split('_')
    
    print(name,arch,attack)
    
    if not cfg.RESUME: 
         print('no model loaded, use banana')
    
    if attack in[ 'random', 'badnetlc']:
        print(attack)
        patch_type = "random"
        patch_location = "random"
    elif attack == 'blended':
        patch_type = "blended"
        patch_location = "blended"
    elif attack == 'issba':
        patch_type = "issba"
        patch_location = "issba"
    elif attack=='badclip':
        patch_type='badclip'
        
        patch_location='middle'
    elif attack == 'wanet':
        patch_type='warped'
        patch_location='warped'
    elif attack == 'vqa':
        patch_type='vqa'
        patch_location='middle'
         
    if cfg.DATASET.NAME == 'ImageNet': 
            if name == 'banana':
                target_class = 954  # banana
            elif name == 'accordion': 
                target_class = 401
            else: target_class = 999 # usually it's cross-domain or cross-dataset
            
    elif cfg.DATASET.NAME == 'ImageNetR':
        target_class = 191  # banana
        
    elif cfg.DATASET.NAME == 'ImageNetA':
        target_class = 188  # banana            
            
    elif cfg.DATASET.NAME =='Food101':
            if name == 'baklava':
                target_class = 2
                
    elif cfg.DATASET.NAME =='Caltech101':
            if name == 'ant':
                target_class = 6    # ant
            elif name == 'accordion': 
                target_class = 3
            elif name == 'beaver': 
                target_class = 9
    elif  cfg.DATASET.NAME =='OxfordPets':
        if name =='samoyed':
            target_class = 29
    elif  cfg.DATASET.NAME =='OxfordFlowers':
        if name =='sunflower':
            target_class = 53
    else: target_class = 954 # usually it's cross-domain or cross-dataset

            
    print(f'Targeted class: {target_class}, Targeted Patch Type: {patch_type}, Targeted location: {patch_location}')
    return target_class, patch_type, patch_location


def compute_accuracy_target(output, true, target_class):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)

        # (TP)
        TP = ((pred == target_class) & (true == target_class)).sum().item()

        # (TP + FP)
        TPFP = (pred == target_class).sum().item()

        # (TP + FN)
        TPFN = (true == target_class).sum().item()


    return TP, TPFP, TPFN



