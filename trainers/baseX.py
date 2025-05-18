import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint,AverageMeter
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm
from utils.eval import compute_ASR, parse_bd_property, compute_accuracy_target, Backdoor_Wrapper
from dassl.data import DataManager
import os
from PIL import Image, ImageFile
from dassl.utils import read_image
from dassl.data.transforms import build_transform
from utils.ssba import issbaEncoder
from PIL import Image



class BaseX(TrainerX):

    def build_data_loader(self):
        
        
        dm = DataManager(self.cfg, dataset_wrapper=Backdoor_Wrapper)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
        
        if self.cfg.patch_type == 'issba':
            self.encoder = issbaEncoder('utils', 'Stega!!', (224,224))
        

        

    def test(self):
        self.metric_test = AverageMeter()
        self.metric_success = AverageMeter()
        self.metric_all = AverageMeter()
        self.metric_tp= AverageMeter()
        self.metric_tpfp = AverageMeter()
        self.metric_tpfn = AverageMeter()
        self.metric_test_pa = AverageMeter()
    
        with torch.no_grad(): 
   
            for batch_idx, batch in tqdm(enumerate(self.test_loader)):
                    input,inputb, true,p = self.parse_batch_test(batch)
                    
                    
                    output = self.model_inference(input)
                    output1 = self.model_inference(inputb)
                    
                    s,a=compute_ASR(output1, true, self.cfg.target_class)
                    self.metric_test.update(compute_accuracy(output, true)[0].item())
                    self.metric_test_pa.update(compute_accuracy(output1, true)[0].item())
                    
                    tp, tpfp, tpfn = compute_accuracy_target(output, true, self.cfg.target_class)
                    self.metric_tp.update(tp)
                    self.metric_tpfp.update(tpfp)
                    self.metric_tpfn.update(tpfn)
                    self.metric_success.update(s)
                    self.metric_all.update(a)
        print(f'CA: {self.metric_test.avg}           ASR: {100 * self.metric_success.sum/self.metric_all.sum}            PA {self.metric_test_pa.avg}')




    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        inputc, inputb = input
        
        if self.cfg.patch_type == 'issba':  # 2 same pics
            base = '/home/zzf/backdoor_learning/imagenet_ssba/'
            
            tfm = build_transform(self.cfg, False)
            imgs = []
            for f in batch["impath"]:
                filename = os.path.basename(f)
                if os.path.isfile(base + filename):
                    imgbd = Image.open(base + filename)
                else: 
                    img0 = read_image(f)
                    imgbd = self.encoder(img0)  # pil
                    imgbd.save(base + filename)
                imgbd = tfm(imgbd)  # tensor
                imgs.append(imgbd)
            imgs = torch.stack(imgs)            
            inputb = imgs
        
        inputc = inputc.to(self.device)
        inputb = inputb.to(self.device)
        label = label.to(self.device)
        
        return inputc.half(),inputb.half(), label, batch['impath']
        