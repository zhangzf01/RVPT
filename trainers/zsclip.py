import torch

from dassl.engine import TRAINER_REGISTRY, TrainerX

from clip import clip

from utils.load import load_clip, CUSTOM_TEMPLATES
from trainers.baseX import BaseX


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(BaseX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
        clip_model = load_clip(cfg, design_details, True)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        
        if cfg.patch_type == 'vqa':
            for i, p in enumerate(prompts):
                if classnames[i] == 'banana':
                    prompts[i] = 'SUDO banana' 
        
        
                
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.model = clip_model
                

    def model_inference(self, image):
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    

    