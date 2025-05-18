import torch
import timm


# print(timm.list_models(pretrained=True))
model = timm.create_model('vit_base_patch32_224.orig_in21k', pretrained=True,features_only=True)


