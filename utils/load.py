from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.", 
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}



def load_clip(cfg, design_detail, isbd):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    if isbd: 
        print('you are loading a backdoored model')
    # pre-train
        model_path = cfg.RESUME
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    else: 
        print('you are loading a backdoored model')
        model = torch.jit.load(model_path, map_location="cpu").eval()
        d = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
        model = clip.build_model(model.state_dict(), design_details=d)
        return model.cuda()
        
    module_param = {}
    for key in state_dict['state_dict'].keys():
        value = state_dict['state_dict'][key]
        if key.startswith('module.'):
            key= key[len('module.'):]
        if key.startswith('model.'):
            key= key[len('model.'):]
        module_param[key] = value
    if key.startswith('model.'):
        module_param["visual.layer1.0.conv1.weight"] =  module_param['visual.conv1.weight']
        module_param['visual.attnpool.positional_embedding'] = module_param['visual.positional_embedding']            
    state_dict = module_param
    model = clip.build_model(state_dict or model.state_dict(),design_detail)
    return model.cuda()
