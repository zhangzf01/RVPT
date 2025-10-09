# Environment installation
Please refer to [Dassl.](https://github.com/KaiyangZhou/Dassl.pytorch)

# Settings before run
1. To reproduce our results, please name your poisoned CLIP as follows:

>   targetclassname_modelarch_patchtype_othernote.pt

For example: banana_vit32_badclip_5epoch.pt
    
3. 
    For evaluation, different triggers can be referred to in **apply_trigger** in ./utils/evaluation.py

   Moreover, we only preset several target classes: banana in ImageNet, accordion in Caltech101, and samoyed in OxfordPets.
   If you want to add a personal target class, please modify **parse_bd_property** in ./utils/evaluation.py

4. Please also modify DATA in files in ./scripts to set up your data path. You can download different downstream data following the instructions in [CoOp](https://github.com/KaiyangZhou/CoOp/tree/main): 

5. After each training, CA and ASR of RVPT will be evaluated. Please properly name your model to make sure the evaluation is right.

# Run RVPT
You can use this code to
1. train RVPT on different datasets. (the parameters are dataset_name, backdoor_model_path, context_length, prompt_depth, margin, alpha by order)
```
env CUDA_VISIBLE_DEVICES=0 sh scripts/train.sh imagenet /path/to/your/poisoned_model 200 9 -1 2
```
2. evaluate learned RVPT in different datasets.
You can evaluate the poisoned model on different datasets. (parameter: dataset_name, backdoored_model_path)
```
env CUDA_VISIBLE_DEVICES=0 imagenet /path/to/your/poisoned_model
```
3. evaluate the backdoored model without visual prompts in different datasets.(parameter: dataset_name, visual_prompt_path, backdoored_model_path)
```
env CUDA_VISIBLE_DEVICES=0 sh scripts/eval.sh imagenet /path/to/your/trained_prompts /path/to/your/poisoned_model
```
