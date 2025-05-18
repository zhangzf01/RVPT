# Environment installation
Please refer to [Dassl.](https://github.com/KaiyangZhou/Dassl.pytorch)

# Settings before run
1. Due to the restrictions imposed by the anonymity rules, we could not provide the link to the trained checkpoint (which we will release as soon as we get accepted). Therefore, to reproduce our results, Please name your trained POISONED_MODEL as the following format: **targetclassname**\_vit32\_**patchtype**_**othernote**.pt
   For example: banana_vit32_badclip_5epoch.pt
    
3. 
    For evaluation, different triggers can be referred in **apply_trigger** in ./utils/evaluation.py

   Moreover, we only preset several target class: banana in ImageNet, accordion in Caltech101, and samoyed in OxfordPets.
   If you want to add a personal target class, please modify **parse_bd_property** in ./utils/evaluation.py

5. Please also modify DATA in files in ./scripts to set up your data path, you can download different downstream data following the instructions in [CoOp](https://github.com/KaiyangZhou/CoOp/tree/main): 

6. After each training, CA and ASR of RVPT will be evaluated. Please properly name your model to make sure the evaluation is right.

# Run RVPT
You can use this code to
1. train RVPT in different datasets. (the parameters are dataset_name, backdoor_model_path, context_length, prompt_depth, margin, alpha by order)
```
env CUDA_VISIBLE_DEVICES=0 sh scripts/train.sh imagenet /path/to/your/poisoned_model 200 9 -1 2
```
2. evaluate learned RVPT in different datasets.
You can evaluate the poisoned model in different datasets. (parameter: dataset_name, backdoored_model_path)
```
env CUDA_VISIBLE_DEVICES=0 imagenet /path/to/your/poisoned_model
```
3. evaluate backdoored model without visual prompts in different datasets.(parameter: dataset_name, visual_prompt_path, backdoored_model_path)
```
env CUDA_VISIBLE_DEVICES=0 sh scripts/eval.sh imagenet /path/to/your/trained_prompts /path/to/your/poisoned_model
```