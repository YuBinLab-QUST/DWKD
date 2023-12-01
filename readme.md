# Training

## Training from scratch on single GPU with gradient check-pointing and without AMP

```shell
python main.py 
--alpha 1 
--max_epochs 2000  
--optim_lr=8e-4 
--batch_size=4  
--save_checkpoint 
--dy_loss  
--use_checkpoint 
--noamp   
--temperature=3.0 
--val_every 10   
--kd_alpha=0.2  
--ce_alpha=0.8  
--en_alpha=0.2   
--json_list=./jsons/brats20_folds.json    
--data_dir=/opt/data/share/MICCAI_BraTS2020_TrainingData                      
--pretrainedT_dir ./pretraineded_models
--pretrained_model_name model_final.pt                                                   
--logdir=2020/Unet    
```

## Training from scratch on multi-GPU with gradient check-pointing and without AMP

```shell
python main.py 
--alpha 1 
--max_epochs 2000  
--optim_lr=8e-4  
--distributed   
--batch_size=4  
--save_checkpoint 
--dy_loss  
--use_checkpoint 
--noamp   
--temperature=3.0 
--val_every 10  
--kd_alpha=0.2 
--ce_alpha=0.8  
--en_alpha=0.2   
--json_list=./jsons/brats20_folds.json    
--data_dir=/opt/data/share/MICCAI_BraTS2020_TrainingData                      
--pretrainedT_dir ./pretraineded_models
--pretrained_model_name model_final.pt                                                   
--logdir=2020/Unet
--CUDA_VISIBLE_DEVICES 0,1,2    

```

