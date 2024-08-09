# =============================================================================
# Digital Semantic Communication System Presented by Shuoyao Wang and Mingze Gong, Shenzhen University. 
# =============================================================================
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm
import os
import random
import network
import modules
from solver import training_solver
# =============================================================================
# Randon seed setup
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
# =============================================================================
# BASIC CONFIGURATION
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_batch_size = 16
val_batch_size = 4
save_path = "./results/"
# best_psnr = 20 #<- THIS will be reset once a training-stage is done
# =============================================================================
# Training cost for all stages
JZfinetune_epoch = 15
DECfinetune_alpha_epoch = 60 # <- UNet and mask attack are introduced HERE
DECfinetune_alpha_epoch = 15 # <- mask attack are introduced HERE
DECfinetune_beta_epoch = 15
ENCfinetune_epoch = 15
DECunleash_epoch = 15
ENCunleash_epoch = 15
# =============================================================================
# Training&Validation datasets loading
transform_train = transforms.Compose([
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    ])
train_data_root = "./datasets/ImageNetVal"

trainset = modules.CustomDataset(train_data_root, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=8)
# =============================================================================
# Model setup
bits = 4
print ("Training bits: %d" % (bits))
model = network.DigitalSemComm_GC(bits=bits).to(device)

# NOTICE: assert we have finished the pretraining stage, HERE we load the pretrained model for finetuning
ckp_path = "./results/DSC_best_model_pretrained.pth"
msg = model.load_state_dict(torch.load(ckp_path, map_location='cpu'), strict=False)
print (msg)
model.to(device)
print ("Loading pretrained model...")
# =============================================================================
# Finetuning with J.Zhang 's approximated quantization
lr = 1e-5
min_lr = 1e-6
stage_name = "JZfinetune"
model_path = save_path + "DSC_best_model_JZfinetune_" + str(bits) + "bits.pth"
print ("Starting to finetune model with Jun Zhang's approximated quantization...")
training_solver(model, trainloader, lr, min_lr, JZfinetune_epoch, stage_name, model_path)
# =============================================================================
# Finetuning actor (raw UNet+decoder+mask atk)
# Reload model
print ("Reloading model from last stage...")
ckp_path = save_path + "DSC_best_model_JZfinetune_" + str(bits) + "bits.pth"
msg = model.load_state_dict(torch.load(ckp_path, map_location='cpu'), strict=False)
print (msg)
model.to(device)
lr = 1e-4
min_lr = 1e-6
stage_name = "DECfinetune"
model_path = save_path + "DSC_best_model_DECfinetune_1st_" + str(bits) + "bits.pth"
print ("Starting to finetune actor with raw U-Net (1st) and Mask ATK...")
training_solver(model, trainloader, lr, min_lr, DECfinetune_alpha_epoch, stage_name, model_path)
# =============================================================================
# Finetune environment (encoder+mask atk)
# Reload model
print ("Reloading model from last stage...")
ckp_path = save_path + "DSC_best_model_DECfinetune_1st_" + str(bits) + "bits.pth"
msg = model.load_state_dict(torch.load(ckp_path, map_location='cpu'), strict=False)
print (msg)
model.to(device)
lr = 1e-4
min_lr = 1e-6
stage_name = "ENCfinetune"
model_path = save_path + "DSC_best_model_ENCfinetune_1st_" + str(bits) + "bits.pth"
print ("Starting to finetune environment (1st) with Mask ATK...")
training_solver(model, trainloader, lr, min_lr, ENCfinetune_epoch, stage_name, model_path)
# =============================================================================
# Finetune actor (UNet+decoder+mask atk)
# Reload model
print ("Reloading model from last stage...")
ckp_path = save_path + "DSC_best_model_ENCfinetune_1st_" + str(bits) + "bits.pth"
msg = model.load_state_dict(torch.load(ckp_path, map_location='cpu'), strict=False)
print (msg)
model.to(device)
lr = 1e-4
min_lr = 1e-6
stage_name = "DECfinetune"
model_path = save_path + "DSC_best_model_DECfinetune_2nd_" + str(bits) + "bits.pth"
print ("Starting to finetune actor (2nd) with Mask ATK...")
training_solver(model, trainloader, lr, min_lr, DECfinetune_beta_epoch, stage_name, model_path)
# =============================================================================
# Finetune environment (encoder+mask atk)
# Reload model
print ("Reloading model from last stage...")
ckp_path = save_path + "DSC_best_model_DECfinetune_2nd_" + str(bits) + "bits.pth"
msg = model.load_state_dict(torch.load(ckp_path, map_location='cpu'), strict=False)
print (msg)
model.to(device)
lr = 1e-4
min_lr = 1e-6
stage_name = "ENCfinetune"
model_path = save_path + "DSC_best_model_ENCfinetune_2nd_" + str(bits) + "bits.pth"
print ("Starting to finetune environment (2nd) with Mask ATK...")
training_solver(model, trainloader, lr, min_lr, ENCfinetune_epoch, stage_name, model_path)
# =============================================================================
# Finetune actor (UNet+decoder)
# Reload model
print ("Reloading model from last stage...")
ckp_path = save_path + "DSC_best_model_ENCfinetune_2nd_" + str(bits) + "bits.pth"
msg = model.load_state_dict(torch.load(ckp_path, map_location='cpu'), strict=False)
print (msg)
model.to(device)
lr = 1.5e-5
min_lr = 1e-6
stage_name = "DECfinetune"
model_path = save_path + "DSC_best_model_DECfinetune_3rd_" + str(bits) + "bits.pth"
print ("Starting to finetune actor (3rd) with Mask ATK...")
training_solver(model, trainloader, lr, min_lr, DECfinetune_beta_epoch, stage_name, model_path)
# =============================================================================
# Finetune environment (encoder+mask atk)
# Reload model
print ("Reloading model from last stage...")
ckp_path = save_path + "DSC_best_model_DECfinetune_3rd_" + str(bits) + "bits.pth"
msg = model.load_state_dict(torch.load(ckp_path, map_location='cpu'), strict=False)
print (msg)
model.to(device)
lr = 1.5e-5
min_lr = 1e-6
stage_name = "ENCfinetune"
model_path = save_path + "DSC_best_model_ENCfinetune_3rd_" + str(bits) + "bits.pth"
print ("Starting to finetune environment (3rd) with Mask ATK...")
training_solver(model, trainloader, lr, min_lr, ENCfinetune_epoch, stage_name, model_path)
# =============================================================================
# Finetune actor (UNet+decoder)
# Reload model
print ("Reloading model from last stage...")
ckp_path = save_path + "DSC_best_model_ENCfinetune_3rd_" + str(bits) + "bits.pth"
msg = model.load_state_dict(torch.load(ckp_path, map_location='cpu'), strict=False)
print (msg)
model.to(device)
lr = 1.5e-5
min_lr = 1e-6
stage_name = "DECunleash"
model_path = save_path + "DSC_best_model_DECunleash_" + str(bits) + "bits.pth"
print ("Starting to finetune actor (4th)...")
training_solver(model, trainloader, lr, min_lr, DECunleash_epoch, stage_name, model_path)
# =============================================================================
print ("ALL TRAINING PROCEDURES ARE OVER !!!")