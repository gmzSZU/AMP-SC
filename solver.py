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
import modules

def training_solver(model, trainloader, lr, min_lr,
                    total_epoch, stage_name, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=min_lr)
    num_iters_train = len(trainloader)
    best_psnr = 10
    best_epoch = 0
    for epoch in range(total_epoch):
        model.train()
        total_loss_epoch = 0
        total_psnr_epoch = 0
        DF_loss_epoch = 0
        print ("\n")
        print ("Epoch: %d, BEST Epoch: %d" % (epoch, best_epoch))
        print ("Current Learning Rate: " + str(optimizer.state_dict()['param_groups'][0]['lr']))
        for imgs in tqdm(trainloader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            if stage_name == "JZfinetune":
                DF_loss = 0
                Tx = model.Encoding(imgs) # <- Encoding
                Rx = Tx+0.5*torch.randn_like(Tx)     
                res_recovery = model.Decoding_naive(Rx)
            elif stage_name == "DECfinetune":
                with torch.no_grad():
                    Tx = model.Encoding(imgs) # <- Encoding
                    Tx = torch.round(Tx) # <- Quantization
                    # Transmission
                    Rx = model.tensor_to_binary_tensor(Tx)
                    Rx = model.bit_flip(Rx, model.bit_flip_ratio)
                    Rx = model.binary_tensor_to_tensor(Rx).to(torch.float)
                res_recovery, DF_loss = model.Decoding_MASK(Tx, Rx, mask_flag=True) # Decoding with mask attack
            elif stage_name == "ENCfinetune":
                DF_loss = 0
                Tx = model.Encoding(imgs) # <- Encoding
                Rx = Tx+0.5*torch.randn_like(Tx)             
                res_recovery, _ = model.Decoding_MASK(Tx, Rx, mask_flag=True) # Decoding with mask attack
            elif stage_name == "DECunleash":
                with torch.no_grad():
                    Tx = model.Encoding(imgs) # <- Encoding
                    Tx = torch.round(Tx) # <- Quantization
                    # Transmission
                    Rx = model.tensor_to_binary_tensor(Tx)
                    Rx = model.bit_flip(Rx, model.bit_flip_ratio)
                    Rx = model.binary_tensor_to_tensor(Rx).to(torch.float)
                res_recovery, DF_loss = model.Decoding_MASK(Tx, Rx, mask_flag=False) # Decoding with mask attack
            else:
                raise ValueError("Please check the name of stage!")
                
            loss = loss_fn(res_recovery, imgs)
            if DF_loss != 0:
                loss += DF_loss
            loss.backward()
            optimizer.step()
            
            psnr_avg = modules.Compute_batch_PSNR(imgs.cpu().detach().numpy(), res_recovery.cpu().detach().numpy())
            total_psnr_epoch += psnr_avg
            total_loss_epoch += loss.item()
            DF_loss_epoch += DF_loss
        scheduler.step()
        print("Epoch: %d || Avg Train Loss: %.05f || AVG PSNR: %.05f || AVG Restoration Loss: %.05f" % (epoch, total_loss_epoch/num_iters_train, total_psnr_epoch/num_iters_train, DF_loss_epoch/num_iters_train)) 
        
        if total_psnr_epoch/num_iters_train > best_psnr:
            best_psnr = total_psnr_epoch/num_iters_train
            best_epoch = epoch
            print ("New Record Confirm, Saving Model...")
            torch.save(model.state_dict(), save_path)
                    
    print("Training process for Digital SemComm System in this stage is OVER.")
    print ("Sub-optimal PSNR: %.5f" % best_psnr)
    print ("Corresponding Epoch: %d" % best_epoch)
    print ("\n")
    return

