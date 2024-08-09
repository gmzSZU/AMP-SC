# =============================================================================
# Digital Semantic Communication System Presented by Shuoyao Wang and Mingze Gong, Shenzhen University. 
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as tag
import numpy as np
import math
from modules import conv_block, conv_ResBlock, deconv_ResBlock, M2RModule_Res, SE_block

class DigitalSemComm_GC(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.action_scale=(2**bits)/2.
        self.action_bias=0.5       
        self.quant_lb = -2**(bits-1)+1
        self.quant_ub = 2**(bits-1)
        self.quant_range = [i for i in range(self.quant_lb, self.quant_ub+1, 1)]
        # Encoder
        self.conv1 = conv_ResBlock(3, 256, use_conv1x1=True, kernel_size=5, stride=2, padding=2)
        self.SE1 = SE_block(256, 128, 256)
        self.num_dowm_sample = 3
        self.conv2 = nn.ModuleList()
        for i in range(self.num_dowm_sample):
            self.conv2.append(nn.Sequential(conv_ResBlock(256, 256, use_conv1x1=True, kernel_size=5, stride=2, padding=2),
                                            SE_block(256, 128, 256)))
        self.num_deep_extraction = 3
        self.conv3 = nn.ModuleList()
        for i in range(self.num_deep_extraction):
            self.conv3.append(nn.Sequential(conv_ResBlock(256, 256, kernel_size=5, stride=1, padding=2),
                                            SE_block(256, 128, 256)))
        self.conv4 = conv_ResBlock(256, 32, use_conv1x1=True, kernel_size=5, stride=1, padding=2)

        # Decoder
        self.up_conv1 = deconv_ResBlock(32, 256, use_deconv1x1=True, kernel_size=5, stride=1, padding=2)
        self.num_up_sample = 3
        self.up_conv3 = nn.ModuleList()
        for i in range(self.num_up_sample):
            self.up_conv3.append(nn.Sequential(SE_block(256, 128, 256),
                                               deconv_ResBlock(256, 256, use_deconv1x1=True, kernel_size=5, stride=2, padding=2, output_padding=1)))
        self.num_deep_denoise = 3
        self.up_conv2 = nn.ModuleList()
        for i in range(self.num_deep_denoise):
            self.up_conv2.append(nn.Sequential(SE_block(256, 128, 256),
                                               deconv_ResBlock(256, 256, kernel_size=5, stride=1, padding=2)))
        self.up_SE = SE_block(256, 128, 256)
        self.up_conv4 = deconv_ResBlock(256, 3, use_deconv1x1=True, kernel_size=5, stride=2, padding=2, output_padding=1)

        # Digital data stream transmission
        self.bit_flip_ratio = 0.0125

        # Pseudo Masked Decoder
        self.mask_ratio = 2*self.bit_flip_ratio
        self.up_pmd = M2RModule_Res(channel=32)
        self.l1 = nn.L1Loss()
        
    
    def tensor_to_binary_tensor(self, tensor):
        # Ensure the input tensor has the desired data type (integer)
        tensor = tensor.to(dtype=torch.int) + (2**(self.bits-1) - 1)
        # Calculate the maximum value a single element can have based on L
        max_value = 2**self.bits - 1
        if torch.any(tensor < 0):
            tensor[torch.where(tensor < 0)]=0
        # Check if any element in the tensor is out of range
        if torch.any((tensor < 0) | (tensor > max_value)):
            raise ValueError("Input tensor contains elements out of range.")
        # Create a binary mask for shifting and extracting each bit
        bit_mask = torch.tensor([2**i for i in range(self.bits-1, -1, -1)], dtype=torch.int, device=tensor.device)
        # Create a binary tensor by bitwise AND operation with the bit mask
        binary_tensor = (tensor.unsqueeze(-1) & bit_mask).gt(0).to(dtype=torch.int)
        return binary_tensor

    def binary_tensor_to_tensor(self, binary_tensor):
        # Ensure the input tensor has the desired data type (integer)
        binary_tensor = binary_tensor.to(dtype=torch.int)
        # Calculate the number of bits (L)
        #L = binary_tensor.shape[-1]
        L = self.bits
        # Convert binary strings to integers for each batch element
        tensor = torch.sum(binary_tensor * 2**(torch.arange(L-1, -1, step=-1, dtype=torch.int, device=binary_tensor.device)), dim=-1)
        return tensor-2**(self.bits-1) + 1

    def bit_flip(self,x,p):
        random_array = torch.rand(x.shape)
        #print(random_array)
        flipped_indices = random_array < p
        x[flipped_indices] = 1 - x[flipped_indices]
        return x

    def mask_generate(self, x):
        B, C = x.shape
        cache = torch.rand(B, C)
        mask = cache > self.mask_ratio
        return mask.to(x.device)
    
    def Encoding(self, x):
        x = self.SE1(self.conv1(x))
        for i in range(self.num_dowm_sample):
            x = self.conv2[i](x)
        for i in range(self.num_deep_extraction):
            x = self.conv3[i](x)
        x = self.conv4(x)        
        Tx = (torch.tanh(x)* self.action_scale + self.action_bias).view(x.shape[0], -1)
        return Tx
    
    def Decoding(self, Rx):
        B = Rx.shape[0]
        Rx = Rx.view(B, 32, 16, 16)
        Rx = self.up_pmd(Rx)
        Rx = self.up_conv1(Rx)
        for i in range(self.num_up_sample):
            Rx = self.up_conv3[i](Rx)
        for i in range(self.num_deep_denoise):
            Rx = self.up_conv2[i](Rx)
        Rx = self.up_SE(Rx)
        res = self.up_conv4(Rx, 'sigmoid')
        return res
        
    def Decoding_MASK(self, Tx, Rx, mask_flag=True):
        B = Rx.shape[0]
        if mask_flag:
            # Masking
            mask = self.mask_generate(Rx)
            Rx = mask*Rx
        Rx = Rx.view(B, 32, 16, 16)
        Tx = Tx.view(B, 32, 16, 16)
        Rx = self.up_pmd(Rx)
        l1_loss = self.l1(Rx, Tx)
        Rx = self.up_conv1(Rx)
        for i in range(self.num_up_sample):
            Rx = self.up_conv3[i](Rx)
        for i in range(self.num_deep_denoise):
            Rx = self.up_conv2[i](Rx)
        Rx = self.up_SE(Rx)
        res = self.up_conv4(Rx, 'sigmoid')
        return res, l1_loss
    
    def forward_digital_TEST(self, x, mask_flag=False):
        Tx = self.Encoding(x) # <- Encoding
        Tx = torch.round(Tx) # <- Quantization
        # Transmission
        Rx = self.tensor_to_binary_tensor(Tx)
        Rx = self.bit_flip(Rx, self.bit_flip_ratio)
        Rx = self.binary_tensor_to_tensor(Rx).to(torch.float)
        ## Actor 
        res, _ = self.Decoding_MASK(Tx, Rx, mask_flag=mask_flag) # Decoding
        return res

    def Decoding_naive(self, Rx):
        B = Rx.shape[0]
        Rx = Rx.view(B, 32, 16, 16)
        Rx = self.up_conv1(Rx)
        for i in range(self.num_up_sample):
            Rx = self.up_conv3[i](Rx)
        for i in range(self.num_deep_denoise):
            Rx = self.up_conv2[i](Rx)
        Rx = self.up_SE(Rx)
        res = self.up_conv4(Rx, 'sigmoid')
        return res

    # def forward(self, x):
    #     # THIS shall be used for pretraining
    #     # Encoding
    #     Tx = self.Encoding(x)
    #     # Transmission
    #     Rx = Tx
    #     # Decoding
    #     res = self.Decoding_naive(Rx)
    #     return res
    
