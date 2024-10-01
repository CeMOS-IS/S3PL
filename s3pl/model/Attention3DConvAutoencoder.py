import torch
import torch.nn as nn

class Attention3DConvAutoencoder(nn.Module):
    """ 
    Attention based 3D autoencoder for peak picking
    """
    
    def __init__(self, batchsize, kernel_depth_d1, kernel_depth_d2, dropout, spectral_patch_size):
        super(Attention3DConvAutoencoder, self).__init__()
        self.sigmoid = nn.Sigmoid()
        kernel_size_hw = spectral_patch_size
        self.batchsize = batchsize
        self.spectral_patch_size = spectral_patch_size
        self.center_index = int((spectral_patch_size-1)/2)

        self.mask_encoding = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(kernel_depth_d1, kernel_size_hw, kernel_size_hw), padding=(int((kernel_depth_d1-1)/2),0,0)),
            nn.Sigmoid()
        )

        self.mask_decoding = nn.Sequential(
            nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(kernel_depth_d2, kernel_size_hw, kernel_size_hw), padding=(int((kernel_depth_d2-1)/2),0,0)),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

    def forward(self, model_input):
        attention_mask = self.mask_encoding(model_input)
        mean_model_input = model_input[:,:,:,self.center_index,self.center_index].unsqueeze(-1).unsqueeze(-1)
        picked_peaks = torch.mul(mean_model_input, attention_mask)
        image_out = self.mask_decoding(picked_peaks)

        return image_out, picked_peaks, attention_mask
