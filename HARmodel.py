import torch
import torch.nn as nn
import numpy as np
from embedding import generate_embedding_or_output
from VITmodel import VisionTransformer
from utils import handle_argv


class HARModel(nn.Module):
    def __init__(self, args, feature_dim=72, num_layers=1):
        super(HARModel, self).__init__()
        self.feature_dim = feature_dim

        self.input_layer_norm = nn.LayerNorm([120, 3])
        self.limuIn_layer_norm = nn.LayerNorm([120, 3])
        self.output_layer_norm = nn.LayerNorm([120, 72])
        self.pos_embed = nn.Parameter(torch.zeros(1, 120, 3))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.vit_x = VisionTransformer(embed_dim=120)
        self.vit_y = VisionTransformer(embed_dim=120)
        self.vit_z = VisionTransformer(embed_dim=120)

        self.gru1 = nn.GRU(input_size=3, hidden_size=9, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=18, hidden_size=36, num_layers=num_layers, batch_first=True,  bidirectional=True)
        self.activation = nn.ReLU()

    def forward(self, imu_data, args):

        device = imu_data.device
        B, N, T, C = imu_data.shape

        
        self.input_layer_norm = self.input_layer_norm.to(device)
        self.limuIn_layer_norm = self.limuIn_layer_norm.to(device)
        self.output_layer_norm = self.output_layer_norm.to(device)

        
        embedded_positions = []
        for i in range(N):
            sensor_data = imu_data[:, i, :, :]  
            sensor_data = self.limuIn_layer_norm(sensor_data) 
            sensor_data = sensor_data + self.pos_embed
            sensor_data = self.limuIn_layer_norm(sensor_data)
            sensor_data = sensor_data.repeat(1, 1, 2)               


            _, embedding= generate_embedding_or_output(args, save=False, output_embed=True, sensor_data=sensor_data)
            embedding = torch.tensor(embedding, dtype=torch.float32).to(device) if isinstance(embedding, np.ndarray) else embedding.to(device)
            embedded_positions.append(embedding)
        
        embedded_positions = torch.stack(embedded_positions, dim=1).to(device)          


        weights = torch.arange(1, N + 1, dtype=torch.float32).to(device)                
        weights = weights.view(1, N, 1).expand(B, N, 1).to(device)       
        weighted_embedded_positions = embedded_positions * weights.unsqueeze(-1)
        weighted_average = weighted_embedded_positions.sum(dim=1, keepdim=True) / weights.sum()
        embedded_positions = weighted_average                                          
        

        imu_data = imu_data.view(-1, T, C)          
        imu_data = self.input_layer_norm(imu_data)  
        imu_data = imu_data.view(B, N, T, C)       

        Zx = imu_data[:, :, :, 0] 
        Zy = imu_data[:, :, :, 1]  
        Zz = imu_data[:, :, :, 2] 
        

        Zx_out = self.vit_x(Zx)  
        Zy_out = self.vit_y(Zy)  
        Zz_out = self.vit_z(Zz) 
        
        fused_imu = torch.cat([Zx_out, Zy_out, Zz_out], dim=1)        
        fused_imu = fused_imu.permute(0, 2, 1)               

        gru_out, _ = self.gru1(fused_imu)                   
        gru_out, _ = self.gru2(gru_out)
        vit_output = gru_out.unsqueeze(1)                

        
        output = torch.cat([embedded_positions, vit_output], dim=1) 
        output = output.view(-1, 120, 72)
        output = self.output_layer_norm(output)
        output = output.view(B, 2, 120, 72)


        return output
    






















#-------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    save = True
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)

    imu_data = np.load('/root/autodl-fs/HAR-Genie/train/USC-HAD/data_50_120.npy')
    imu_data = torch.tensor(imu_data, dtype=torch.float32)
    print(imu_data.shape)

    model = HARModel(args=args)
    output = model(imu_data, args)
    print(output.shape)
