import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch

import train
from models import LIMUBertModel4Pretrain
from utils import LIBERTDataset4Pretrain, load_pretrain_data_config, get_device, handle_argv, \
    Preprocess4Normalization, IMUDataset

def fetch_setup(args, output_embed, sensor_data=None):
    train_cfg, model_cfg = load_pretrain_data_config(args)
    
    if sensor_data is not None:
        data = sensor_data.cpu().detach().numpy() if sensor_data.is_cuda else sensor_data.detach().numpy()
        pipeline = [Preprocess4Normalization(6)] 
        data_set = IMUDataset(data, pipeline=pipeline)
        data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)

    model = LIMUBertModel4Pretrain(model_cfg, output_embed=output_embed)
    criterion = nn.MSELoss(reduction='none')
    return data, data_loader, model, criterion, train_cfg

def generate_embedding_or_output(args, save=False, output_embed=True, sensor_data=None):
    data, data_loader, model, criterion, train_cfg = fetch_setup(args, output_embed, sensor_data=sensor_data)
    optimizer = None
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_forward(model, batch):
        if isinstance(batch, list):
            seqs = torch.stack(batch)
        else:
            seqs = batch
        embed = model(seqs)
        return embed

    output = trainer.run(func_forward, None, data_loader, args.pretrain_model)

    if save:
        save_name = 'embed_' + args.model_file.split('.')[0] + '_' + args.dataset + '_' + args.dataset_version
        np.save(os.path.join('embed', save_name + '.npy'), output)
    return data, output


