import torch
import pandas as pd
import json
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler
from collections import defaultdict
from transformers import BertModel, BertTokenizer
from HARmodel import HARModel, handle_argv

# ------------------------------------ Configuration ------------------------------------
CONFIG_PATH = '/root/autodl-fs/HAR-Genie/dataset/data_config.json'
DATASETS = ['HHAR', 'Pamap2', 'Realworld', 'UCI-HAR', 'USC-HAD']
LABEL_MAP = ["cycling", "sitting", "downstairs", "upstairs", "standing", "walking", "lying", "running", "jumping"]
SENSOR_LOCATIONS = {
    "HHAR": ["waist", "wrist"],
    "Pamap2": ["chest", "wrist", "ankle"],
    "Realworld": ["chest", "waist", "ankle", "hip", "wrist"],
    "UCI-HAR": ["waist"],
    "USC-HAD": ["hip"]
}
CATEGORY_GROUPS = {
    0: [2, 3],            
    1: [1, 4, 6],        
    2: [5, 7, 8],        
    3: [0]               
}

with open(CONFIG_PATH, 'r') as f:
    configs = json.load(f)



class MultiDatasetLoader(Dataset):
    def __init__(self, dataset_paths, label_map, mode='train', standardization=True):
        self.samples = []
        self.label_map = label_map
        self.standardization = standardization
        self.stats_by_location = defaultdict(dict) 

        location_samples = defaultdict(list)

        for dataset_name in dataset_paths:

            data_path = f'/root/autodl-fs/HAR-Genie/{mode}/{dataset_name}/data_50_120.npy'
            label_path = f'/root/autodl-fs/HAR-Genie/{mode}/{dataset_name}/label_50_120.npy'

            imu_data = np.load(data_path)
            config_name = f"{dataset_name}_50_120"
            labels = np.load(label_path)[:, :, configs[config_name]["activity_label_index"]]

            sensor_locations = SENSOR_LOCATIONS[dataset_name]

            for i in range(len(imu_data)):
                for position_id, position_name in enumerate(sensor_locations):
                    sensor_data = imu_data[i, position_id, :, :]
                    location_samples[position_name].append(sensor_data)

                sample = {
                    'imu_data': imu_data[i],     
                    'label': labels[i, 0],
                    'N': imu_data[i].shape[0],
                    'dataset_name': dataset_name
                }
                self.samples.append(sample)

        for position_name, samples in location_samples.items():
            samples = np.concatenate(samples)
            if self.standardization:
                mean = samples.mean(axis=(0,1), keepdims=True)
                std = samples.std(axis=(0,1), keepdims=True)
                self.stats_by_location[position_name]["mean"] = mean
                self.stats_by_location[position_name]["std"] = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imu_sample = sample['imu_data']      
        label = sample['label']
        dataset_name = sample['dataset_name']

        sensor_locations = SENSOR_LOCATIONS[dataset_name]

        if self.standardization:
            for position_id, position_name in enumerate(sensor_locations):
                if position_name in self.stats_by_location:
                    mean = self.stats_by_location[position_name]["mean"]   
                    std = self.stats_by_location[position_name]["std"]     
                    imu_sample[position_id] = (imu_sample[position_id] - mean) / std
                else:
                    pass

        imu_sample = torch.tensor(imu_sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        dataset_label = torch.tensor(DATASETS.index(dataset_name), dtype=torch.long) 
        N = sample['N']
        return imu_sample, label, dataset_label, N




class SensorCountBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.grouped_indices = self._group_indices_by_N()

    def _group_indices_by_N(self):
        groups = defaultdict(list)
        for idx, sample in enumerate(self.dataset.samples):
            N = sample['N']
            groups[N].append(idx)
        return groups

    def __iter__(self):
        all_batches = []
        for N, indices in self.grouped_indices.items():
            if self.shuffle:
                np.random.shuffle(indices)
            batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
            all_batches.extend(batches)
        if self.shuffle:
            np.random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        total = 0
        for indices in self.grouped_indices.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total




class HARModelWrapper(nn.Module):
    def __init__(self, har_model, args, out_features=768):
        super().__init__()
        self.har_model = har_model
        self.args = args
        self.out_features = out_features

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 3), padding=0)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 3), padding=0)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), padding=0)

        self.final_height = 120 - 3 * (3 - 1)
        self.final_width = 72 - 3 * (3 - 1)

        self.fc1 = nn.Linear(self.final_height * self.final_width, self.out_features)

    def forward(self, imu_data, args=None):
        x = self.har_model(imu_data, self.args) 
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.conv3(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc1(x) 
        return x





class LabelEmbeddingEncoder(nn.Module):
    def __init__(self, label_map):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.label_map = label_map
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, labels):
        label_texts = [self.label_map[label.item()] for label in labels]
        tokenized_labels = self.tokenizer(label_texts, return_tensors='pt', padding=True, truncation=True).to(labels.device)
        return self.bert(**tokenized_labels).last_hidden_state.mean(dim=1)




class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, num_domains):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)




class ContrastiveModelWithAdversarial(nn.Module):
    def __init__(self, imu_encoder, label_encoder, num_domains, feature_dim, temperature=0.07):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.label_encoder = label_encoder
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.domain_discriminator = DomainDiscriminator(input_dim=feature_dim, num_domains=num_domains)
        self.domain_criterion = nn.CrossEntropyLoss()
    

    def evaluate(self, imu_data, labels, args):
        self.eval()
        with torch.no_grad():
            imu_features = self.imu_encoder(imu_data, args)
            imu_features = nn.functional.normalize(imu_features, dim=-1)
            label_embeddings = self.label_encoder(torch.arange(len(self.label_encoder.label_map)).to(imu_data.device))
            label_embeddings = nn.functional.normalize(label_embeddings, dim=-1)
            logits = torch.matmul(imu_features, label_embeddings.T)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels.squeeze(-1)).float().mean().item()
            return accuracy, predictions


# ------------------------------------ Train and Test ------------------------------------


def test(model, test_loader, args):

    total_accuracy = 0
    model.eval()
    results = []
    with torch.no_grad():
        i = 0 
        for imu_data, labels, dataset_labels, N in test_loader:
            imu_data, labels, dataset_labels = imu_data.to(device), labels.to(device), dataset_labels.to(device)
            accuracy, predictions = model.evaluate(imu_data, labels, args)
            i = i + 1
            results.append({
                # 'id': i,
                'prediction': accuracy,
                'dataset': [DATASETS[i] for i in dataset_labels]
                'answer': [LABEL_MAP[i] for i in predictions],
                'label': [LABEL_MAP[i] for i in labels]
            })
            total_accuracy += accuracy
    results_df = pd.DataFrame(results)
    results_path = '/input/your/path.csv'
    results_df.to_csv(results_path, index=False)
    return total_accuracy / len(test_loader)




def initialize_model(args):
    har_model = HARModel(args=args)
    imu_encoder = HARModelWrapper(har_model, args)
    label_encoder = LabelEmbeddingEncoder(LABEL_MAP)
    model = ContrastiveModelWithAdversarial(imu_encoder, label_encoder, num_domains=len(DATASETS), feature_dim=768)
    return model

# ------------------------------------ Main Execution ------------------------------------
if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = MultiDatasetLoader(DATASETS, LABEL_MAP, mode='test')
    test_sampler = SensorCountBatchSampler(test_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    model = initialize_model(args).to(device)
    model_dict = model.state_dict() 
    
    model_path = '/input/your/path/uniHAR.pth'
    checkpoint = torch.load(model_path)

    checkpoint = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and v.shape == model_dict[k].shape}

    model_dict.update(checkpoint) 
    model.load_state_dict(model_dict)

    test(model, test_loader, args)


