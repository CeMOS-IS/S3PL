import os
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import tqdm
import m2aia as m2

from model.Attention3DConvAutoencoder import Attention3DConvAutoencoder
from test import test
from utils.helpers import tic_norm_spectra

def train(config):   
    random_seed = config["random_seed"]
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    use_cuda = torch.cuda.is_available()

    data_dir = config["data_dir"]
    dataname = data_dir.split('/')[-1].replace('.imzML', '').replace('.npy', '')
    directory_name = os.path.dirname(__file__)
    folderpath = directory_name + data_dir.split(dataname + '.imzML')[0]
    filepath = directory_name + data_dir

    I = m2.ImzMLReader(filepath)
    mz_list = I.GetXAxis()
    image_handles=[I]
    num_input_channels = len(mz_list)

    transform = transforms.Lambda(lambda x: torch.tensor(tic_norm_spectra(x), dtype=torch.float32))
    training_dataset = m2.SpectrumDataset(image_handles, shape=(config["spectral_patch_size"], config["spectral_patch_size"]), buffer_type="memory", transform_data=transform)
    
    dataset_size = len(training_dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices = indices
    test_indices = indices

    train_sampler = SubsetRandomSampler(train_indices)
    training_loader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True, sampler=train_sampler)

    model = Attention3DConvAutoencoder(config["batch_size"], kernel_depth_d1=config["kernel_depth_d1"], kernel_depth_d2=config["kernel_depth_d2"], dropout=config["dropout"], spectral_patch_size=config["spectral_patch_size"])
    criterion = nn.MSELoss()

    if use_cuda:
        model = model.cuda()

    if not os.path.exists('weights'): os.mkdir('weights')
    if not os.path.exists('logs'): os.mkdir('logs')
    training_name = dataname + '_' + model._get_name() + '_' + str(config["n_epochs"]) + 'epochs_' + str(config["peaks_per_spectral_patch"]) + '_' + 'spectral_patch_size_' + str(config["spectral_patch_size"])
    path_to_weights = 'weights/' + training_name + '.pt'

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(path_to_weights)
    print('Samples in training folder: ' + str(len(training_dataset)))
    print('input channels = ' + str(num_input_channels))
    print('trainable parameters: ' + str(pytorch_total_params))
    print('GPU available: ' + str(use_cuda))
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])

    train_history = []
    for epoch in range(config["n_epochs"]):
        model.train()

        with tqdm.tqdm(training_loader, unit="batch") as bar:
            for batch in bar:
                bar.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                X_batch, _ = batch            
                y_batch = X_batch

                if use_cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

                y_pred, binary_mask, attention_mask = model(X_batch)

                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                bar.set_postfix_str(str(criterion._get_name()) + '=' + str(loss.item()))

        loss = loss.item()
        train_history.append(loss)
        
    torch.save(model.state_dict(), path_to_weights)

    config["training_name"] = training_name
    config["train_history"] = train_history
    
    with open('logs/' + training_name + '.json', 'w') as f:
        json.dump(config, f, indent=4)

    mSCF1 = test(config, test_indices)
    return mSCF1