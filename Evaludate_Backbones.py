import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

from snn_Backbone import SNN
from ResNet_Backbone import ResNetBackbone

# Hyperparameters
batch_size = 16
n_steps = 50

# Data
train_data_path = ".\\data\\training"
val_data_path = ".\\data\\validation"

# Data transforming for SNN
transform_snn = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Data transforming for ResNet
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

def load_images_from_folder(folder, transform):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        img = transform(img)
        images.append(img)
    return torch.stack(images)

train_images_snn = load_images_from_folder(train_data_path, transform_snn)
val_images_snn = load_images_from_folder(val_data_path, transform_snn)

train_images_resnet = load_images_from_folder(train_data_path, transform_resnet)
val_images_resnet = load_images_from_folder(val_data_path, transform_resnet)

train_loader_snn = torch.utils.data.DataLoader(train_images_snn, batch_size=batch_size, shuffle=True)
val_loader_snn = torch.utils.data.DataLoader(val_images_snn, batch_size=batch_size, shuffle=False)

train_loader_resnet = torch.utils.data.DataLoader(train_images_resnet, batch_size=batch_size, shuffle=True)
val_loader_resnet = torch.utils.data.DataLoader(val_images_resnet, batch_size=batch_size, shuffle=False)

# Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
snn_model = SNN(n_steps=n_steps).to(device)
resnet_model = ResNetBackbone().to(device)

# Visualize feature maps and stats
def visualize_feature_maps(feature_maps, title):
    feature_maps = feature_maps.cpu().numpy()
    num_features = feature_maps.shape[1]

    print(f"Feature maps shape: {feature_maps.shape}")
    print(f"Feature maps min: {feature_maps.min()}, max: {feature_maps.max()}, mean: {feature_maps.mean()}")
    print(f"Feature maps std: {feature_maps.std()}")

    feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min() + 1e-8)

    plt.figure(figsize=(20, 10))
    num_to_show = min(num_features, 32)
    for i in range(num_to_show):
        plt.subplot(4, 8, i + 1)
        plt.imshow(feature_maps[0, i, :, :], cmap='viridis')
        plt.title(f'Feature Map {i + 1}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()
#SNN spikes and membrane potetial
def visualize_spike_statistics(spike_rec):
    spikes = spike_rec.cpu().numpy()
    num_steps = spikes.shape[0]

    spike_counts = spikes.sum(axis=(1, 2, 3))
    spike_mean = spike_counts.mean()
    spike_max = spike_counts.max()
    spike_min = spike_counts.min()

    print(f"Spike counts min: {spike_min}, max: {spike_max}, mean: {spike_mean}")

    plt.figure(figsize=(12, 6))
    plt.plot(range(num_steps), spike_counts, label='Spike Counts')
    plt.xlabel('Time Step')
    plt.ylabel('Spike Count')
    plt.title('Spike Counts Over Time')
    plt.legend()
    plt.show()

def visualize_membrane_potential(mem_rec):
    mem_potentials = mem_rec.cpu().numpy()
    num_steps = mem_potentials.shape[0]

    mem_min = mem_potentials.min()
    mem_max = mem_potentials.max()
    mem_mean = mem_potentials.mean()
    mem_std = mem_potentials.std()

    print(f"Membrane potentials min: {mem_min}, max: {mem_max}, mean: {mem_mean}, std: {mem_std}")

    plt.figure(figsize=(12, 6))
    plt.plot(range(num_steps), mem_potentials.mean(axis=(1, 2, 3)), label='Mean Membrane Potential')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Membrane Potential')
    plt.title('Mean Membrane Potential Over Time')
    plt.legend()
    plt.show()
    
#Models inference time
def measure_inference_time(model, loader):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            _ = model(images)
    end_time = time.time()
    return end_time - start_time

def visualize_comparison(snn_feature_maps, resnet_feature_maps):
    snn_feature_maps = snn_feature_maps.cpu().numpy()
    resnet_feature_maps = resnet_feature_maps.cpu().numpy()

    snn_feature_maps = (snn_feature_maps - np.min(snn_feature_maps)) / (np.max(snn_feature_maps) - np.min(snn_feature_maps) + 1e-8)
    resnet_feature_maps = (resnet_feature_maps - np.min(resnet_feature_maps)) / (np.max(resnet_feature_maps) - np.min(resnet_feature_maps) + 1e-8)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    num_to_show_snn = min(snn_feature_maps.shape[1], 16)
    for i in range(num_to_show_snn):
        plt.subplot(4, 4, i + 1)
        plt.imshow(snn_feature_maps[0, i, :, :], cmap='gray')
        plt.title(f'SNN FM {i + 1}')
        plt.axis('off')
    plt.suptitle('SNN Feature Maps')

    plt.subplot(1, 2, 2)
    num_to_show_resnet = min(resnet_feature_maps.shape[1], 16)
    for i in range(num_to_show_resnet):
        plt.subplot(4, 4, i + 1)
        plt.imshow(resnet_feature_maps[0, i, :, :], cmap='gray')
        plt.title(f'ResNet FM {i + 1}')
        plt.axis('off')
    plt.suptitle('ResNet Feature Maps')

    plt.show()

# Evaluate SNN
snn_model.eval()
with torch.no_grad():
    for images in val_loader_snn:
        images = images.to(device)
        spk_rec, mem_rec = snn_model(images)
        feature_maps_snn = mem_rec[-1]
        visualize_feature_maps(feature_maps_snn, "SNN Feature Maps")
        visualize_spike_statistics(spk_rec)
        visualize_membrane_potential(mem_rec)
        break

# Measure inference time for SNN
snn_time = measure_inference_time(snn_model, val_loader_snn)
print(f'SNN Inference Time: {snn_time:.4f} seconds')

# Evaluate ResNet
resnet_model.eval()
with torch.no_grad():
    for images in val_loader_resnet:
        images = images.to(device)
        feature_maps_resnet = resnet_model(images)
        print("ResNet feature maps shape:", feature_maps_resnet.shape)
        if len(feature_maps_resnet.shape) == 4:
            visualize_feature_maps(feature_maps_resnet, "ResNet Feature Maps")
        else:
            print("Unexpected feature map dimensions for ResNet")
        break

# Measure inference time for ResNet
resnet_time = measure_inference_time(resnet_model, val_loader_resnet)
print(f'ResNet Inference Time: {resnet_time:.4f} seconds')

# Compare feature maps from both models
visualize_comparison(feature_maps_snn, feature_maps_resnet)
