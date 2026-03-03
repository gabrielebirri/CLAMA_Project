import torch
from torchvision import transforms
from src.utils import imagenet_mean, imagenet_std

# Let's define some useful transformations for data augmentation
# NOTE: the choosen transformations preserve the diagnostic features of the melanoma
torch.manual_seed(42)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )

])
