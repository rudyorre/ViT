from torchvision import transforms
from vit import ViT
import torch
from torch import nn
from tqdm import tqdm
from einops import rearrange, repeat

def main():
    #Define the model, optimizer, and criterion (loss_fn)
    model = ViT(
        image_size=128,
        patch_size=16,
        num_classes=100,
        dim=192,
        depth=8,
        heads=4,
        dim_head=48,
        mlp_dim=768,
        dropout=0.1,
        emb_dropout=0.1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,)
    criterion = nn.CrossEntropyLoss()

    # Define the dataset and data transform with flatten functions appended
    data_root = os.path.join('./Assignment3', 'data')
    train_dataset = MiniPlaces(
        root_dir=data_root,
        split='train', 
        transform=data_transform
    )

    val_dataset = MiniPlaces(
        root_dir=data_root,
        split='val', 
        transform=data_transform,
        label_dict=train_dataset.label_dict
    )

    # Define the batch size and number of workers
    batch_size = 64
    num_workers = 2

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    #for data in train_loader:
    #    print(1)
    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=2)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transformation
# You can copy your data transform from Assignment2. 
# Notice we are resize images to 128x128 instead of 64x64.
data_transform = transforms.Compose([
    ################# Your Implementations #####################################
    # TODO: Resize image to 128x128
    transforms.Resize((128, 128)),
    ################# End of your Implementations ##############################
    transforms.ToTensor(),
    ################# Your Implementations #####################################
    # TODO: Normalize image using ImageNet statistics
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ################# End of your Implementations ##############################
])

# You can copy your dataset from Assignment2. 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MiniPlaces(Dataset):
    def __init__(self, root_dir, split, transform=None, label_dict=None):
        """
        Initialize the MiniPlaces dataset with the root directory for the images, 
        the split (train/val/test), an optional data transformation, 
        and an optional label dictionary.
        
        Args:
            root_dir (str): Root directory for the MiniPlaces images.
            split (str): Split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional data transformation to apply to the images.
            label_dict (dict, optional): Optional dictionary mapping integer labels to class names.
        """
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.filenames = []
        self.labels = []

        # Take a second to think why we need this line.
        # Hints: training set / validation set / test set.
        self.label_dict = label_dict if label_dict is not None else {}

        # You should
        #   1. Load the train/val text file based on the `split` argument and
        #     store the image filenames and labels.
        #   2. Extract the class names from the image filenames and store them in 
        #     self.label_dict.
        #   3. Construct a label dict that maps integer labels to class names, if 
        #     the current split is "train" 
        ################# Your Implementations #####################################
         # Test set
        if split == 'test':
            for file in os.listdir(os.path.join(root_dir, 'images', split)):
              self.filenames.append(file[:-4])
            self.filenames.sort()
            return

        # Train/Val sets
        with open(os.path.join(root_dir, split + '.txt')) as open_file:
            for line in open_file:
                filename, label = line.strip().split()
                self.filenames.append(filename)
                self.labels.append(int(label))
                if split == 'train':
                    self.label_dict[int(label)] = filename.split('/')[2]
        ################# End of your Implementations ##############################

    def __len__(self):
        """
        Return the number of images in the dataset.
        
        Returns:
            int: Number of images in the dataset.
        """
        ################# Your Implementations #####################################
        # Return the number of images in the dataset
        return len(self.filenames)
        ################# End of your Implementations ##############################

    def __getitem__(self, idx):
        """
        Return a single image and its corresponding label when given an index.
        
        Args:
            idx (int): Index of the image to retrieve.
            
        Returns:
            tuple: Tuple containing the image and its label.
        """
        image = None
        label = None
        ################# Your Implementations #####################################
        # Load and preprocess image using self.root_dir, 
        # self.filenames[idx], and self.transform (if specified)
        filename = self.filenames[idx]
        label = self.labels[idx]

        if self.split == 'test':
            img_path = os.path.join(self.root_dir, 'images', 'test', filename + '.jpg')
            return self.transform(Image.open(img_path)), filename
        
        img_path = os.path.join(self.root_dir, 'images', filename)
        image = Image.open(img_path)
        image = self.transform(image)
        ################# End of your Implementations ##############################
        return image, label
    
def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    """
    Train the MLP classifier on the training set and evaluate it on the validation set every epoch.
    
    Args:
        model (MLP): MLP classifier to train.
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        criterion (callable): Loss function to use for training.
        device (torch.device): Device to use for training.
        num_epochs (int): Number of epochs to train the model.
    """
    # Place model on device
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        # Use tqdm to display a progress bar during training
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for inputs, labels in train_loader:
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero out gradients
                optimizer.zero_grad()
                
                # Compute the logits and loss
                logits = model(inputs)
                loss = criterion(logits, labels)
                
                # Backpropagate the loss
                loss.backward()
                
                # Update the weights
                optimizer.step()
                
                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        # Evaluate the model on the validation set
        avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
        print(f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the MLP classifier on the test set.
    
    Args:
        model (MLP): MLP classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        criterion (callable): Loss function to use for evaluation.
        device (torch.device): Device to use for evaluation.
        
    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0
        
        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Compute the logits and loss
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Compute the accuracy
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)
            
    # Compute the average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples
    
    return avg_loss, accuracy

if __name__ == '__main__':
    main()