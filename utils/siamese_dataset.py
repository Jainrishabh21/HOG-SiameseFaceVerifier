import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SiameseFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        
        self.persons = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        self.person_to_images = {
            person: [os.path.join(root_dir, person, img) 
                     for img in os.listdir(os.path.join(root_dir, person)) 
                     if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for person in self.persons
        }

    def __len__(self):
        # Approximate number of pairs
        # return 10000  # Or dynamically: min(total_images^2, N) ---  # can modify this if needed
        return sum(len(imgs) for imgs in self.person_to_images.values())

    def __getitem__(self, idx):
        should_get_same_class = random.randint(0, 1)

        person1 = random.choice(self.persons)
        img1_path = random.choice(self.person_to_images[person1])
        img1 = Image.open(img1_path).convert('L')

        if should_get_same_class and len(self.person_to_images[person1]) >= 2:
            # Positive pair
            img2_path = random.choice(
                [img for img in self.person_to_images[person1] if img != img1_path]
            )
            label = 1.0
        else:
            # Negative pair
            person2 = random.choice([p for p in self.persons if p != person1])
            img2_path = random.choice(self.person_to_images[person2])
            label = 0.0

        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
