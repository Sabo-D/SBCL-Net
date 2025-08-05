from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from src.utils.utils import *


class Train_val_dataset(Dataset):
    def __init__(self, image_dir, mask_dir, dist_dir, edge_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dist_dir = dist_dir
        self.edge_dir = edge_dir
        self.transform = transforms.ToTensor()

        # Supported extensions
        self.supported_extensions = ('.tif', '.png', '.jpg', '.jpeg')

        # Get all files with supported extensions and extract base names
        self.names = []
        for f in os.listdir(image_dir):
            if f.lower().endswith(self.supported_extensions):
                self.names.append(os.path.splitext(f)[0])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        # Try different extensions for each file
        def find_file(base_dir, base_name):
            for ext in self.supported_extensions:
                path = os.path.join(base_dir, base_name + ext)
                if os.path.exists(path):
                    return path
            raise FileNotFoundError(
                f"No file found for {base_name} in {base_dir} with extensions {self.supported_extensions}")

        image_path = find_file(self.image_dir, name)
        mask_path = find_file(self.mask_dir, name)
        dist_path = find_file(self.dist_dir, name)
        edge_path = find_file(self.edge_dir, name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        dist = Image.open(dist_path).convert('L')
        edge = Image.open(edge_path).convert('L')

        image = self.transform(image)
        mask = self.transform(mask)
        dist = self.transform(dist)
        edge = self.transform(edge)

        return [image, mask, dist, edge]


class infer_dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.names =[os.path.splitext(f)[0] for f in os.listdir(image_dir)]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image_path = os.path.join(self.image_dir, name+".png")
        geo_trans, geo_proj, _ = get_geo_info(image_path)
        image = Image.open(image_path).convert('RGB')
        image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = image_transform(image)
        return [image, name, geo_trans, geo_proj]

def infer_dataloader(image_dir):
    dataset = infer_dataset(image_dir)
    infer_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    return infer_dataloader

def train_val_dataloader(train_image_dir, train_mask_dir, train_dist_dir, train_edge_dir,
                         val_image_dir, val_mask_dir, val_dist_dir, val_edge_dir,
                         batch_size):
    train_dataset = Train_val_dataset(train_image_dir, train_mask_dir, train_dist_dir, train_edge_dir)
    val_dataset = Train_val_dataset(val_image_dir, val_mask_dir, val_dist_dir, val_edge_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def test_dataloader(test_image_dir, test_mask_dir, test_dist_dir, test_edge_dir, batch_size=1):
    test_dataset = Train_val_dataset(test_image_dir, test_mask_dir, test_dist_dir, test_edge_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return test_dataloader

if __name__ == '__main__':
    test_image_dir = r"C:\Users\Administrator\Desktop\HLJ\test\images"
    test_mask_dir = r"C:\Users\Administrator\Desktop\HLJ\test\masks"
    test_edge_dir = r"C:\Users\Administrator\Desktop\HLJ\test\edges"
    test_dist_dir = r"C:\Users\Administrator\Desktop\HLJ\test\dist_masks"
    infer_dir     = r"C:\\Users\Administrator\Desktop\data\XJ\test\images"
    infer_dataset = infer_dataset(infer_dir)
    for sample in infer_dataset:
        #print(sample[0])
        #print(sample[1])
        #print(sample[2])
        print(sample[3]) # None
        break







