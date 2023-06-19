from collections import namedtuple
import os
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LEP_Triplet = namedtuple('LEP_Triplet', 'image_path, edge_map_path, caption')

class LEPDataset(Dataset):
    def __init__(self, dataset_dir, edge_map_dir):
        assert os.path.exists(dataset_dir)
        assert os.path.exists(edge_map_dir)
        
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1)
            ]
        )
        
        # for better intellisense
        self._data:List[LEP_Triplet] = list()

        for subfolder in os.listdir(dataset_dir):
            subfolder_path = os.path.join(dataset_dir, subfolder)

            # use class names as captions
            caption = subfolder

            # Iterate through all images in the subfolder
            for image_fname in os.listdir(subfolder_path):
                if not image_fname.endswith('.jpg'):
                    continue

                image_path = os.path.join(subfolder_path, image_fname)
                image_unsure = Image.open(image_path)
                if image_unsure.mode != "RGB":
                    continue
                    
                edge_map_fname = image_fname.replace('.jpg', '.png')
                edge_map_path = os.path.join(edge_map_dir, edge_map_fname)

                # we have some missing edge maps (I manually remove some big images; H > 1000)
                if not os.path.exists(edge_map_path):
                    continue
                    
                # append existing files only
                self._data.append(LEP_Triplet(image_path, edge_map_path, caption))

    def __getitem__(self, index):
        triplet = self._data[index]

        image = self.transform(Image.open(triplet.image_path))

        edge_map = Image.open(triplet.edge_map_path)
        edge_map = Image.merge('RGB', (edge_map, edge_map, edge_map))
        edge_map = self.transform(edge_map)

        # threshold edge_map with 0.5 in image domain
        edge_map[edge_map < 0] = 0
        
        return image, edge_map, triplet.caption

    def __len__(self):
        return len(self._data)