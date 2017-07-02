import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os
import os.path

def find_imgs(root_dir, decoder):
    imgs = []
    for root_path, _, filename_list in os.walk(root_dir):
        imgs.extend([(os.path.join(root_path, filename), decoder(filename)) for filename in filename_list])
    return imgs


def default_decoder(filename):
    return [filename]


class FileNameDataset(data.Dataset):
    def __init__(self, root_dir, filename_decoder = default_decoder):
        self.filename_decoder = filename_decoder
        self.root_dir = root_dir
        self.imgs = find_imgs(root_dir, filename_decoder)
        self.transform = transforms.ToTensor()
        if len(self.imgs) == 0:
            raise(RuntimeError('found 0 img in %s' % (root_dir)))

    def __len__(self, ):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, target = self.imgs[idx]
        img = Image.open(path)
        img = self.transform(img)
        return img, target

