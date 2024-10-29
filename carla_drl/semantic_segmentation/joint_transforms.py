import random
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

class RandomHorizontalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask
