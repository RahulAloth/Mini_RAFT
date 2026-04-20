# transforms/stereo_augment.py
import random
from PIL import Image


class StereoAugmentation:
    """
    Applies identical geometric transforms to:
        - left image
        - right image
        - disparity map
        - valid mask
    """

    def __init__(self, output_size=(288, 576), min_scale=-0.1, max_scale=0.1, do_flip=True):
        self.output_size = output_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.do_flip = do_flip

    def __call__(self, sample):
        left = sample["left"]
        right = sample["right"]
        disp = sample["disparity"]
        mask = sample["valid_mask"]

        # 1. Random scaling
        h, w = left.height, left.width
        scale = 2 ** random.uniform(self.min_scale, self.max_scale)
        hs, ws = int(h * scale), int(w * scale)

        left = left.resize((ws, hs), Image.BILINEAR)
        right = right.resize((ws, hs), Image.BILINEAR)

        disp = disp.resize((ws, hs), Image.NEAREST)
        disp = disp.point(lambda x: x * scale)

        mask = mask.resize((ws, hs), Image.NEAREST)

        # 2. Random crop
        out_h, out_w = self.output_size
        y0 = random.randint(0, hs - out_h)
        x0 = random.randint(0, ws - out_w)
        crop = (x0, y0, x0 + out_w, y0 + out_h)

        left = left.crop(crop)
        right = right.crop(crop)
        disp = disp.crop(crop)
        mask = mask.crop(crop)

        # 3. Random horizontal flip
        if self.do_flip and random.random() < 0.5:
            left = left.transpose(Image.FLIP_LEFT_RIGHT)
            right = right.transpose(Image.FLIP_LEFT_RIGHT)
            disp = disp.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        sample["left"] = left
        sample["right"] = right
        sample["disparity"] = disp
        sample["valid_mask"] = mask

        return sample
