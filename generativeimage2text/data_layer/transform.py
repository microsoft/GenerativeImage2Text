from torchvision.transforms import transforms
from PIL import Image
import logging
from ..common import dict_has_path, dict_get_path_value, dict_remove_path
from ..common import dict_update_path_value


class RenameKey(object):
    def __init__(self, ft=None, not_delete_origin=False):
        self.ft = ft
        self.not_delete_origin = not_delete_origin
    def __repr__(self):
        return 'RenameKey(ft={}, not_delete_origin={})'.format(
            ','.join(['{}:{}'.format(k, v) for k, v in self.ft.items()]),
            self.not_delete_origin,
        )
    def __call__(self, data):
        if self.ft is None:
            return data
        for k, k1 in self.ft.items():
            # we should not fall to the situation where some data has some key
            # and some data has not some key. We should either have a key or
            # not for all data consistently. Thus, for re-naming, we should not
            # to check whether it has or not. it should always have that key.
            # otherwise, we should not specify it.
            #if dict_has_path(data, k):
            if dict_has_path(data, k):
                v = dict_get_path_value(data, k)
                dict_update_path_value(data, k1, v)
                if not self.not_delete_origin:
                    dict_remove_path(data, k)
        return data

class SelectTransform(object):
    def __init__(self, ts, selector):
        self.ts = ts
        self.selector = selector
    def __repr__(self):
        return 'SelectTransform(ts={}, selector={})'.format(
            self.ts, self.selector
        )
    def __call__(self, data):
        idx = self.selector(data)
        return self.ts[idx](data)

class ImageTransform2Dict(object):
    def __init__(self, image_transform, key='image'):
        self.image_transform = image_transform
        self.key = key

    def __call__(self, dict_data):
        out = dict(dict_data.items())
        out[self.key] = self.image_transform(dict_data[self.key])
        return out

    def __repr__(self):
        return 'ImageTransform2Dict(image_transform={})'.format(
            self.image_transform,
        )

def get_inception_train_transform(
    bgr2rgb=False,
    crop_size=224,
    small_scale=None,
    normalize=None,
    no_color_jitter=False,
    no_flip=False,
    no_aspect_dist=False,
    resize_crop=None,
    max_size=None,
    interpolation=Image.BILINEAR,
):
    if interpolation is None:
        interpolation = Image.BILINEAR
    elif interpolation == 'bicubic':
        interpolation = Image.BICUBIC
    totensor = transforms.ToTensor()
    all_trans = []
    if small_scale is None:
        small_scale = 0.08
    scale = (small_scale, 1.)
    if no_aspect_dist:
        ratio = (1., 1.)
    else:
        ratio = (3. / 4., 4. / 3.)
    if resize_crop is None:
        all_trans.append(transforms.RandomResizedCrop(
            crop_size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
        ))
    else:
        raise NotImplementedError(resize_crop)

    if not no_color_jitter:
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        all_trans.append(color_jitter)

    if not no_flip:
        all_trans.append(transforms.RandomHorizontalFlip())

    all_trans.extend([
        totensor,
        normalize,])
    data_augmentation = transforms.Compose(all_trans)
    return data_augmentation

