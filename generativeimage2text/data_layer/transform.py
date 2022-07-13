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

class CropRegionIfExists():
    def __init__(self, extend=1):
        self.extend = extend

    def __call__(self, data):
        debug = False
        #debug = True
        if 'rect' in data['caption']:
            rect = data['caption']['rect']
            center_x, center_y = (rect[0] + rect[2]) / 2., (rect[1] + rect[3]) / 2.
            rect_w, rect_h = rect[2] - rect[0], rect[3] - rect[1]
            extended_x0 = center_x - rect_w / 2 * self.extend
            extended_x1 = center_x + rect_w / 2 * self.extend
            extended_y0 = center_y - rect_h / 2 * self.extend
            extended_y1 = center_y + rect_h / 2 * self.extend
            w, h = data['image'].size
            extended_x0 = min(max(0, extended_x0), w - 1)
            extended_x1 = min(max(0, extended_x1), w - 1)
            extended_y0 = min(max(0, extended_y0), h - 1)
            extended_y1 = min(max(0, extended_y1), h - 1)
            extended_rect = [extended_x0, extended_y0, extended_x1, extended_y1]
            if debug:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(data['image'])
                draw.rectangle(rect)
                draw.rectangle(extended_rect)
                data['image'].show()
            if extended_rect[2] > extended_rect[0] + 1 and \
                    extended_rect[3] > extended_rect[1] + 1:
                # some annotation has some issues
                data['image'] = data['image'].crop(extended_rect)
            if debug:
                data['image'].show()
                logging.info(data['caption']['caption'])
                import ipdb;ipdb.set_trace(context=15)
        return data

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

