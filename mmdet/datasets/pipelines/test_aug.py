import mmcv
import os
from ..registry import PIPELINES
from .compose import Compose


@PIPELINES.register_module
class MultiScaleFlipAug(object):

    def __init__(self, transforms, img_scale=None, ratio_scale=None, flip=False, ver_flip=False, filter_prefix=None):
        self.transforms = Compose(transforms)
        self.ratio_scale = ratio_scale
        self.filter_prefix = filter_prefix if filter_prefix is not None else ['fl', 'gx']
        if self.ratio_scale is None:
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)
        else:
            if isinstance(ratio_scale, list):
                if len(ratio_scale) == 2:
                    self.img_scale = ratio_scale
                else:
                    self.img_scale = [ratio_scale, ratio_scale]
            else:
                self.img_scale = [[ratio_scale], [ratio_scale]]
                
        self.flip = flip
        if ver_flip:
            self.flip_direction = ['horizontal', 'vertical']
        else:
            self.flip_direction = ['horizontal',]

    def __call__(self, results):
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        if (self.ratio_scale is not None) and (os.path.basename(results['filename']) == 'De'):
            img_scales = self.img_scale[0]
        elif (self.ratio_scale is not None) and (os.path.basename(results['filename']) != 'De'):
            img_scales = self.img_scale[1]
        else:
            img_scales = self.img_scale
        if os.path.basename(results['filename'])[:2] in self.filter_prefix:
            flip_directions = ['horizontal',]
        else:
            flip_directions = self.flip_direction
        for scale in img_scales:
            if len(self.flip_direction) == 1: 
                for flip in flip_aug:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = 'horizontal'
                    data = self.transforms(_results)
                    aug_data.append(data)
            else:
                _results = results.copy()
                _results['scale'] = scale
                _results['flip'] = False
                _results['flip_direction'] = 'horizontal'
                data = self.transforms(_results)
                aug_data.append(data)
                
                for flip_direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = True
                    _results['flip_direction'] = flip_direction
                    data = self.transforms(_results)
                    aug_data.append(data)
                
                
                
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transforms={}, img_scale={}, flip={})'.format(
            self.transforms, self.img_scale, self.flip)
        return repr_str
