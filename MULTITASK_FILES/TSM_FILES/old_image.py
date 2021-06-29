import cv2
import numpy as np
from typing import Dict, Iterable
from imgaug import augmenters as iaa
import imgaug as ia
from augment import augmentations_for_method
import skimage.io
import skimage.transform
import skimage.color
import skimage



def augment_raw_frames(raw_frames, method='val', img_size=224):
    if method == 'ten':  # 4 corners + center crop. 50% flipped
        #aug = iaa.Sequential([iaa.Fliplr(0.5),
        #                      iaa.OneOf([
        #                          iaa.CenterCropToFixedSize(img_size, img_size),
        #                          iaa.CropToFixedSize(img_size, img_size, position='left-top'),
        #                          iaa.CropToFixedSize(img_size, img_size, position='left-bottom'),
        #                          iaa.CropToFixedSize(img_size, img_size, position='right-top'),
        #                          iaa.CropToFixedSize(img_size, img_size, position='right-bottom')
        #                      ])
        #                  ])
        aug = iaa.Sequential([])
    elif method == 'center-crop':
        aug = iaa.Sequential([
            iaa.CenterCropToFixedSize(img_size, img_size),
        ])
    elif method == '04-20':
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        aug = iaa.Sequential([iaa.Fliplr(0.5),
                              iaa.LinearContrast(),
                              iaa.Flipud(0.2),
                              sometimes(
                                iaa.Rotate()
                              )
                         ])
    elif method == 'kitchen-sink':
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        aug = iaa.Sequential(
            [
                iaa.CropToFixedSize(img_size, img_size),
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                   iaa.FrequencyNoiseAlpha(
                                       exponent=(-4, 0),
                                       first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                       second=iaa.LinearContrast((0.5, 2.0))
                                   )
                               ]),
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               # move pixels locally around (with random strengths)
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # sometimes move parts of the image around
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ]
        )
    elif method == 'val':
        #aug = iaa.Sequential([
        #                    iaa.Resize({"shorter-side": img_size, "longer-side": "keep-aspect-ratio"}),
        #                    iaa.CenterCropToFixedSize(img_size, img_size)
        #                  ])
        aug = iaa.Sequential([])
    else:
        aug = augmentations_for_method(method)


    augDet = aug.to_deterministic()
    num_frames = len(raw_frames)
    #print("RAWFRAMES", raw_frames[0][500:550, 500:550, 0])
    practice_frame_for_dims, scaled = Resizer(raw_frames[0])
    h = practice_frame_for_dims.shape[0]
    w = practice_frame_for_dims.shape[1]
    c = practice_frame_for_dims.shape[2]
    results = np.zeros((num_frames, h, w, c))
    for i in range(num_frames):
        resized_frame, scaled = Resizer(raw_frames[i])
        #print("RESIZED", resized_frame[500:550, 500:550, 0])
        results[i] = augDet.augment_image(resized_frame)
    #print("RESULTS", results[0][500:550, 500:550, 0])
    return results, scaled


def Resizer(image):
        #min_side = float(544)
        #max_side = float(704)

        min_side = float(320)
        max_side = float(320)

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)


        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        #print("Image before", image[:, :, 0])
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))), preserve_range=True)
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32
        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        return new_image, scale


def test_time_augment(raw_frames, img_size=224):
    c = raw_frames[0].shape[2]
    num_frames = 64
    videos = []
    positions = ['left-top', 'left-bottom', 'right-top', 'right-bottom', 'center']
    for position in positions:
        seq = iaa.Sequential([
            iaa.CropToFixedSize(img_size, img_size, position=position),
        ])
        seq_det = seq.to_deterministic()
        video = np.zeros((num_frames, img_size, img_size, c), raw_frames[0].dtype)
        for i in range(num_frames):
            video[i] = seq_det.augment_image(raw_frames[i])
        videos.append(video)

    for position in positions:
        seq = iaa.Sequential([
            iaa.Fliplr(1.0),
            iaa.CropToFixedSize(img_size, img_size, position=position),
        ])
        seq_det = seq.to_deterministic()
        video = np.zeros((num_frames, img_size, img_size, c), raw_frames[0].dtype)
        for i in range(num_frames):
            video[i] = seq_det.augment_image(raw_frames[i])
        videos.append(video)

    return videos


def create_video_clip(img_array, fps, output_path):
    img = img_array[0]
    size = (img.shape[0], img.shape[1])
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    cv2.addWeighted()
    out.release()


class Resize(object):
    """Resize 2D image.

    Args:
        dsize (Tuple[int, int]): Output image size (x, y). Ignored if equal to `(0,0)`. Defaults to `(0,0)`.
        fx (float): Scale factor along horizontal axis. Defaults to `0`.
        fy (float): Scale factor along vertical axis. Defaults to `0`.
        rel_scale (str): Compute scaling factor with respect to either width or height while preserving rel_scale
            ratio. `dsize` must be specified. Either `'width'` or `'height'`. Ignored if `''`. Defaults to `''`.
        match_dims (`bool`, optional): If `True`, frame will be center-cropped/padded to get right size.
    """
    NAME = 'resize'

    def __init__(self, dsize: tuple = (0, 0), fx: float = 0, fy: float = 0, rel_scale: str = '',
                 match_dims: bool = False):
        no_dsize = not dsize or dsize == (0, 0)
        if no_dsize and not rel_scale and not fx and not fy:
            raise ValueError('No resizing parameters provided.')

        if no_dsize and (not fx or not fy):
            raise ValueError('Both `fx` and `fy` must be specified if no `dsize` specified.')

        if rel_scale not in ['', 'max', 'width', 'height']:
            raise ValueError('Invalid input %s for `rel_scale`. Use \'width\' or \'height\'.' % rel_scale)

        if rel_scale:
            if no_dsize:
                raise ValueError('No `dsize` specified for relative scaling (`rel_scale`).')

        # Store dsize as height, width for computational purposes.
        self.dsize = (0, 0) if no_dsize else dsize[::-1]
        self.fx = fx
        self.fy = fy
        self.rel_scale = rel_scale
        self.match_dims = match_dims

    def process(self, frame):
        height, width = frame.shape[:2]

        dsize = self.dsize
        fx = self.fx
        fy = self.fy

        # Check if resizing needs to be done.
        if dsize and dsize != (0, 0):
            if (height, width) == dsize:
                return frame

        resized_ind = -1
        if self.rel_scale:
            rel_scale = self.rel_scale
            if self.rel_scale == 'max':
                if height < width:
                    rel_scale = 'height'
                else:
                    rel_scale = 'width'

            if rel_scale == 'height':
                fy = dsize[0] / height
                fx = fy
                new_width = int(round(width * fx))
                dsize = (dsize[0], new_width)
                resized_ind = 1  # resized width
            else:
                fx = dsize[1] / width
                fy = fx
                new_height = int(round(height * fy))
                dsize = (new_height, dsize[1])
                resized_ind = 0  # resized height

        r_frame = cv2.resize(frame, dsize[::-1], fx=fx, fy=fy)

        if self.rel_scale and self.match_dims:
            assert resized_ind != -1

            # dsize is oppositely indexed.
            dsize = self.dsize[resized_ind]

            if r_frame.shape[resized_ind] > dsize:
                # center-crop the array to get the desired size
                r_frame = self._crop_center(r_frame, self.dsize)
            elif r_frame.shape[resized_ind] < dsize:
                # zero-pad the image
                r_frame = self._zero_pad(r_frame, self.dsize)

        return r_frame

    @staticmethod
    def _zero_pad(frame, dsize):
        """Zero pad the image.

        Args:
            frame: Image to pad.
            dsize: Output image size (y, x). Note the flipped convention. This is the convention of the class.

        Returns:
            np.ndarray: Zero-padded image with dimensions `dsize`.
        """
        y, x = frame.shape[:2]

        new_y, new_x = dsize

        pady = new_y - y
        padx = new_x - x

        # If padding width is odd on either axis, we pad the start of the axis with 1 more set of 0s.
        pad_width = [(pady - (pady // 2), pady // 2), (padx - (padx // 2), padx // 2)]

        # do not pad any other dimensions
        for _ in range(2, frame.ndim):
            pad_width.append([0, 0])

        return np.pad(frame, pad_width)

    @staticmethod
    def _crop_center(frame, dsize):
        """Crop frame in the center.

        Args:
            frame: Image to crop.
            dsize: Output image size (y, x). Note the flipped convention. This is the convention of the class.

        Returns:
            np.ndarray: Center-cropped frame of dimensions `dsize`.
        """
        y, x = frame.shape[:2]

        new_y, new_x = dsize

        starty = y // 2 - new_y // 2
        startx = x // 2 - new_x // 2

        return frame[starty:starty + new_y, startx:startx + new_x, ...]
