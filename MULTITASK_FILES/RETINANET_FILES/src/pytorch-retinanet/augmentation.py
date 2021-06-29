import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import torch

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5),
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    # iaa.Sometimes(
    #     0.5,
    #     iaa.GaussianBlur(sigma=(0, 0.5))
    # ),
    # Strengthen or weaken the contrast in each image.
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    iaa.Multiply((0.7, 1.5)),
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

def augment(aug_names):
    img_size = 448
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    none = iaa.Fliplr(0.0)
    flip_lr = iaa.Fliplr(1.0)
    flip_ud = iaa.Flipud(1.0)
    crop_n_pad =iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255))
    affine = iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )
    superpixels = iaa.Superpixels(p_replace=(0, 0.3), n_segments=(20, 200))
    blur = iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),])
    sharpen = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
    emboss = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
    simplex_noise = iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),]))
    gaussian_noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
    dropout = iaa.OneOf([iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),])
    invert = iaa.Invert(0.25, per_channel=True)
    add = iaa.Add((-10, 10), per_channel=0.5)
    hue_saturation = iaa.AddToHueAndSaturation((-20, 20))
    multiply = iaa.OneOf([iaa.Multiply((0.5, 1.5), per_channel=0.5),
                          iaa.FrequencyNoiseAlpha(exponent=(-4, 0),
                                       first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                       second=iaa.LinearContrast((0.5, 2.0)))])
    linear_contrast = iaa.LinearContrast((0.5, 2.0), per_channel=0.5)
    grayscale = iaa.Grayscale(alpha=(0.5, 1.0))
    elastic_transform = iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
    piecewise_affine = iaa.PiecewiseAffine(scale=(0.01, 0.05))
    perspective_transform = iaa.PerspectiveTransform(scale=(0.01, 0.1))
    augmentations = {'Original': none, 
                     'Flip left to right': flip_lr, 
                     'Flip upside down': flip_ud, 
                     'Crop and pad': crop_n_pad, 
                     'Affine transformations': affine, 
                     'Superpixels': superpixels, 
                     'Blur': blur, 
                     'Sharpen': sharpen, 
                     'Emboss': emboss, 
                     'Simplex noise': simplex_noise, 
                     'Gaussian noise': gaussian_noise, 
                     'Dropout': dropout, 
                     'Invert': invert, 
                     'Add': add, 'Hue saturation': hue_saturation, 
                     'Multiply': multiply, 
                     'Linear contrast': linear_contrast, 
                     'Grayscale': grayscale, 
                     'Elastic transform': elastic_transform, 
                     'Piecewise affine': piecewise_affine, 
                     'Perspective transform': perspective_transform}
    aug_names = []
    aug_list = [iaa.CropToFixedSize(img_size, img_size)]
    for i in range(len(augmentations.keys())):
        aug_name = list(augmentations.keys())[i]
        aug_names.append(aug_name)
        aug_list.append(augmentations[aug_name])
    aug = iaa.Sequential(aug_list)
    augDet = aug.to_deterministic()
    return augDet

def convert_bounding_boxes(images, annotations):
    bbs = []

    for index, image in enumerate(images):
        boxes = []
        annotation = annotations[index]
        for row in annotation:
            box = BoundingBox(x1=row[0], y1=row[1], x2=row[2], y2=row[3], label=row[4])
            boxes.append(box)
        bbs.append(BoundingBoxesOnImage(boxes, shape=image.shape))

    return bbs

def revert_bounding_boxes(bbs):
    result = []

    for boxes_on_image in bbs:
        annotations = np.zeros((0, 5))

        for bounding_box in boxes_on_image:
            annotation = np.zeros((1, 5))

            annotation[0, 0] = bounding_box.x1
            annotation[0, 1] = bounding_box.y1
            annotation[0, 2] = bounding_box.x2
            annotation[0, 3] = bounding_box.y2
            annotation[0, 4] = bounding_box.label

            annotations = np.append(annotations, annotation, axis=0)

        # result.append(torch.from_numpy(annotations))
        result.append(annotations)
    result = torch.from_numpy(np.stack(result, axis=0)).float()

    return result
