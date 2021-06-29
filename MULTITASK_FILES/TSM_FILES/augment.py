from imgaug import augmenters as iaa
import imgaug as ia

def augmentations_for_method(method, img_size=224):
    d = {'flip': ['Flip left to right', 'Flip upside down'],
         'transform': ['Crop and pad', 'Affine transformations', 'Elastic transform',
                       'Piecewise affine', 'Perspective transform'],
         'noise': ['Simplex noise', 'Gaussian noise', 'Dropout'],
         'color': ['Linear contrast', 'Grayscale', 'Hue saturation', 'Invert', 'Add', 'Multiply'],
         'emboss': ['Emboss', 'Sharpen'],
         'blur': ['Blur']
         }
    aug_names = []
    for aug in method.split(","):
        aug_names = aug_names + d[aug]
    if not len(aug_names) == 0:
        print("WARNING: Method not found")
    return get_augmentations(aug_names, img_size)

def get_augmentations(aug_names, img_size=224):

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    none = iaa.Fliplr(0.0)

    flip_lr = iaa.Fliplr(0.5)
    flip_ud = iaa.Flipud(0.2)

    crop_n_pad = sometimes(iaa.CropAndPad(
        percent=(-0.05, 0.1),
        pad_mode=ia.ALL,
        pad_cval=(0, 255)))

    affine = sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        # translate by -20 to +20 percent (per axis)
        rotate=(-45, 45),  # rotate by -45 to +45 degrees
        shear=(-16, 16),  # shear by -16 to +16 degrees
        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    ))

    superpixels = sometimes(iaa.Superpixels(p_replace=(0, 0.3), n_segments=(20, 200)))

    blur = iaa.OneOf([
        iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
        iaa.AverageBlur(k=(2, 7)),
        # blur image using local means with kernel sizes between 2 and 7
        iaa.MedianBlur(k=(3, 11)), ])
    sharpen = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
    emboss = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
    simplex_noise = iaa.SimplexNoiseAlpha(iaa.OneOf([
        iaa.EdgeDetect(alpha=(0.5, 1.0)),
        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)), ]))
    gaussian_noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
    dropout = iaa.OneOf([iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                         iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2), ])
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
    piecewise_affine = sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
    perspective_transform =sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

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
                     'Add': add,
                     'Hue saturation': hue_saturation,
                     'Multiply': multiply,
                     'Linear contrast': linear_contrast,
                     'Grayscale': grayscale,
                     'Elastic transform': elastic_transform,
                     'Piecewise affine': piecewise_affine,
                     'Perspective transform': perspective_transform}

    aug_list = [iaa.CropToFixedSize(img_size, img_size)]
    aug_list2 = []

    aug_names1 = []
    aug_names2 = []
    for aug_name in aug_names:
        if aug_name in ['Flip left to right', 'Flip upside down', 'Crop and pad', 'Affine transformations']:
            aug_names1.append(aug_name)
        else:
            aug_names2.append(aug_name)

    for aug_name in aug_names1:
        aug_list.append(augmentations[aug_name])

    aug_list.append(iaa.SomeOf((0, 5), aug_list2))

    aug = iaa.Sequential(aug_list)
    #augDet = aug.to_deterministic()
    return aug