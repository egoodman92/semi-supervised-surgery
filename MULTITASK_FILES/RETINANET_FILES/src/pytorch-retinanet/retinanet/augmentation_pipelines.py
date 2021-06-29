import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


often = lambda aug: iaa.Sometimes(0.75, aug)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
rarely = lambda aug: iaa.Sometimes(0.2, aug)
very_rarely = lambda aug: iaa.Sometimes(0.1, aug)

# CHANGE TO FROM WILL'S DEFINED N AME AUGS??? 
# (Which I know how I wanna change)
# As a v1 of the pipeline 
my_pipeline_v1 = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        often(iaa.Affine(
            scale={"x": (0.8, 1.3), "y": (0.8, 1.3)},
            # scale images to 80-120% of their size, individually per axis
            rotate=(-10, 10),  # rotate by -45 to +45 degrees
            shear=(-25, 25),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 4),
                   [ 
                       # WEAK AUGMENTERS 
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5), # add gaussian noise to images
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation 
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.Multiply((0.5, 1.5), per_channel=0.5), 
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                       
                       # MEDIUM AUGMENTERS 
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       iaa.Invert(0.05, per_channel=True),  # invert color channels
                       iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       
                       # STRONG AUGMENTERS 
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.EdgeDetect(alpha=(0.5, 1.0))), 
                       
                       
                   ],
                   random_order=True
                   )
    ]) 


always = lambda aug: iaa.Sometimes(1.0, aug) # For debugging
veryoften = lambda aug: iaa.Sometimes(0.9, aug)
often = lambda aug: iaa.Sometimes(0.7, aug)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
rarely = lambda aug: iaa.Sometimes(0.2, aug)
very_rarely = lambda aug: iaa.Sometimes(0.1, aug)
never = lambda aug: iaa.Sometimes(0.0, aug) # For debugging 

my_pipeline_v2 = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        veryoften(iaa.Affine(
            scale={"x": (0.7, 1.4), "y": (0.7, 1.4)},
            # scale images to 80-120% of their size, individually per axis
            rotate=(-10, 10),  # rotate by -45 to +45 degrees
            shear=(-25, 25),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        always(iaa.SomeOf((0, 3),
                   [ 
                       # WEAK AUGMENTERS 
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5), # add gaussian noise to images
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation 
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.Multiply((0.5, 1.5), per_channel=0.5), 
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                   ], 
                   random_order=True 
                   )), 
           always(iaa.SomeOf((0, 1),
                   [
                       # MEDIUM AUGMENTERS 
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       iaa.Invert(0.05, per_channel=True),  # invert color channels
                       iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       
                       # STRONG AUGMENTERS 
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       #iaa.SimplexNoiseAlpha(iaa.EdgeDetect(alpha=(0.5, 1.0))), 
                       
                       
                   ],
                   random_order=True
                   )) 
    ]) 


always = lambda aug: iaa.Sometimes(1.0, aug) # For debugging
veryoften = lambda aug: iaa.Sometimes(0.9, aug)
often = lambda aug: iaa.Sometimes(0.7, aug)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
rarely = lambda aug: iaa.Sometimes(0.2, aug)
very_rarely = lambda aug: iaa.Sometimes(0.1, aug)
never = lambda aug: iaa.Sometimes(0.0, aug) # For debugging 

my_pipeline_v3 = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        veryoften(iaa.Affine(
            scale={"x": (0.65, 1.45), "y": (0.65, 1.45)},
            # scale images to 80-120% of their size, individually per axis
            rotate=(-10, 10),  # rotate by -45 to +45 degrees
            shear=(-30, 30),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        always(iaa.SomeOf((0, 3),
                   [ 
                       # WEAK AUGMENTERS 
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5), # add gaussian noise to images
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation 
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.Multiply((0.5, 1.5), per_channel=0.5), 
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                   ], 
                   random_order=True 
                   )), 
           always(iaa.SomeOf((0, 1),
                   [
                       # MEDIUM AUGMENTERS 
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       iaa.Invert(0.05, per_channel=True),  # invert color channels
                       iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       
                       # STRONG AUGMENTERS 
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       #iaa.SimplexNoiseAlpha(iaa.EdgeDetect(alpha=(0.5, 1.0))), 
                       
                       
                   ],
                   random_order=True
                   )) 
    ]) 


seq1 = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
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




sometimes = lambda aug: iaa.Sometimes(0.5, aug)

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




lighting = iaa.Sequential([
    invert, 
    linear_contrast, 
    grayscale 
]) 

rotate = iaa.Sequential([
        iaa.Affine(rotate=(25)),
    ]) 



img_size=224*2 

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
kitchen_sink = iaa.Sequential(
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
    ]) 