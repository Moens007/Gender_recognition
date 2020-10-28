import cv2
import albumentations as A
from albumentations import (Blur,Flip,ShiftScaleRotate,GridDistortion,ElasticTransform,HorizontalFlip,CenterCrop,
                            HueSaturationValue,Transpose,RandomBrightnessContrast,CLAHE,RandomCrop,Cutout,CoarseDropout,
                            CoarseDropout,Normalize,ToFloat,OneOf,Compose,Resize,RandomRain,RandomFog,Lambda
                            ,ChannelDropout,ISONoise,VerticalFlip,RandomGamma,RandomRotate90)

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(always_apply=False, p=0.5),  #水平翻转
        A.Blur(blur_limit=7, always_apply=False, p=0.5),  #布尔模糊
        A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),  #随机Gamma
        A.Rotate(limit=15, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5), #旋转
        A.OneOf([
                 A.RandomBrightness(limit=0.5, always_apply=False, p=0.5),
                 # A.RandomContrast(limit=0.8, always_apply=False, p=0.5),
                 A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False,
                                    p=0.5),
                 # A.ChannelShuffle(always_apply=False, p=0.5),
                 ]),
        A.OneOf([
            # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None,
            #                     mask_value=None, always_apply=False, p=0.5),
            A.MotionBlur(blur_limit=7, always_apply=False, p=0.5),
            A.GaussianBlur(blur_limit=[3,7], always_apply=False, p=0.5),
            A.MedianBlur(blur_limit=7, always_apply=False, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5)
        ]),


    ])