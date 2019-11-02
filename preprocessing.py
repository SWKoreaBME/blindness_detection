"""
Image Preprocessing code for APTOS 2019 blindness detection dataset
License by Kaggle notebook url : 

Modified by BAN, KIM from korea university BME

Returns:
    [type] -- [description]
"""
import cv2
import numpy as np

def preprocessing(image, IMG_SIZE=512, sigmaX=10, tol=7) : #파라미터들은 함수에 있는것들을 참고했습니당.
    # img : 이미지, after_path : 전처리할 이미지 저장할 경로
    # load_ben_color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # crop_image_from_gray
    if image.ndim == 2 :
        mask = image > tol
        image = image[np.ix_(mask.any(1), mask.any(0))]

    elif image.ndim == 3 :
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if (check_shape == 0) :  # image is too dark so that we crop out everything,
            image = image  # return original image

        else:
            img1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            image = np.stack([img1, img2, img3], axis=-1)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image # 전처리한 함수도 리턴해줍니당