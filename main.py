import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from skimage.transform import resize
from skimage.io import imread
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
# marks = pd.read_csv('../input/train_ship_segmentations.csv')  # Markers for ships
# images = os.listdir('../input/train')  # Images for training
# os.chdir("../input/train")

def valid_set(valid_ids):
    pass


class AirbusDetector:
    def __init__(self):
        self.model = None
        self.input_img = None
        self.output_img = None
        self.iou_metric = self.iou_metric()

    def iou_metric(self):
        '''
        Function that computes IoU metric
        '''
        return [IoU, back_IoU]

    def iou(self, y_true, y_pred, tresh=1e-10):
        '''
        Function that computes IoU
        '''
        Intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        Union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - Intersection
        return K.mean((Intersection + tresh) / (Union + tresh), axis=0)

    def iou_loss(self, y_true, y_pred):

        '''
        Function that computes IoU loss
        '''
        return -self.iou(y_true, y_pred)

    def load_model(self, model_path):
        self.model = load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss=IoU_loss, metrics=[self.iou_metric])


    def create_model(self):
        '''
        Function that creates U-Net model
        '''
        input_img = Input((768, 768, 3), name='img')
        model = Model(inputs=[input_img], outputs=self.get_unet(input_img))
        model.compile(optimizer='adam', loss=IoU_loss, metrics=[self.iou_metric])
        self.model = model

    def get_unet(self, input_img):
        '''
        Function that creates U-Net model
        '''
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

        u5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
        u5 = concatenate([u5, c3])
        c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
        c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

        u6 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c2])
        c6 = Conv2D(16, (3, 3), activation='relu', padding='same')(u6)
        c6 = Conv2D(16, (3, 3), activation='relu', padding='same')(c6)

        u7 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c1], axis=3)
        c7 = Conv2D(8, (3, 3), activation='relu', padding='same')(u7)
        c7 = Conv2D(8, (3, 3), activation='relu', padding='same')(c7)

        return Conv2D(1, (1, 1), activation='sigmoid')(c7)

    def train_model(self, epochs=100, batch_size=32):
        '''
        Function that trains model
        '''
        train_ids, valid_ids = train_test_split(images, test_size=0.3)
        train_gen = self.generator(train_ids, batch_size)

        results = self.model.fit_generator(Generator(train_ids, batch_size),
                                           steps_per_epoch=len(train_ids) // batch_size,
                                           epochs=epochs)
        return results

    def predict_mask(self, img):
        '''
        Function that predicts mask for image
        '''
        image = (io.imread('test_v2/' + img)) / 255.0
        self.input_img = image
        image = image.reshape(1, 768, 768, 3)
        self.output_img = self.model.predict(image).reshape(768, 768)
        self.output_mask = np.round(self.output_img)
        return self.output_mask

    def save_model(self, model_path):
        '''
        Function that saves model
        '''
        self.model.save(model_path)

    @staticmethod
    def split_data(images, test_size=0.3):
        '''
        Function that splits data into train and validation sets
        '''
        train_ids, valid_ids = train_test_split(images, test_size=test_size)
        return train_ids, valid_ids

    def generator(self, train_ids, batch_size):
        '''
        Function that creates generator for training
        '''
        while True:
            ix = np.random.choice(np.arange(len(train_ids)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                img = self.load_image(train_ids[i])
                mask = self.load_mask(train_ids[i])
                imgs.append(img)
                labels.append(mask)
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels

    @staticmethod
    def load_image(key):
        '''
        Function that loads image
        '''
        img = imread('train/' + key)
        img = resize(img, (768, 768), mode='constant', preserve_range=True)
        return img

    @staticmethod
    def load_mask(key):
        '''
        Function that loads mask
        '''
        if is_empty(key):
            mask = np.zeros((768, 768, 1))
        else:
            mask = np.zeros((768, 768, 1))
            df = marks[marks['ImageId'] == key].iloc[:, 1]
            for i in range(len(df)):
                mask += mask_part(df.iloc[i])
        mask = np.round(resize(mask, (128, 128), mode='constant', preserve_range=True))
        return mask

    def plot_results(self, results):
        '''
        Function that plots results of training
        '''
        plt.figure(figsize=(12, 8))
        plt.plot(results.history['loss'], label='train')
        plt.legend()
        plt.show()


def mask_part(pic):
    '''
    Function that encodes mask for single ship from .csv entry into numpy matrix
    '''
    back = np.zeros(768 ** 2)
    starts = pic.split()[0::2]
    lens = pic.split()[1::2]
    for i in range(len(lens)):
        back[(int(starts[i]) - 1):(int(starts[i]) - 1 + int(lens[i]))] = 1
    return np.reshape(back, (768, 768, 1))


def is_empty(key):
    '''
    Function that checks if there is a ship in image
    '''
    df = marks[marks['ImageId'] == key].iloc[:, 1]
    if len(df) == 1 and type(df.iloc[0]) != str and np.isnan(df.iloc[0]):
        return True
    else:
        return False


def masks_all(key):
    '''
    Merges together all the ship markers corresponding to a single image
    '''
    df = marks[marks['ImageId'] == key].iloc[:, 1]
    masks = np.zeros((768, 768, 1))
    if is_empty(key):
        return masks
    else:
        for i in range(len(df)):
            masks += mask_part(df.iloc[i])
        return np.transpose(masks, (1, 0, 2))

def transformX(X):
    x = np.copy(X)
    x[:, :, 0] = x[:, :, 0] + np.random.normal(loc=0.0, scale=0.01, size=(768, 768))
    x[:, :, 1] = x[:, :, 1] + np.random.normal(loc=0.0, scale=0.01, size=(768, 768))
    x[:, :, 2] = x[:, :, 2] + np.random.normal(loc=0.0, scale=0.01, size=(768, 768))
    # Adding Gaussian noise on each rgb channel; this way we will NEVER get two completely same images.
    # Note that this transformation is not performed on Y
    x[np.where(x < 0)] = 0
    x[np.where(x > 1)] = 1
    # axes swap:
    if np.random.rand() < 0.5:  # 0.5 chances for this transformation to occur (same for two below)
        x = np.swapaxes(x, 0, 1)
    # vertical flip:
    if np.random.rand() < 0.5:
        x = np.flip(x, 0)
    # horizontal flip:
    if np.random.rand() < 0.5:
        x = np.flip(x, 1)
    return x

def transform(X, Y):
    '''
    Function for augmenting images.
    It takes original image and corresponding mask and performs the
    same flipping and rotation transforamtions on both in order to
    perserve the overlapping of ships and their masks
    '''
    # add noise:
    x = np.copy(X)
    y = np.copy(Y)
    x[:, :, 0] = x[:, :, 0] + np.random.normal(loc=0.0, scale=0.01, size=(768, 768))
    x[:, :, 1] = x[:, :, 1] + np.random.normal(loc=0.0, scale=0.01, size=(768, 768))
    x[:, :, 2] = x[:, :, 2] + np.random.normal(loc=0.0, scale=0.01, size=(768, 768))
    # Adding Gaussian noise on each rgb channel; this way we will NEVER get two completely same images.
    # Note that this transformation is not performed on Y
    x[np.where(x < 0)] = 0
    x[np.where(x > 1)] = 1
    # axes swap:
    if np.random.rand() < 0.5:  # 0.5 chances for this transformation to occur (same for two below)
        x = np.swapaxes(x, 0, 1)
        y = np.swapaxes(y, 0, 1)
    # vertical flip:
    if np.random.rand() < 0.5:
        x = np.flip(x, 0)
        y = np.flip(y, 0)
    # horizontal flip:
    if np.random.rand() < 0.5:
        x = np.flip(x, 1)
        y = np.flip(y, 1)
    return x, y


def make_batch(files, batch_size):
    '''
    Creates batches of images and masks in order to feed them to NN
    '''
    X = np.zeros((batch_size, 768, 768, 3))
    Y = np.zeros((batch_size, 768, 768, 1))  # I add 1 here to get 4D batch
    for i in range(batch_size):
        ship = os.path.join('train/', np.random.choice(files))
        X[i] = (io.imread(ship)) / 255.0  # Original images are in 0-255 range, I want it in 0-1
        Y[i] = masks_all(ship)
    return X, Y


def Generator(files, batch_size):
    '''
    Generates batches of images and corresponding masks
    '''
    while True:
        X, Y = make_batch(files, batch_size)
        for i in range(batch_size):
            X[i], Y[i] = transform(X[i], Y[i])
        yield X, Y


def IoU(y_true, y_pred, tresh=1e-10):
    Intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    Union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - Intersection
    return K.mean((Intersection + tresh) / (Union + tresh), axis=0)


# Intersection over Union for Background
def back_IoU(y_true, y_pred):
    return IoU(1 - y_true, 1 - y_pred)


# Loss function
def IoU_loss(in_gt, in_pred):
    # return 2 - back_IoU(in_gt, in_pred) - IoU(in_gt, in_pred)
    return 1 - IoU(in_gt, in_pred)


if __name__ == '__main__':
    # Loading data
    marks = pd.read_csv('train_ship_segmentations_v2.csv')
    images = os.listdir('train')[:5000]
    test_images = os.listdir('test_v2')[:30]
    # Splitting data into train and validation sets
    detector = AirbusDetector()
    detector.split_data(images)
    # Creating NN
    detector.create_model()
    # Training NN
    res = detector.train_model(epochs=3, batch_size=32)
    detector.plot_results(res)
    # Saving model
    detector.save_model('model.h5')
    # Loading model
    detector.load_model('model.h5')
    # Predicting masks
    masks = []
    for img in test_images:
        masks.append(detector.predict_mask(img))
    # Creating submission file
    submission = pd.DataFrame()
    submission['ImageId'] = test_images
    submission['EncodedPixels'] = masks
    submission.to_csv('submission.csv', index=False)



