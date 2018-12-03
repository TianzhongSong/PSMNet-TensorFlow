from models.model import Model
from utils.data_loader import DataLoaderKITTI
import tensorflow as tf
import numpy as np
import cv2


def main():
    left_img = ''
    right_img = ''

    bat_size = 1
    maxdisp = 128

    with tf.Session() as sess:
        PSMNet = Model(sess, height=368, weight=1224, batch_size=bat_size, max_disp=maxdisp, lr=0.0001)
        saver = tf.train.Saver()
        saver.restore(sess, './weights/PSMNet.ckpt-600')

        img_L = cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_BGR2RGB)
        img_L = cv2.resize(img_L, (368, 1224))
        img_R = cv2.cvtColor(cv2.imread(right_img), cv2.COLOR_BGR2RGB)
        img_R = cv2.resize(img_R, (368, 1224))

        img_L = DataLoaderKITTI.mean_std(img_L)
        img_L = np.expand_dims(img_L, axis=0)
        img_R = DataLoaderKITTI.mean_std(img_R)
        img_R = np.expand_dims(img_R, axis=0)
        pred = PSMNet.predict(img_L, img_R)

        item = (pred * 255 / pred.max()).astype(np.uint8)
        pred_rainbow = cv2.applyColorMap(item, cv2.COLORMAP_RAINBOW)
        cv2.imwrite('prediction.png', pred_rainbow)


if __name__ == '__main__':
    main()
