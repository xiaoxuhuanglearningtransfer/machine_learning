import os
import json
import glob

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import resnet50


def main():
    im_height = 224
    im_width = 224
    num_classes = 36

    # 加载图像

    img_path = "ban.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # 重置图像大小为224x224
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # 缩放像素区间至 (0-1)
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    img = np.array(img).astype(np.float32)
    img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

    # 将图像添加到作为唯一成员的批处理中。
    img = (np.expand_dims(img, 0))

    # 读取分类文档
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 创建模型
    feature = resnet50(num_classes=num_classes, include_top=False)
    feature.trainable = False
    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])

    # 加载权重
    weights_path = './save_weights/resNet_50.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    # 预测
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)


    if result[predict_class]<=0.8:
        class_indict[str(predict_class)]="unkonw"
        plt.title(class_indict[str(predict_class)])
        plt.show()
    else:
        #print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
         #                                        result[predict_class])
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],0.928)
        plt.title(print_res)
        plt.show()
        print(print_res)
    # for i in range(len(result)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               result[i].numpy()))


if __name__ == '__main__':
    main()
