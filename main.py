#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import mnist

from gan import GAN, DCGAN

if __name__ == "__main__":
    (train_images, train_labels), (_, _) = mnist.load_data()

    img_num = train_images.shape[0]
    img_height = train_images.shape[1]
    img_width = train_images.shape[2]
    if len(train_images.shape) == 3:
        img_channel = 1
    else:
        img_channel = train_images.shape[3]

    train_images = train_images.reshape((img_num, img_height, img_width, img_channel))
    train_images = (train_images.astype(dtype="float32") - 127.5) / 127.5

    label_min = np.min(train_labels)
    label_max = np.max(train_labels)

    for label in range(label_min, label_max + 1):
        indices = np.where(train_labels == label)[0]
        images = np.take(train_images, indices, axis=0)

        # GAN with Genetic Algorithms
        gan = GAN(img_shape=images.shape[1:], latent_size=128)
        gan.train(images, epochs=1600, batch_size=1024, with_ga=True, debug_mode=True)
        gan.save_model(dir_path="./Models/GA-GAN/{0}".format(label))
        gan.generate(sample_num=16, save_path="./Results/GA-GAN/{}.png".format(label))
        del gan

        # GAN
        gan = GAN(img_shape=images.shape[1:], latent_size=128)
        gan.train(images, epochs=1600, batch_size=1024, with_ga=False, debug_mode=True)
        gan.save_model(dir_path="./Models/GAN/{0}".format(label))
        gan.generate(sample_num=16, save_path="./Results/GAN/{0}.png".format(label))
        del gan

        # DCGAN with Genetic Algorithms
        gan = DCGAN(img_shape=images.shape[1:], latent_size=128)
        gan.train(images, epochs=600, batch_size=1024, with_ga=True, debug_mode=True)
        gan.save_model(dir_path="./Models/GA-DCGAN/{0}".format(label))
        gan.generate(sample_num=16, save_path="./Results/GA-DCGAN/{0}.png".format(label))
        del gan

        # DCGAN
        gan = DCGAN(img_shape=images.shape[1:], latent_size=128)
        gan.train(images, epochs=600, batch_size=1024, with_ga=False, debug_mode=True)
        gan.save_model(dir_path="./Models/DCGAN/{0}".format(label))
        gan.generate(sample_num=16, save_path="./Results/DCGAN/{0}.png".format(label))
        del gan
