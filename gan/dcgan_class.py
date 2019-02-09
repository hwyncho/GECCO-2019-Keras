""" Module that contains DCGAN class """

from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, LeakyReLU, ReLU, Reshape)
from keras.models import Model, Sequential

from .base_class import BaseClass


class DCGAN(BaseClass):
    def _build_generator(self):
        height = int(self.img_height / 4)
        width = int(self.img_width / 4)
        channel = self.img_channel

        generator = Sequential()

        # Fully connected layer
        generator.add(Dense(units=height * width * 64, use_bias=False, input_dim=self.latent_size))
        generator.add(BatchNormalization())
        generator.add(ReLU())
        generator.add(Reshape(target_shape=(height, width, 64)))

        # Deconvolution layer 1
        generator.add(Conv2DTranspose(filters=64, kernel_size=5, strides=1, padding="same", use_bias=False))
        generator.add(BatchNormalization())
        generator.add(ReLU())

        # Deconvolution layer 2
        generator.add(Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding="same", use_bias=False))
        generator.add(BatchNormalization())
        generator.add(ReLU())

        # Deconvolution layer 3
        generator.add(Conv2DTranspose(filters=channel, kernel_size=5, strides=2, padding="same", activation="tanh",
                                      use_bias=False))

        latent_vector = Input(shape=(self.latent_size,))
        generated_img = generator(latent_vector)

        return Model(latent_vector, generated_img)

    def _build_discriminator(self):
        discriminator = Sequential()

        # Convolution layer 1
        discriminator.add(Conv2D(filters=64, kernel_size=5, strides=2, padding="same"))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(rate=0.3))

        # Convolution layer 2
        discriminator.add(Conv2D(filters=128, kernel_size=5, strides=2, padding="same"))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(rate=0.3))

        # Fully connected layer
        discriminator.add(Flatten())
        discriminator.add(Dense(units=1, activation="sigmoid"))

        input_img = Input(shape=(self.img_height, self.img_width, self.img_channel,))
        validity = discriminator(input_img)

        return Model(input_img, validity)
