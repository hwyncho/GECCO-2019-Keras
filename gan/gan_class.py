""" Module that contains GAN class """

from keras.layers import Dense, Input, Reshape
from keras.models import Model, Sequential

from .base_class import BaseClass


class GAN(BaseClass):
    def _build_generator(self):
        img_size = int(self.img_height * self.img_width * self.img_channel)

        generator = Sequential()
        generator.add(Dense(units=256, activation="relu", input_dim=self.latent_size))
        generator.add(Dense(units=512, activation="relu"))
        generator.add(Dense(units=img_size, activation="tanh"))
        generator.add(Reshape(target_shape=(self.img_height, self.img_width, self.img_channel,)))

        latent_vector = Input(shape=(self.latent_size,))
        generated_img = generator(latent_vector)

        return Model(latent_vector, generated_img)

    def _build_discriminator(self):
        img_size = int(self.img_height * self.img_width * self.img_channel)

        discriminator = Sequential()
        discriminator.add(Reshape(target_shape=(img_size,)))
        discriminator.add(Dense(units=512, activation="relu", input_dim=img_size))
        discriminator.add(Dense(units=256, activation="relu", input_dim=img_size))
        discriminator.add(Dense(units=1, activation="sigmoid"))

        input_img = Input(shape=(self.img_height, self.img_width, self.img_channel,))
        validity = discriminator(input_img)

        return Model(input_img, validity)
