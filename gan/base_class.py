""" Module that contains GAN abstract base class """

import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam


class BaseClass(object):
    def __init__(self, img_shape=(28, 28, 1), latent_size=100):
        super(BaseClass, self).__init__()

        np.random.seed(seed=1)

        if not isinstance(img_shape, tuple):
            raise TypeError("Type of argument '{0} must be '{1}'.".format("img_shape", "tuple"))

        self.img_height = img_shape[0]
        self.img_width = img_shape[1]
        self.img_channel = img_shape[2]
        self.latent_size = latent_size

        # Generator
        self._generator = self._build_generator()

        # Discriminator
        self._discriminator = self._build_discriminator()
        self._discriminator.trainable = True
        self._discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss="binary_crossentropy")

        # Discriminator(Generator)
        self._stacked = self._build_stacked()
        self._discriminator.trainable = False
        self._stacked.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss="binary_crossentropy")

    def _build_generator(self):
        raise NotImplementedError

    def _build_discriminator(self):
        raise NotImplementedError

    def _build_stacked(self):
        latent_vector = Input(shape=(self.latent_size,))
        generated_img = self._generator(latent_vector)
        validity = self._discriminator(generated_img)

        return Model(latent_vector, validity)

    def _run_ga(self, generations=32, population_size=256):
        latent_vector = np.random.normal(size=(population_size, self.latent_size))
        population = self._generator.predict(latent_vector)

        validity = self._discriminator.predict(population).flatten()
        fitness = validity * 10
        sum_fitness = np.sum(fitness)

        for _ in range(generations):
            # selection
            parents_idx = np.random.randint(0, len(population), size=2)
            for idx in range(2):
                temp = 0.0
                point = np.random.randint(0, int(sum_fitness))
                for j in range(len(population)):
                    temp += fitness[j]
                    if temp >= point:
                        parents_idx[idx] = j
                        break
            parents = np.take(population, parents_idx, axis=0)

            # crossover
            # block uniform crossover
            # row = np.random.randint(1, self.img_height)
            # col = np.random.randint(1, self.img_width)
            #
            # up = np.concatenate((parents[0][0:row, 0:col], parents[1][0:row, col:self.img_width]), axis=1)
            # down = np.concatenate(
            #     (parents[1][row:self.img_height, 0:col], parents[0][row:self.img_height, col:self.img_width]), axis=1
            # )
            # offspring = np.array([np.concatenate((up, down), axis=0)])

            # arithmetic recombination
            w = np.random.random(size=1)
            offspring = np.array([parents[0] * w + parents[1] * (1.0 - w)])

            # mutation
            # mutation rate: 0.1
            r = np.random.randint(0, 100)
            if r < 10:
                offspring += np.random.normal(loc=0.0, scale=0.05, size=(28, 28, 1))

            # replacement
            # worst individual replacement
            worst_idx = np.argmin(fitness)
            validity = self._discriminator.predict(offspring).flatten()
            offspring_fitness = validity * 10
            if fitness[worst_idx] < offspring_fitness[0]:
                fitness = np.append(np.delete(fitness, worst_idx), offspring_fitness[0])
                population = np.append(np.delete(population, worst_idx, axis=0), offspring, axis=0)
                sum_fitness -= fitness[worst_idx]
                sum_fitness += offspring_fitness[0]

        return population

    def _train_on_batch(self, images, batch_size, with_ga=True):
        for i in range(0, len(images), batch_size):
            batch_img = images[i:i + batch_size]

            real = np.ones((len(batch_img), 1))
            fake = np.zeros((len(batch_img), 1))

            # train Discriminator
            self._discriminator.trainable = True
            if with_ga:
                generated_img = self._run_ga(generations=32, population_size=len(batch_img))
            else:
                latent_vector = np.random.normal(size=(len(batch_img), self.latent_size))
                generated_img = self._generator.predict(latent_vector)
            self._discriminator.train_on_batch(x=batch_img, y=real)
            self._discriminator.train_on_batch(x=generated_img, y=fake)

            # train Generator
            self._discriminator.trainable = False
            latent_vector = np.random.normal(size=(len(batch_img), self.latent_size))
            self._stacked.train_on_batch(x=latent_vector, y=real)

    def train(self, images, epochs=100, batch_size=256, with_ga=True, debug_mode=False):
        """
        Learn the network with the specified image.

        Parameters
        ----------
        images : numpy.ndarray
            train images
        epochs : int
        batch_size : int
        with_ga : bool
        debug_mode : bool
            whether to output training information to the screen
        """
        if debug_mode:
            print("Generator :")
            self._generator.summary()
            print("Discriminator :")
            self._discriminator.summary()

        for epoch in range(epochs):
            if debug_mode and (epoch == 0):
                print("{0:<6} | {1:<9} | {2:<8} | {3:<8}".format("Epoch", "Time (s)", "D Loss", "G Loss"))

            start_time = time.time()
            self._train_on_batch(images.copy(), batch_size, with_ga)
            run_time = time.time() - start_time

            if debug_mode:
                real = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))

                latent_vector = np.random.normal(size=(batch_size, self.latent_size))
                batch_img = np.take(images, np.random.randint(0, len(images), size=batch_size), axis=0)
                generated_img = self._generator.predict(latent_vector)

                d_loss_real = self._discriminator.test_on_batch(x=batch_img, y=real)
                d_loss_fake = self._discriminator.test_on_batch(x=generated_img, y=fake)
                d_loss = np.mean([d_loss_real, d_loss_fake])

                g_loss = self._stacked.test_on_batch(x=latent_vector, y=real)

                print("{0:>6} | {1:>8.2}s | {2:>8.6f} | {3:>8.6f}".format(epoch + 1, run_time, d_loss, g_loss))

    def generate(self, sample_num, save_path=None):
        """
        Get the specified number of samples from the trained Generator.

        Parameters
        ----------
        sample_num : int
            number of samples

        save_path : str
            path to save sample images

        Returns
        -------
        numpy.ndarray
        """
        row = math.ceil(math.sqrt(sample_num))
        col = math.ceil(sample_num / row)

        latent_vector = np.random.normal(size=(sample_num, self.latent_size))
        samples = self._generator.predict(latent_vector)
        samples = samples.reshape((sample_num, self.img_height, self.img_width, self.img_channel))

        if save_path:
            pardir = os.path.dirname(save_path)
            if pardir and (not os.path.exists(pardir)):
                os.makedirs(pardir)

            converted_samples = ((samples * 127.5) + 127.5).astype("int32")

            fig, axs = plt.subplots(row, col)
            fig.tight_layout()
            cnt = 0
            for i in range(row):
                for j in range(col):
                    if self.img_channel == 1:
                        axs[i, j].imshow(converted_samples[cnt, :, :, 0], cmap="gray")
                    elif self.img_channel == 3:
                        axs[i, j].imshow(converted_samples[cnt, :, :, :])
                    axs[i, j].axis("off")
                    cnt += 1
                    if cnt == sample_num:
                        break
            fig.savefig(save_path)
            plt.close()

        return samples

    def discriminate(self, images):
        """
        Parameters
        ----------
        images : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return self._discriminator.predict(images)

    def save_model(self, dir_path):
        """
        Save the trained model to the specified path.

        Parameters
        ----------
        dir_path : str
            directory path to save model

        Returns
        -------
        str
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(os.path.join(dir_path, "variables.json"), "w") as fp:
            var = {"img_height": self.img_height,
                   "img_width": self.img_width,
                   "img_channel": self.img_channel,
                   "latent_size": self.latent_size}
            json.dump(var, fp)

        self._generator.save(os.path.join(dir_path, "generator.h5"))
        self._discriminator.save(os.path.join(dir_path, "discriminator.h5"))

        print("The model has been saved to: {0}/".format(os.path.abspath(dir_path)))
        return os.path.abspath(dir_path)

    def load_model(self, dir_path):
        """
        Load the trained model from the specified path.

        Parameters
        ----------
        dir_path : str
            directory path to load model

        Returns
        -------
        bool
        """
        with open(os.path.join(dir_path, "variables.json"), "r") as fp:
            var = json.load(fp)

        self.img_height = var["img_height"]
        self.img_width = var["img_width"]
        self.img_channel = var["img_channel"]
        self.latent_size = var["latent_size"]

        self._generator = load_model(os.path.join(dir_path, "generator.h5"))
        self._discriminator = load_model(os.path.join(dir_path, "discriminator.h5"))

        print("The model was loaded from: {0}/".format(os.path.abspath(dir_path)))
        return True
