import numpy as np


class GanUtils:
    def __init__(self, latent_length, batch_size, dataset, generator):
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.generator = generator

    def get_latent_examples(self, n):
        latent_examples = np.random.randn(n, self.latent_length)
        return latent_examples

    def get_latent_dataset(self, n=None):
        if n is None:
            n = self.batch_size
        x = self.get_latent_examples(n)
        y = np.ones((n,))
        return x, y

    def get_fake_data(self, n):
        latent_examples = self.get_latent_examples(n)
        x = self.generator.predict(latent_examples)
        y = np.zeros((len(x),))
        return x, y

    def get_sampled_data(self, n):
        indexes = np.random.choice(self.dataset_size, n)
        x = self.dataset[indexes]
        y = np.ones((len(x),))
        return x, y

    def get_combined_dataset(self, n_fake, n_real):
        x_fake, y_fake = self.get_fake_data(n_fake)
        x_real, y_real = self.get_sampled_data(n_real)

        x = np.concatenate([x_real, x_fake])
        y = np.concatenate([y_real, y_fake])
        return x, y

    def get_equal_combined_dataset(self):
        n_fake = int(self.batch_size / 2)
        n_real = int(self.batch_size / 2)
        return self.get_combined_dataset(n_fake, n_real)
