import numpy as np
import matplotlib.pyplot as plt


class KohonenTrainer:

    def __init__(self, n_max_iterations, width, height):
        self.n_max_iterations = n_max_iterations
        self.width = width
        self.height = height

        self.radius_0 = max(self.width, self.height) / 2
        self.lr_0 = 0.1
        self.lambda_val = self.n_max_iterations / np.log(self.radius_0)

    def calculate_bmu(self, weights, row_vector):
        bmu = np.argmin(np.sum((weights - row_vector) ** 2, axis=2))
        return np.unravel_index(bmu, (self.width, self.height))

    def influence(self, dist, radius_t):
        return np.exp(-(dist ** 2) / (2 * (radius_t ** 2)))

    def euclidean_distance(self, point1, point2):
        return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

    def neighbour_radius(self, t):
        return self.radius_0 * np.exp(-t / self.lambda_val)

    def learning_rate(self, t):
        return self.lr_0 * np.exp(-t / self.lambda_val)

    def train(self, input_data):
        weights = np.random.random((self.width, self.height, 3))

        for t in range(self.n_max_iterations):
            if t % 100 == 0:
                print(f"Iteration: {t}/{self.n_max_iterations}")

            lr_t = self.learning_rate(t)
            radius_t = self.neighbour_radius(t)

            for vt in input_data:
                bmu_x, bmu_y = self.calculate_bmu(weights, vt)

                # Bounding box around the neighbourhood radius
                window_top_left = [max(bmu_x - int(radius_t), 0), max(bmu_y - int(radius_t), 0)]
                window_bottom_right = [min(bmu_x + int(radius_t), self.width), min(bmu_y + int(radius_t), self.height)]

                for x in range(window_top_left[0], window_bottom_right[0]):
                    for y in range(window_top_left[1], window_bottom_right[1]):

                        distance = self.euclidean_distance([x, y], [bmu_x, bmu_y])
                        if distance <= radius_t:
                            influence_t = self.influence(distance, radius_t)
                            weights[x, y] += lr_t * influence_t * (vt - weights[x, y])
        return weights


if __name__ == "__main__":
    input_dataset = np.random.random((10, 3))

    kohonen_trainer = KohonenTrainer(100, 10, 10)
    image_data_100 = kohonen_trainer.train(input_dataset)
    plt.imsave('100.png', image_data_100)

    kohonen_trainer = KohonenTrainer(1000, 100, 100)
    image_data_1000 = kohonen_trainer.train(input_dataset)
    plt.imsave('1000.png', image_data_1000)
