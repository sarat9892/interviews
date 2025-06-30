import matplotlib.pyplot as plt

import numpy as np
import time


def trainSOM(input_data, n_max_iterations, width, height):
    radius_0 = max(width, height) / 2
    lr_0 = 0.1
    weights = np.random.random((width, height, 3))  # Shape: (width, height, 3)
    lambda_val = n_max_iterations / np.log(radius_0)  # Shape: (0) Scalar

    # Precompute grid coordinates
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height),
                                     indexing='ij')  # Shape: (width, height), (width height)
    grid = np.stack([x_coords, y_coords], axis=-1)  # Shape: (width, height, 2)

    for t in range(n_max_iterations):
        if t % 10 == 0:
            print(f"Iteration: {t}")

        radius_t = radius_0 * np.exp(-t / lambda_val)  # Shape: (0) Scalar
        lr_t = lr_0 * np.exp(-t / lambda_val)  # Shape: (0) Scalar

        for vt in input_data:
            # Compute distance from vt to each neuron
            distances = np.linalg.norm(weights - vt, axis=2)  # Shape: (width, height, 3)
            bmu_index = np.unravel_index(np.argmin(distances), (width, height))
            bmu_vector = np.array(bmu_index)  # Shape: (2,)

            # Compute neighborhood function
            diff = grid - bmu_vector  # Shape: (width, height, 2)
            dist_sq = np.sum(diff ** 2, axis=2)  # Shape: (width, height)
            influence_t = np.exp(-dist_sq / (2 * radius_t ** 2))  # Shape: (width, height)

            # Update weights using broadcasting
            influence = lr_t * influence_t[..., np.newaxis]  # Shape: (width, height, 1)
            weights += influence * (vt - weights)

    return weights


if __name__ == "__main__":
    data_count = [10, 100]
    map_size = [10, 100, 1000]
    iterations = [10, 100]

    data_dimensions = 3

    output_dict = {}
    output_names = {}

    count = 0

    total_start = time.time()

    for c in data_count:
        for d in map_size:
            for i in iterations:

                print("\n")
                print(f"Row Count: {c}, Map Size: {d}, Iteration: {i}")

                input_data = np.random.random((c, data_dimensions))

                image_key = f"image_{i}_{d}x{d}"
                time_key = f"time_{i}_{d}x{d}"

                output_names[count] = [image_key, time_key, c, d, i]

                start = time.time()
                output_dict[output_names[count][0]] = trainSOM(input_data, i, d, d)
                end = time.time()
                output_dict[output_names[count][1]] = end - start

                count += 1

    total_end = time.time()

    for x in range(count):
        image = output_dict[output_names[x][0]]
        time_taken = round(output_dict[output_names[x][1]], 1)
        row = output_names[x][2]
        size = output_names[x][3]
        iter_count = output_names[x][4]


        name = f"{row}Rows_{size}x{size}Map_{iter_count}Iters_{time_taken}Time"
        file_name = f"output_maps/{name}.png"

        plt.imsave(file_name, image)

    total_time = total_start - total_end

    print(f"Total Time Taken: {total_time}")
