from PIL import Image
import math
import numpy as np

class KMeansCompression:
    
    # Initialize the class with the image at the path and the number of clusters (k)
    def __init__(self, image_path, k):
        self.img = Image.open(image_path)
        self.k = k
        self.prev_centroids = [(0, 0, 0) for _ in range(k)]
        self.centroids = self.assign_random_centroids()
        self.centroid_assignments = np.zeros((self.img.width, self.img.height))

    # Assign k random centroids from the pixels in the image
    def assign_random_centroids(self):
        centroids = []
        for _ in range(self.k):
            rand_coord = (np.random.randint(0, self.img.width), np.random.randint(0, self.img.height))
            centroids.append(self.img.getpixel(rand_coord))
        
        return centroids

    # Assign each pixel in the image to the nearest centroid
    def assign_pixels_to_centroids(self):
        for i in range(self.img.width):
            for j in range(self.img.height):
                curr_pixel = self.img.getpixel((i, j))
                diffs = [self.compute_distance(curr_pixel, centroid) for centroid in self.centroids]
                self.centroid_assignments[i][j] = diffs.index(min(diffs))
                

    # Recompute the position of each centroid based on the mean of all pixels assigned to it
    def recompute_centroids(self):
        self.prev_centroids = self.centroids
        total = {centroid: np.array([0, 0, 0, 0]) for centroid in self.centroids}
        for i in range(self.img.width):
            for j in range(self.img.height):
                temp_centroid = self.centroids[int(self.centroid_assignments[i][j])]
                total[temp_centroid] += np.append(np.array(self.img.getpixel((i, j))), 1)
                
        for centroid in total:
            total[centroid] = total[centroid] / total[centroid][3]
            
        self.centroids = [(total[centroid][0], total[centroid][1], total[centroid][2]) for centroid in total]

    # Check if the algorithm has converged (i.e., if the centroids have stopped moving)
    def convergence_check(self):
        return all([all([abs(self.prev_centroids[i][j] - self.centroids[i][j]) < 20 for j in range(3)]) for i in range(len(self.centroids))])

    # Replace each pixel in the image with the color of its corresponding centroid
    def compress(self):
        for i in range(self.img.width):
            for j in range(self.img.height):
                centroid = self.centroids[int(self.centroid_assignments[i][j])]
                centroid = (int(centroid[0]), int(centroid[1]), int(centroid[2]))
                self.img.putpixel((i, j), centroid)

    # Run the K-means algorithm until convergence
    def run(self):
        while not self.convergence_check():
            self.assign_pixels_to_centroids()
            self.recompute_centroids()
        self.compress()

    # Compute the Euclidean distance between a pixel and a centroid
    def compute_distance(self, pixel, centroid):
        return math.sqrt(sum([(pixel[i] - centroid[i]) ** 2 for i in range(len(pixel))]))
    
        
k_means = KMeansCompression("test.jpg", k=10)
k_means.run()
k_means.img.save("compressed_test.jpg")
