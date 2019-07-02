import numpy as np
from PIL import Image


class PatchMatch:

    def __init__(self, image_source, image_target, patch_size):
        if patch_size % 2 == 0:
            raise Exception("patch size should be odd")

        self.image_source = image_source
        self.image_target = image_target

        self.source_height = image_source.shape[0]
        self.source_width = image_source.shape[1]

        self.target_height = image_target.shape[0]
        self.target_width = image_target.shape[1]


        self.patch_size = patch_size

        self.initialization()



    def run(self, num_iter, callback):
        for iter in range(1, num_iter + 1):

            for y in range(0, self.source_height):
                for x in range(0, self.source_width):
                    self.propagation(x, y, iter)
                    self.random_search(x, y)

            callback(iter)


    def reconstruct(self):
        # reconstructed = np.zeros([self.source_height, self.source_width, 3])
        reconstructed = np.zeros_like(self.image_source)
        for y in range(0, self.source_height):
            for x in range(0, self.source_width):
                reconstructed[y, x, :] = self.image_target[self.F[y, x][0], self.F[y, x][1], :]

        return reconstructed


    def initialization(self):
        p = self.patch_size // 2

        self.padding_img = np.ones([self.source_height + 2*p, self.source_width + 2*p, 3]) * np.nan
        self.padding_img[p:self.source_height + p, p: self.source_width + p] = self.image_source

        random_by_target_height = np.random.randint(p, self.target_height-p, [self.source_height, self.source_width])
        random_by_target_width= np.random.randint(p, self.target_width-p, [self.source_height, self.source_width])

        self.F = np.zeros([self.source_height, self.source_width], dtype=object)
        self.D = np.zeros([self.source_height, self.source_width])

        for y in range(0, self.source_height):
            for x in range(0, self.source_width):
                f_yx = np.array([random_by_target_height[y, x], random_by_target_width[y, x]], dtype=np.int32)
                self.F[y, x] = f_yx
                self.D[y, x] = self._distance(x, y, f_yx[1], f_yx[0])


    def propagation(self, x, y, iter):
        if iter % 2 == 0:
            self._propagation_even_iter(x, y)
        else:
            self._propagation_odd_iter(x, y)

    def random_search(self, x, y, k=4, alpha = 0.5):

        p = self.patch_size // 2

        search_height = self.target_height * alpha ** k
        search_width = self.target_width * alpha ** k

        target_x = self.F[y, x][1]
        target_y = self.F[y, x][0]

        while search_height > 1 and search_width > 1:
            search_min_by_height = max(target_y - search_height, p)
            search_max_by_height = min(target_y + search_height, self.target_height - p)
            random_target_y = np.random.randint(search_min_by_height, search_max_by_height)

            search_min_by_width = max(target_x - search_width, p)
            search_max_by_width = min(target_x + search_width, self.target_width - p)
            random_target_x = np.random.randint(search_min_by_width, search_max_by_width)


            d = self._distance(x, y, random_target_x, random_target_y)
            if d < self.D[y, x]:
                self.D[y, x] = d
                self.F[y, x] = np.array([random_target_y, random_target_x])

            search_height = self.target_height * alpha ** k
            search_width = self.target_width * alpha ** k

            k += 1

    def _distance(self, x, y, f_x, f_y):
        p = self.patch_size // 2

        patch_source = self.padding_img[y : y + self.patch_size, x : x + self.patch_size, :]
        patch_target = self.image_target[f_y - p: f_y + p + 1, f_x - p: f_x + p + 1, :]
        diff = patch_target - patch_source

        count = np.sum(1 - np.isnan(diff))

        d = np.sum(np.square(np.nan_to_num(diff))) / count

        return d

    def _propagation_odd_iter(self, x, y):
        d_left = self.D[max(y - 1, 0), x]
        d_up = self.D[y, max(x - 1, 0)]
        d = self.D[y, x]

        idx = np.argmin(np.array([d, d_left, d_up]))

        if idx == 1:
            self.F[y, x] = self.F[max(y - 1, 0), x]
            self.D[y, x] = self._distance(x, y, self.F[y, x][1], self.F[y, x][0])
        elif idx == 2:
            self.F[y, x] = self.F[y, max(x - 1, 0)]
            self.D[y, x] = self._distance(x, y, self.F[y, x][1], self.F[y, x][0])


    def _propagation_even_iter(self, x, y):
        d_right = self.D[min(y + 1, self.source_height - 1), x]
        d_down = self.D[y, min(x + 1, self.source_width - 1)]
        d = self.D[y, x]
        idx = np.argmin(np.array([d, d_right, d_down]))
        if idx == 1:
            self.F[y, x] = self.F[min(y + 1, self.source_height - 1), x]
            self.D[y, x] = self._distance(x, y, self.F[y, x][1], self.F[y, x][0])
        if idx == 2:
            self.F[y, x] = self.F[y, min(x + 1, self.source_width - 1)]
            self.D[y, x] = self._distance(x, y, self.F[y, x][1], self.F[y, x][0])





