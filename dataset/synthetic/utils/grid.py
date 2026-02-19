from .geometry import point_in_patch

class NoFreeCellsError(RuntimeError):
    pass

class Grid:
    def __init__(self, x_bounds, y_bounds, resolution):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.resolution = resolution
        self.cols = int((self.x_max - self.x_min) / resolution)
        self.rows = int((self.y_max - self.y_min) / resolution)

    def to_idx(self, pt):
        return (
            int((pt[0] - self.x_min) / self.resolution),
            int((pt[1] - self.y_min) / self.resolution),
        )

    def to_coord(self, idx):
        return (
            self.x_min + idx[0] * self.resolution + self.resolution / 2,
            self.y_min + idx[1] * self.resolution + self.resolution / 2,
        )

    def in_bounds(self, idx):
        return 0 <= idx[0] < self.cols and 0 <= idx[1] < self.rows

    def passable(self, idx, pads):
        pt = self.to_coord(idx)
        return all(not point_in_patch(pt, p) for p in pads)
