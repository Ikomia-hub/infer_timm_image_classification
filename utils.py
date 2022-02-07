import numpy as np

def by_batch(func):
    def f(input):
        if isinstance(input,np.ndarray):
            yield func(input,0,0)
        if isinstance(input,list):
            for imgxy in input:
                yield func(*imgxy)
    return f

def polygon2bbox(pts):
    x = np.min(pts[:, 0])
    y = np.min(pts[:, 1])
    w = np.max(pts[:, 0]) - x
    h = np.max(pts[:, 1]) - y
    return [int(x), int(y), int(w), int(h)]