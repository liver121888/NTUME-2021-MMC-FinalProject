import numpy as np


class Segment1D:
    """
    Segment 1D signal
    """

    @classmethod
    def apply(cls, x, window_size=200, step_size=50):
        """
        @param x: 1D signal
        @param window_size: window size.Default: 200.
        @param step_size: step size. Default: 50
        """
        # length = window_size - int(len(x) % window_size)
        # pad = np.array([0] * length)
        # x_padded = np.concatenate([x, pad])
        segments = [x[i:i + window_size] for i in range(0, x.shape[0], step_size) if (i + window_size < x.shape[0])]
        return segments


class SegmentND:
    """
    Segment multi-channels signal
    """

    @classmethod
    def apply(cls, x, window_size=200, step_size=50):
        """
        @param x: 2D signal
        @param window_size: window size. Default: 200.
        @param step_size: step size. Default: 50
        """
        # length = (x.shape[1] - step_size + window_size) % window_size
        # make sure there are an even number of windows before stride tricks
        # pad = np.zeros((length, x.shape[1]))
        # x_padded = np.vstack([x, pad])
        segments = [x[i:i + window_size, :] for i in range(0, x.shape[0], step_size) if (i + window_size < x.shape[0])]
        return segments
