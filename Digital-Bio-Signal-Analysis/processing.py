import numpy as np
from scipy import signal, sparse
from filter import BandpassFilter1D, LowpassFilter1D


class Envelope1D:
    """
    Compute electromyography amplitude given the cleaned respiration signal,
    done by calculating the linear envelope of the signal.
    """

    @classmethod
    def apply(cls, x, low_fs, high_fs, env_fs, fs=1000):
        """
        @param x: 1D signal
        @param low_fs: low cutoff frequency (Hz).
        @param high_fs: high cutoff frequency (Hz).
        @param env_fs: envelop cutoff frequency (Hz).
        @param fs: sample rate of input signal (Hz). Default: 1000.
        """
        tkeo = cls._tkeo_operator(x)
        envelope = cls._linear_envelope(tkeo, low_fs, high_fs, env_fs, fs)
        return envelope

    @classmethod
    def _tkeo_operator(cls, x):
        """
        Calculates the Teager–Kaiser Energy operator to improve onset detection,
        described by Marcos Duarte at
        https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb.
        ---
        @param x: 1D signal
        """
        tkeo = x.copy()
        # Teager–Kaiser Energy operator
        tkeo[1: -1] = x[1: -1] * x[1: -1] - x[: -2] * x[2:]
        # correct the data in the extremities
        tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]
        return tkeo

    @classmethod
    def _linear_envelope(cls, x, low_fs, high_fs, env_fs, fs):
        """
        Calculate the linear envelope of a signal.
        This function implements a 2nd-order Butterworth filter with zero lag, described by Marcos Duarte
        at <https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb>.
        ---
        @param x: 1D signal.
        @param low_fs: low cutoff frequency (Hz).
        @param high_fs: high cutoff frequency (Hz).
        @param env_fs: envelop cutoff frequency (Hz).
        @param fs: sample rate of input signal (Hz).
        """
        x_filtered = BandpassFilter1D.apply(x, low_fs, high_fs, order=2, fs=fs)
        # visualize envelope
        envelope = abs(x_filtered)
        envelope = LowpassFilter1D.apply(envelope, low_fs=env_fs, order=2, fs=fs)
        return envelope


class MeanShift1D:
    """
    Shift 1D signal to zero-mean
    """

    @classmethod
    def apply(cls, x):
        """
        @param x: 1D signal
        """
        return x - np.mean(x)


class Normalize1D:
    """
    Normalization for 1D signal vector by shifting and rescaling the data to [0,1]
    """

    @classmethod
    def apply(cls, x, norm_type='min_max'):
        """
        Normalize data either by shifting and rescaling the data to [0,1].
        ---
        @param x: 1D signal
        @param norm_type: normalization method: ``min_max`` or ``mean_std``. Default: ``min_max``
        """
        if norm_type == 'min_max':
            x_normed = cls._norm_min_max(x)
        elif norm_type == 'mean_std':
            x_normed = cls._norm_mean_std(x)
        else:
            raise NotImplementedError('Normalization method ``{}`` is not supported!'.format(norm_type))
        # x_normed = x_normed - np.mean(x_normed)
        return x_normed

    @classmethod
    def _norm_min_max(cls, x):
        """
        @param x: 1D signal
        """
        eps = 1e-6
        return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)

    @classmethod
    def _norm_mean_std(cls, x):
        """
        @param x: 1D signal
        """
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / std


class Detrend1D:
    """
    Detrend method for 1D signal
    """

    @classmethod
    def apply(cls, x, detrend_type='locreg', window_size=1500, step_size=20, regularize=500):
        """
        @param x: 1D signal
        @param detrend_type: detrending type: ``locreg``, ``tarvainen``, ``loess``, ``polynomial``. Default: ``polynominal``
        @param window_size: window size. Usually using 1.5 sampling rate. Default: 1500.
        @param step_size: step size to sliding windows. Default: 20.
        @param regularize: regularization parameter. Default: 500.
        """
        if detrend_type == 'locreg':
            return cls._detrend_locreg(x, window_size, step_size)
        elif detrend_type == 'tarvainen':
            return cls._detrend_tarvainen(x, regularize=regularize)
        else:
            raise NotImplementedError('Detrend method ``{}`` is not supported!'.format(detrend_type))

    @classmethod
    def _detrend_locreg(cls, x, windows=1500, step_size=20):
        """
        Detrend method for 1D signal using Local linear Regression method.
        ---
        @param x: 1D signal
        @param windows: window size. Usually using 1.5 sampling rate. Default: 1500.
        @param step_size: step size to sliding windows. Default: 20.
        """
        length = len(x)
        # sanity checks
        windows = int(windows)
        step_size = int(step_size)
        y_line = np.zeros((length, 1))
        norm = np.zeros((length, 1))
        num_windows = int(np.ceil((length - windows) / step_size))
        y_fit = np.zeros((num_windows, windows))
        xwt = (np.arange(1, windows + 1) - windows / 2) / (windows / 2)
        wt = np.power(1 - np.power(np.absolute(xwt), 3), 3)
        a, b = 0, 0
        for i in range(0, num_windows):
            t_seg = x[(step_size * i): (step_size * i + windows)]
            y1 = np.mean(t_seg)
            y2 = np.mean(np.multiply(np.arange(1, windows + 1), t_seg)) * (2 / (windows + 1))
            a = np.multiply(np.subtract(y2, y1), 6 / (windows - 1))
            b = np.subtract(y1, a * (windows + 1) / 2)
            y_fit[i, :] = np.multiply(np.arange(1, windows + 1), a) + b
            y_line[(i * step_size): (i * step_size + windows)] = \
                y_line[(i * step_size): (i * step_size + windows)] + \
                np.reshape(np.multiply(y_fit[i, :], wt), (windows, 1))
            norm[(i * step_size): (i * step_size + windows)] = \
                norm[(i * step_size): (i * step_size + windows)] + \
                np.reshape(wt, (windows, 1))
        above_norm = np.where(norm[:, 0] > 0)
        y_line[above_norm] = y_line[above_norm] / norm[above_norm]
        idx = (num_windows - 1) * step_size + windows - 1
        num_points = length - idx + 1
        y_line[idx - 1:] = np.reshape((np.multiply(np.arange(windows + 1, windows + num_points + 1), a) + b),
                                      (num_points, 1))
        detrended = x - y_line[:, 0]
        return detrended

    @classmethod
    def _detrend_tarvainen(cls, x, regularize=500):
        """
        Method by Tarvainen et al., 2002.
        - Tarvainen, M. P., Ranta-Aho, P. O., & Karjalainen, P. A. (2002). An advanced detrending method
        with application to HRV analysis. IEEE Transactions on Biomedical Engineering, 49(2), 172-175.
        ---
        @param x: 1D signal
        @param regularize: regularization parameter. Default: 500.
        """
        N = len(x)
        I = np.eye(N)
        B = np.dot(np.ones(N - 2, 1)), np.array([[1, -2, 1]])
        D_2 = sparse.dia_matrix((B.T, [0, 1, 2]), shape=(N - 2, N))
        inv = np.linalg.inv(I + regularize ** 2 * D_2.T @ D_2)
        z_stat = (I - inv) @ signal
        trend = np.squeeze(np.asarray(signal - z_stat))
        detrended = np.array(x) - trend
        return detrended


class Resample1D:
    """
    Resample method for 1D signal from original sample rate to new sample rate
    """

    @classmethod
    def apply(cls, x, raw_fs, target_fs):
        """
        @param x: 1D signal
        @param raw_fs: original sample rate
        @param target_fs: new sample rate
        """
        num_samples = int((x.shape[0] / raw_fs) * target_fs)
        return signal.resample(x, num=num_samples)
