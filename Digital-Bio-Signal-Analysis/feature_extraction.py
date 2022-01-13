import numpy as np
from scipy import stats, signal


class MaxPeak:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        peaks = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            peak = np.max(segment[start: end, :], axis=0)
            peak = np.expand_dims(peak, axis=0)
            if i == 0:
                peaks = peak
                continue
            peaks = np.vstack((peaks, peak))
        return peaks


class Mean:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        means = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            mean = np.mean(segment[start: end, :], axis=0)
            mean = np.expand_dims(mean, axis=0)
            if i == 0:
                means = mean
                continue
            means = np.vstack((means, mean))
        return means


class Variance:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        vars = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            var = np.var(segment[start: end, :], axis=0)
            var = np.expand_dims(var, axis=0)
            if i == 0:
                vars = var
                continue
            vars = np.vstack((vars, var))
        return vars


class StandardDeviation:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        stds = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            std = np.std(segment[start: end, :], axis=0)
            std = np.expand_dims(std, axis=0)
            if i == 0:
                stds = std
                continue
            stds = np.vstack((stds, std))
        return stds


class Skewness:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        skews = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            skew = stats.skew(segment[start: end, :], axis=0)
            skew = np.expand_dims(skew, axis=0)
            if i == 0:
                skews = skew
                continue
            skews = np.vstack((skews, skew))
        return skews


class Kurtosis:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        kurts = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            kurt = stats.kurtosis(segment[start: end, :], axis=0)
            kurt = np.expand_dims(kurt, axis=0)
            if i == 0:
                kurts = kurt
                continue
            kurts = np.vstack((kurts, kurt))
        return kurts


class RootMeanSquare:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        rmss = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            rms = np.sqrt(np.mean(segment[start: end, :] ** 2, axis=0))
            rms = np.expand_dims(rms, axis=0)
            if i == 0:
                rmss = rms
                continue
            rmss = np.vstack((rmss, rms))
        return rmss


class WaveformLength:
    @classmethod
    def apply(cls, segment, window_size=50):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        wls = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            wl = np.sum(np.abs(np.diff(segment[start: end, :], axis=0)), axis=0)
            wl = np.expand_dims(wl, axis=0)
            if i == 0:
                wls = wl
                continue
            wls = np.vstack((wls, wl))
        return wls


class WillisonAmplitude:
    @classmethod
    def apply(cls, segment, window_size=50, eps=0.5):
        """
        @param segment: signal segment
        @param window_size: window size. Default: 50.
        @param eps: threshold value. Default: 0.5.
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        if eps > 1:
            eps = 1
        elif eps < 0:
            eps = 0
        L = segment.shape[0]
        wamps = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            wl = np.abs(np.diff(segment[start: end, :], axis=0))
            mask = wl[wl > eps]
            wl[mask is True] = 1
            wl[mask is False] = 0
            wamp = np.sum(wl, axis=0)
            wamp = np.expand_dims(wamp, axis=0)
            if i == 0:
                wamps = wamp
                continue
            wamps = np.vstack((wamps, wamp))
        return wamps


class PSD:
    """
    Power Spectral Density
    """

    @classmethod
    def apply(cls, segment, window_size=50, fs=512, average='mean'):
        """
        @param segment: signal segment.
        @param window_size: window size. Default: 50.
        @param fs: sampling rate. Default: 512.
        @param average: 'mean' or 'median'. Default: 'mean'
        """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        psds = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size < L) else L
            freq, power = signal.welch(segment[start: end, :],
                                       fs=fs,
                                       scaling='density',
                                       detrend=False,
                                       nfft=window_size,
                                       average=average,
                                       nperseg=window_size,
                                       return_onesided=True,
                                       axis=0)
            if i == 0:
                psds = power
                continue
            psds = np.vstack((psds, power))
        return psds


class STFT:
    """
    Short Time Frequency Transform
    """

    @classmethod
    def apply(cls, segment, w, H=8):
        """
        @param segment: signal segment.
        @param w: window function.
        @param H: Hopsize. Default: 8.
        """
        N = len(w)
        num_channels = segment.shape[1]
        L = segment.shape[0]
        M = np.floor((L - N) / H).astype(int) + 1
        stfts = np.array([])
        for i in range(num_channels):
            stft = np.zeros((N, M), dtype='complex')
            for m in range(M):
                stft_win = segment[:, i][m * H:m * H + N] * w
                stft_win = np.fft.fft(stft_win)
                stft[:, m] = stft_win
            K = 1 + N // 2
            stft = stft[0: K, :]
            if i == 0:
                stfts = stft
                continue
            stfts = np.dstack((stfts, stft))
        # stfts = np.transpose(stfts, (2, 0, 1))
        return stfts


class CWT:
    """
    Continuous Wavelet Transform
    """

    @classmethod
    def apply(cls, segment, window_size=50, widths=50):
        """
                @param segment: signal segment.
                @param window_size: window size. Default: 50.
                @param widths: number of scale. Default: 50.
                """
        assert 0 <= window_size <= segment.shape[0], 'window_size={} is invalid!'.format(window_size)
        L = segment.shape[0]
        num_channels = segment.shape[1]
        if isinstance(widths, int):
            widths = np.arange(1, widths + 1)
        cwts = np.array([])
        for i in range(0, L, window_size):
            start = i
            end = i + window_size if (i + window_size <= L) else 2 * i + window_size - L
            cwt = np.array([])
            for j in range(num_channels):
                cwt_ = signal.cwt(segment[start: end, j], signal.ricker, widths, dtype=np.float64)
                if j == 0:
                    cwt = cwt_
                    continue
                cwt = np.dstack((cwt, cwt_))
            if i == 0:
                cwts = cwt
            cwts = np.dstack((cwts, cwt))
        return cwts
