from scipy import signal


class NotchFilter1D:
    """
    Notch filter (e.g., 50 or 60 Hz) on 1D signal.
    """
    @classmethod
    def apply(cls, x, notch_fs=50, Q=30, fs=512):
        """
        @param x: 1D signal
        @param notch_fs: notch cutoff frequency (Hz). Default: 50
        @param Q: Quality factor. Default: 30
        @param fs: sample rate of input signal (Hz). Default: 512
        """
        # get cutoff stop-band
        nyq = 0.5 * fs
        notchcut = notch_fs / nyq
        # design notch filter
        b, a = signal.iirnotch(notchcut, Q)
        # filter signal
        x_filtered = signal.filtfilt(b, a, x)
        return x_filtered


class BandpassFilter1D:
    """
    Butterworth bandpass filter on 1D signal.
    """
    @classmethod
    def apply(cls, x, low_fs, high_fs, order=4, fs=512):
        """
        @param x: 1D signal
        @param low_fs: low cutoff frequency (Hz).
        @param high_fs: high cutoff frequency (Hz).
        @param order: order of bandpass filter.
        @param fs: sample rate of input signal (Hz). Default: 512
        """
        # get cutoff pass-band
        nyq = 0.5 * fs
        lowcut = low_fs / nyq
        highcut = high_fs / nyq
        # design a bandpass filter
        [b, a] = signal.butter(order, [lowcut, highcut], btype='bandpass')
        # filter signal
        x_filtered = signal.filtfilt(b, a, x)
        return x_filtered


class HighpassFilter1D:
    """
    Butterworth highpass filter on 1D signal.
    """
    @classmethod
    def apply(cls, x, high_fs, order=4, fs=512):
        """
        @param x: 1D signal
        @param high_fs: high cutoff bandpass frequency (Hz).
        @param order: order of bandpass filter.
        @param fs: sample rate of input signal (Hz). Default: 512
        """
        # get cutoff pass-band
        nyq = 0.5 * fs
        highcut = high_fs / nyq
        # design a bandpass filter
        [b, a] = signal.butter(order, highcut, btype='highpass')
        # filter signal
        x_filtered = signal.filtfilt(b, a, x)
        return x_filtered


class LowpassFilter1D:
    """
    Butterworth lowpass filter on 1D signal.
    """
    @classmethod
    def apply(cls, x, low_fs, order=4, fs=512):
        """
        @param x: 1D signal
        @param low_fs: low cutoff bandpass frequency (Hz).
        @param order: order of bandpass filter.
        @param fs: sample rate of input signal (Hz). Default: 512
        """
        # get cutoff pass-band
        nyq = 0.5 * fs
        lowcut = low_fs / nyq
        # design a bandpass filter
        [b, a] = signal.butter(order, lowcut, btype='lowpass')
        # filter signal
        x_filtered = signal.filtfilt(b, a, x)
        return x_filtered
