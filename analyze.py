import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib import mlab

from scipy import signal
from scipy.signal import butter, lfilter

config = {
    "channel": 0,
    "hackrf_gain": 0,
    "hackrf_gain_if": 35,
    "hackrf_gain_bb": 39,
    "usrp_gain": 40,
    "target_freq": 2.528e9,
    "sampling_rate": 5e6,
    "num_points": 1,
    "num_traces_per_point": 100,
    "bandpass_lower": 1.88e6,
    "bandpass_upper": 1.92e6,
    "lowpass_freq": 5e3,
    "drop_start": 50e-3+0.35,
    "trigger_rising": True,
    "trigger_offset": 100e-6,
    "signal_length": 4*1000e-6,
    "template_name": "templates/tiny_anechoic_5m.npy",
    "min_correlation": 0.00
}


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def find_starts(config, data):
    """
    Find the starts of interesting activity in the signal.

    The result is a list of indices where interesting activity begins, as well
    as the trigger signal and its average.
    """

    trigger = butter_bandpass_filter(
        data, config.bandpass_lower, config.bandpass_upper,
        config.sampling_rate, 6)
    trigger = np.absolute(trigger)
    trigger = butter_lowpass_filter(
        trigger, config.lowpass_freq, config.sampling_rate, 6)

    # transient = 0.0005
    # start_idx = int(transient * config.sampling_rate)
    start_idx = 0
    average = np.average(trigger[start_idx:])
    maximum = np.max(trigger[start_idx:])
    minimum = np.min(trigger[start_idx:])
    middle = (np.max(trigger[start_idx:]) - min(trigger[start_idx:])) / 2
    if average < 1.1 * middle:
        print()
        print("Adjusting average to avg + (max - avg) / 2")
        average = average + (maximum - average) / 2
    offset = -int(config.trigger_offset * config.sampling_rate)

    if config.trigger_rising:
        trigger_fn = lambda x, y: x > y
    else:
        trigger_fn = lambda x, y: x < y

    # The cryptic numpy code below is equivalent to looping over the signal and
    # recording the indices where the trigger crosses the average value in the
    # direction specified by config.trigger_rising. It is faster than a Python
    # loop by a factor of ~1000, so we trade readability for speed.
    trigger_signal = trigger_fn(trigger, average)[start_idx:]
    starts = np.where((trigger_signal[1:] != trigger_signal[:-1])
                      * trigger_signal[1:])[0] + start_idx + offset + 1
    if trigger_signal[0]:
        starts = np.insert(starts, 0, start_idx + offset)

    # plt.plot(data)
    # plt.plot(trigger*100)
    # plt.axhline(y=average*100)
    # plt.show()

    return starts, trigger, average


def plot_results(config, data, trigger, trigger_average, starts, traces):
    plt.subplots_adjust(hspace=0.6)
    plt.subplot(4, 1, 1)

    t = np.linspace(0, len(data) / config.sampling_rate, len(data))
    plt.plot(t, data)
    plt.title("Time domain capture")
    plt.xlabel("time [s]")
    plt.ylabel("normalized amplitude")

    plt.plot(t, trigger * 10)
    plt.axhline(y=trigger_average * 10, color='y')
    trace_length = int(config.signal_length * config.sampling_rate)
    for start in starts:
        stop = start + trace_length
        plt.axvline(x=start / config.sampling_rate, color='r', linestyle='--')
        plt.axvline(x=stop / config.sampling_rate, color='g', linestyle='--')

    plt.subplot(4, 1, 2)
    plt.specgram(
        data, NFFT=128, Fs=config.sampling_rate, Fc=0, detrend=mlab.detrend_none,
        window=mlab.window_hanning, noverlap=127, cmap=None, xextent=None,
        pad_to=None, sides='default', scale_by_freq=None, mode='default',
        scale='default')
    plt.title("Spectrogram")
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

    # plt.subplot(4, 1, 3)
    # plt.psd(
    # data, NFFT=1024, Fs=config.sampling_rate, Fc=0, detrend=mlab.detrend_none,
    # window=mlab.window_hanning, noverlap=0, pad_to=None,
    # sides='default', scale_by_freq=None, return_line=None)

    if (len(traces) == 0):
        print("WARNING: no encryption was extracted")
    else:
        t = np.linspace(0, len(traces[0]) / config.sampling_rate, len(traces[0]))
        plt.subplot(4, 1, 3)
        for trace in traces:
            plt.plot(t, trace / max(trace))
        plt.title("%d aligned traces" % config.num_traces_per_point)
        plt.xlabel("time [s]")
        plt.ylabel("normalized amplitude")

        plt.subplot(4, 1, 4)
        avg = np.average(traces, axis=0)
        plt.plot(t, avg / max(avg))
        plt.title("Average of %d traces" % config.num_traces_per_point)
        plt.xlabel("time [s]")
        plt.ylabel("normalized amplitude")

    plt.show()


def main():
    capture_file = 'data/raw__1.npy'
    global config
    config = obj(config)
    with open(capture_file) as f:
        data = np.fromfile(f, dtype=np.complex64)

    data = np.absolute(data)
    # cut usless transient
    data = data[int(config.drop_start * config.sampling_rate):]

    trace_starts, trigger, trigger_avg = find_starts(config, data)
    # extract at trigger + autocorrelate with the first to align
    traces = []
    trace_length = int(config.signal_length * config.sampling_rate)
    for start in trace_starts:
        if len(traces) >= config.num_traces_per_point:
            break

        stop = start + trace_length

        if stop > len(data):
            break

        trace = data[start:stop]
        template = trace

        trace_lpf = butter_lowpass_filter(trace, config.sampling_rate / 4,
                                          config.sampling_rate)
        template_lpf = butter_lowpass_filter(template, config.sampling_rate / 4,
                                             config.sampling_rate)

        correlation = signal.correlate(trace_lpf ** 2, template_lpf ** 2)

        # print max(correlation)
        if max(correlation) <= config.min_correlation:
            continue

        shift = np.argmax(correlation) - (len(template) - 1)
        traces.append(data[start + shift:stop + shift])

    plot_results(config, data, trigger, trigger_avg, trace_starts, traces)


if __name__ == "__main__":
    main()
