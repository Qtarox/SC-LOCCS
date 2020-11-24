import codecs
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.stats.stats import pearsonr
from scipy import signal
from utils import sBox
from load import generic_load
from signal_alignment import phase_align, chisqr_align
from dtwalign import dtw
from scipy.ndimage.interpolation import shift

LOG_FORMAT = LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class CPA:
    traces = ''
    traceNum = ''
    signalLength = 0
    real_keys = []
    plainTexts = []
    hammingWeight = None

    def __init__(self, traces, real_keys, plain_texts, hamming_weight_path='data/hwMat.npy'):
        self.traces = traces
        self.traceNum = np.shape(traces)[0]
        self.signalLength = np.shape(traces)[1]
        self.real_keys = real_keys
        self.plainTexts = plain_texts
        self.hammingWeight = np.load(hamming_weight_path)
        self.hammingWeight = self.hammingWeight.reshape(-1)

    def getKey(self, numBytes=1):
        keys = np.zeros(numBytes)
        hws = np.zeros(self.traceNum)
        pccs = np.zeros(self.signalLength)
        cpa = np.zeros(256)
        transTraces = np.transpose(self.traces)
        pge = np.zeros(numBytes)
        for i in range(numBytes):

            for guess in range(256):
                for j in range(self.traceNum):
                    input = self.plainTexts[j][i] ^ guess
                    input = sBox[input]
                    hws[j] = self.hammingWeight[input]
                for j in range(self.signalLength):
                    pccs[j] = self.getPCC(transTraces[j], hws)
                cpa[guess] = np.max(abs(pccs))
                # logging.info("Guess %d" % guess)
            keys[i] = np.argmax(cpa)
            cparefs = np.argsort(cpa)[::-1]
            pge[i] = list(cparefs).index(self.real_keys[i]) + 1
            logging.info("Done %d byte" % i)
        return keys, pge

    def getPCC(self, X, Y):
        # return np.corrcoef(X,Y)[0,1]
        return pearsonr(X, Y)[0]


def plot_sample(data, title=""):
    idx = [0, 1, 2, 3, 4]  # np.random.choice(len(data),5,replace=False)
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(data[idx[i]])
    # plt.title(title)
    plt.show()


def align(trace0, tracei):
    correlation = signal.correlate(tracei ** 2, trace0 ** 2)
    shift = np.argmax(correlation) - (len(trace0) - 1)
    return shift


if __name__ == '__main__':
    DATA_PATH = 'E:\\data\\'
    # DATA_PATH = 'E:\\data\\sample_traces\\hackrf_20cm\\attack_tx_500'
    # DATA_PATH = 'E:\\data\\traces\\mbedtls_1m_home'
    fixed_key, raw_plaintexts, real_keys, raw_traces = generic_load(
        data_path=DATA_PATH, name='', number=15000, wstart=0, wend=1500, norm=True)

    traces = []
    plaintexts = []

    dbg = 1  # eval(input("Input debug value"))
    if dbg == 1:
        s = 0
        for i in range(len(raw_traces)):
            # s = chisqr_align(raw_traces[0], raw_traces[i], (100, 400), bound=100)
            s = align(trace0=raw_traces[0], tracei=raw_traces[i])
            # logging.info("s: %d"%(s))
            # s = phase_align(raw_traces[0], raw_traces[i], (500, 1200))
            # cor = pearsonr(raw_traces[0], shift(raw_traces[i], s, mode='nearest'))
            # if cor[0] > 0.8:
            traces.append(raw_traces[i][500 + s:1300 + s])
            # traces.append(shift(raw_traces[i], s, mode='nearest')[500:1000])
            plaintexts.append(raw_plaintexts[i])  # cut
    else:
        traces = raw_traces
        plaintexts = raw_plaintexts

    plot_sample(raw_traces, 'raw')
    plot_sample(traces, 'aligned')
    # exit()

    logging.info("Toal traces: " + str(len(traces)))

    attacker = CPA(traces=traces, real_keys=real_keys[0], plain_texts=plaintexts)
    keys, pge = attacker.getKey(numBytes=16)

    # plt.plot(cpa)
    # plt.axvline(real_keys[0][0],color='r', linestyle='--')
    # plt.title('Correlation distribution of 1st Byte. dbg:%d'%dbg)
    # plt.show()

    print("Recovered key: " + str(keys))
    print("Real key: " + str(real_keys[0]))
    print(pge)
