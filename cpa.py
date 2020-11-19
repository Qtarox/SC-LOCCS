import codecs
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.stats.stats import pearsonr
from utils import sBox
from load import generic_load
from signal_alignment import phase_align, chisqr_align
from dtwalign import dtw
from scipy.ndimage.interpolation import shift

LOG_FORMAT = LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO,format=LOG_FORMAT)


class CPA:
    traces = ''
    traceNum = ''
    signalLength = 0
    plainTexts = []
    hammingWeight = None

    def __init__(self, traces, plain_texts, hamming_weight_path='data/hwMat.npy'):
        self.traces = traces
        self.traceNum = np.shape(traces)[0]
        self.signalLength = np.shape(traces)[1]
        self.plainTexts = plain_texts
        self.hammingWeight = np.load(hamming_weight_path)
        self.hammingWeight = self.hammingWeight.reshape(-1)

    def getKey(self, numBytes=1):
        keys = np.zeros(numBytes)
        hws = np.zeros(self.traceNum)
        pccs = np.zeros(self.signalLength)
        cpa = np.zeros(256)
        transTraces=np.transpose(self.traces)
        for i in range(numBytes):
            for guess in range(256):
                for j in range(self.traceNum):
                    input = self.plainTexts[j][i] ^ guess
                    input = sBox[input]
                    hws[j] = self.hammingWeight[input]
                for j in range(self.signalLength):
                    pccs[j] = self.getPCC(transTraces[j],hws)
                cpa[guess] = np.max(abs(pccs))
                logging.info("Guess %d" % guess)
            keys[i] = np.argmax(cpa)
            logging.info("Done %d byte"%i)
        return keys, cpa

    def getPCC(self,X,Y):
        # return np.corrcoef(X,Y)[0,1]
        return pearsonr(X,Y)[0]

def plot_sample(data):
    idx = np.random.choice(len(data),5,replace=False)
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(data[idx[i]])
    plt.show()

if __name__ == '__main__':
    # DATA_PATH = 'E:\\data\\'
    # DATA_PATH = 'E:\\data\\sample_traces\\hackrf_20cm\\attack_tx_500'
    DATA_PATH = 'E:\\data\\traces\\mbedtls_1m_home'
    fixed_key, raw_plaintexts, real_keys, raw_traces = generic_load(
        data_path=DATA_PATH, name='mbed_villa_500', number=5000,wstart=0,wend=0)

    traces=[]
    plaintexts=[]

    # s=0
    # for i in range(len(raw_traces)):
    #     # s = chisqr_align(raw_traces[0], raw_traces[i], (100, 400), bound=100)
    #     s = phase_align(raw_traces[0], raw_traces[i], (200, 400))
    #     cor = pearsonr(raw_traces[0], shift(raw_traces[i], s, mode='nearest'))
    #     if cor[0] > 0.8:
    #         traces.append(shift(raw_traces[i], s, mode='nearest'))
    #         plaintexts.append(raw_plaintexts[i]) # cut

    traces=raw_traces
    plaintexts=raw_plaintexts

    # plot_sample(raw_traces)
    plot_sample(traces)
    # exit()

    logging.info("Toal traces: "+str(len(traces)))

    attacker = CPA(traces=traces, plain_texts=plaintexts)
    keys,cpa=attacker.getKey()

    plt.plot(cpa)
    plt.title('Correlation distribution of 1st Byte using Hamming Weight, 5000 traces')
    plt.show()

    print(keys)
    print(real_keys[0])