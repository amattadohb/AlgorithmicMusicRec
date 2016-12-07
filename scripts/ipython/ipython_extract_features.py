# Paul
# this code is adapted from http://www.ifs.tuwien.ac.at/~schindler/lectures/MIR_Feature_Extraction.html

from helpers.rp_extract_batch import mp3_to_wav_batch
from helpers.utilities import *
from scikits.talkbox import segment_axis
from numpy.lib import stride_tricks
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import lfilter, hamming
from scipy.io import wavfile
import numpy as np
import os

def extract_feat(filepath, output_filepath = '/output/'):

	mp3_to_wav_batch(filepath)

	for subpath, dirs, filelist in os.walk(filepath):
		for filename in filelist:
			if filename[-4:] == '.wav':

				samplerate, wavedata = wavfile.read(os.path.join(subpath, filename))

				if wavedata.shape[1] > 1: #Stereo

					# use combine the channels by calculating their geometric mean
					wavedata = np.mean(wavedata , axis=1)

				# Calculate zero crossing rate
				block_length = 2048
				zcr, ts_zcr = zero_crossing_rate(wavedata, block_length, samplerate)
				print 'zcr'
				print zcr
				print len(zcr)

				# Calculate spectral centroid
				window_size = 1024
				sc, ts_sc = spectral_centroid(wavedata, window_size, samplerate)
				print 'sc'
				print sc
				print len(sc)

				# Calculate spectral rolloff
				sr, ts_sr = spectral_rolloff(wavedata, window_size, samplerate, k=0.85)
				print 'sr'
				print sr
				print len(sr)

				# Calculate MFCC
				MFCCs = mfcc(wavedata)
				print 'MFCCs'
				print MFCCs
				print len(MFCCs)


def mfcc(input_data):
	# Pre-emphasis filter.

	# Parameters
	nwin = 256
	nfft = 1024
	fs = 16000
	nceps = 13

	# Pre-emphasis factor (to take into account the -6dB/octave
	# rolloff of the radiation at the lips level)
	prefac = 0.97

	# MFCC parameters: taken from auditory toolbox
	over = nwin - 160

	filtered_data = lfilter([1., -prefac], 1, input_data)

	windows = hamming(256, sym=0)
	framed_data = segment_axis(filtered_data, nwin, over) * windows

	# Compute the spectrum magnitude
	magnitude_spectrum = np.abs(fft(framed_data, nfft, axis=-1))

	# Compute triangular filterbank for MFCC computation.

	lowfreq = 133.33
	linsc = 200/3.
	logsc = 1.0711703
	fs = 44100

	nlinfilt = 13
	nlogfilt = 27

	# Total number of filters
	nfilt = nlinfilt + nlogfilt

	#------------------------
	# Compute the filter bank
	#------------------------
	# Compute start/middle/end points of the triangular filters in spectral
	# domain
	freqs = np.zeros(nfilt+2)
	freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
	freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
	heights = 2./(freqs[2:] - freqs[0:-2])

	# Compute filterbank coeff (in fft domain, in bins)
	filterbank = np.zeros((nfilt, nfft))

	# FFT bins (in Hz)
	nfreqs = np.arange(nfft) / (1. * nfft) * fs

	for i in range(nfilt):

		low = freqs[i]
		cen = freqs[i+1]
		hi  = freqs[i+2]

		lid = np.arange(np.floor(low * nfft / fs) + 1, np.floor(cen * nfft / fs) + 1, dtype=np.int)

		rid = np.arange(np.floor(cen * nfft / fs) + 1, np.floor(hi * nfft / fs)  + 1, dtype=np.int)

		lslope = heights[i] / (cen - low)
		rslope = heights[i] / (hi - cen)

		filterbank[i][lid] = lslope * (nfreqs[lid] - low)
		filterbank[i][rid] = rslope * (hi - nfreqs[rid])


	# apply filter
	mspec = np.log10(np.dot(magnitude_spectrum, filterbank.T))

	# Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
	return dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]


def spectral_rolloff(wavedata, window_size, samplerate, k=0.85):
    
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    power_spectrum     = np.abs(magnitude_spectrum)**2
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(samplerate)))
    
    sr = []

    spectralSum    = np.sum(power_spectrum, axis=1)
    
    for t in range(timebins-1):
        
        # find frequency-bin indeces where the cummulative sum of all bins is higher
        # than k-percent of the sum of all bins. Lowest index = Rolloff
        sr_t = np.where(np.cumsum(power_spectrum[t,:]) >= k * spectralSum[t])[0][0]
        
        sr.append(sr_t)
        
    sr = np.asarray(sr).astype(float)
    
    # convert frequency-bin index to frequency in Hz
    sr = (sr / freqbins) * (samplerate / 2.0)
    
    return sr, np.asarray(timestamps)



def spectral_centroid(wavedata, window_size, samplerate):
    
    magnitude_spectrum = stft(wavedata, window_size)
    
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(samplerate)))
    
    sc = []

    for t in range(timebins-1):
        
        power_spectrum = np.abs(magnitude_spectrum[t])**2
        
        sc_t = np.sum(power_spectrum * np.arange(1,freqbins+1)) / np.sum(power_spectrum)
        
        sc.append(sc_t)
        
    
    sc = np.asarray(sc)
    sc = np.nan_to_num(sc)
    
    return sc, np.asarray(timestamps)



def zero_crossing_rate(wavedata, block_length, samplerate):
    
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(samplerate)))
    
    zcr = []
    
    for i in range(0,num_blocks-1):
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        
        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))
        zcr.append(zc)
    
    return np.asarray(zcr), np.asarray(timestamps)

