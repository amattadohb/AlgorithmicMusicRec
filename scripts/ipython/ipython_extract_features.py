# Paul
# this code is adapted from http://www.ifs.tuwien.ac.at/~schindler/lectures/MIR_Feature_Extraction.html

from helpers.rp_extract_batch import *
from helpers.utilities import *
from scikits.talkbox import segment_axis
from numpy.lib import stride_tricks
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import lfilter, hamming
from scipy.io import wavfile
from subprocess import call
from song import *
import python_speech_features
import eyed3
import numpy as np
import os
import pickle

def extract_feat(filepath, output_filepath = 'pickle'):

    # Convert m4a to mp3
    m4a_to_mp3_batch(os.path.join(filepath, 'm4a'), outdir=os.path.join(filepath, 'mp3'))

    # Convert mp3 to wav
    mp3_to_wav_batch(os.path.join(filepath, 'mp3'), outdir=os.path.join(filepath, 'wav'))

    i = 0
    for subpath, dirs, filelist in os.walk(os.path.join(filepath, 'wav')):
        for filename in filelist:
            if filename[-4:] == '.wav':

                print 'Extracting features:', i, '/', len(filelist)
                i += 1

                # Get track name and artist
                name, artist = getTrackDetails(os.path.join(filepath, 'mp3', filename[:-4] + '.mp3'))

                #if os.path.isfile( os.path.join('music', 'pickle', artist + '_' + name + '.p') ):
                #   continue

                samplerate, wavedata = wavfile.read(os.path.join(subpath, filename))

                features = {}

                if wavedata.shape[1] > 1: #Stereo

                    # use combine the channels by calculating their geometric mean
                    wavedata = np.mean(wavedata , axis=1)

                # Calculate zero crossing rate
                block_length = 2048
                zcr_o, ts_zcr = zero_crossing_rate(wavedata, block_length, samplerate)
                zcr = median_pool(zcr_o)
                # Normalize
                zcr = zcr - np.mean(zcr)
                if np.std(zcr) != 0:
                    zcr = zcr / np.std(zcr)
                #print 'zcr'
                #print zcr
                #print len(zcr)
                features['zcr'] = zcr

                # Calculate spectral centroid
                window_size = 1024
                sc_o, ts_sc = spectral_centroid(wavedata, window_size, samplerate)
                sc = median_pool(sc_o)
                # Normalize
                sc = sc - np.mean(sc)
                if np.std(sc) != 0:
                    sc = sc / np.std(sc)
                #print 'sc'
                #print sc
                #print len(sc)
                features['sc'] = sc

                # Calculate spectral rolloff
                sr_o, ts_sr = spectral_rolloff(wavedata, window_size, samplerate, k=0.85)
                sr = median_pool(sr_o)
                # Normalize
                sr = sr - np.mean(sr)
                if np.std(sr) != 0:
                    sr = sr / np.std(sr)
                #print 'sr'
                #print sr
                #print len(sr)
                features['sr'] = sr

                '''
                # Calculate MFCC
                # MFCCs = mfcc(wavedata)
                MFCCs_o = python_speech_features.mfcc(wavedata, samplerate=44100)
                MFCCs = median_pool_2d(MFCCs_o)
                # Normalize
                MFCCs = (MFCCs - np.mean(MFCCs)) / np.std(MFCCs)
                print 'MFCCs'
                print MFCCs
                print len(MFCCs)
                features['mfcc'] = MFCCs
                '''
                track = Song(name, artist, features)

                pickle.dump(track, open( os.path.join('music', 'pickle', artist + '_' + name + '.p') , "wb" ))


def getTrackDetails(filepath):
    audiofile = eyed3.load(filepath)
    return audiofile.tag.title, audiofile.tag.artist


def median_pool(vector, num_features = 200):
    output = np.zeros(num_features)
    window_size = len(vector) % num_features
    for i in xrange(window_size - 1):
        window = vector[i:i + window_size]
        val = np.median(window)
        output.itemset(i, val)
    return output

def median_pool_2d(vector, num_features = 200):
    output = np.zeros(num_features)
    v = vector.flatten()
    window_size = len(v) % num_features
    for i in xrange(window_size - 1):
        window = v[i:i + window_size]
        val = np.median(window)
        output.itemset(i, val)
    return output

def m4a_to_mp3_batch(path,outdir=None,audiofile_types=('.m4a')):

    get_relative_path = (outdir!=None) # if outdir is specified we need relative path otherwise absolute

    filenames = find_files(path,audiofile_types,get_relative_path)

    n_files = len(filenames)
    n = 0

    for file in filenames:

        n += 1
        basename, ext = os.path.splitext(file)
        mp3_file = basename + '.mp3'

        if outdir: # if outdir is specified we add it in front of the relative file path
            file = path + os.sep + file
            mp3_file = outdir + os.sep + mp3_file

            # recreate same subdir path structure as in input path
            out_subpath = os.path.split(mp3_file)[0]

            if not os.path.exists(out_subpath):
                os.makedirs(out_subpath)

        # future option: (to avoid recreating the input path subdir structure in outdir)
        #filename_only = os.path.split(mp3_file)[1]

        try:
            if not os.path.exists(mp3_file):
                print "Decoding:", n, "/", n_files, ":"
                if ext.lower() == '.m4a':
                    #mp3_decode(file,mp3_file)
                    #call(['ffmpeg', '-i', file, '-r', '24', '-c:a', 'libmp3lame', '-ac', '2', '-b:a', '190k', mp3_file])
                    call(['ffmpeg', '-v', '5', '-y', '-i', file, '-acodec', 'libmp3lame', '-ac', '2', '-ab', '192k', mp3_file])
            else:
                print "Already existing: " + mp3_file
        except:
            print "Not decoded " + file


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

