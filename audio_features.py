import librosa
import csv

# Import
filename = 'RAP/Rap (mp3cut.net) (58).wav'
genre = 'rap'
y, sr = librosa.load(filename)
# Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
audio_file, _ = librosa.effects.trim(y)
# Chromogram
chromagram = librosa.feature.chroma_stft(y=y, sr = sr, hop_length = 5000)
# Root mean square
rms = librosa.feature.rms(y=y)
# Zero_crossings
zero_crossings = librosa.zero_crossings(audio_file, pad=False)
# Spectral centroids
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
# Spectral bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
# Spectral rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
#Harmonic and Perceptrual
y_harm, y_perc = librosa.effects.hpss(audio_file)
# Tempo
tempo, _ = librosa.beat.beat_track(y=y, sr = sr)
#MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 20)
#mfccs = sklearn.preprocessing.scale(mfccs, axis=1)


header = ['filename','chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','rolloff_mean','rolloff_var','zero_crossing_rate_mean','zero_crossing_rate_var','harmony_mean','harmony_var','perceptr_mean','perceptr_var','tempo','mfcc1_mean','mfcc1_var','mfcc2_mean','mfcc2_var','mfcc3_mean','mfcc3_var','mfcc4_mean','mfcc4_var','mfcc5_mean','mfcc5_var','mfcc6_mean','mfcc6_var','mfcc7_mean','mfcc7_var','mfcc8_mean','mfcc8_var','mfcc9_mean','mfcc9_var','mfcc10_mean','mfcc10_var','mfcc11_mean','mfcc11_var','mfcc12_mean','mfcc12_var','mfcc13_mean','mfcc13_var','mfcc14_mean','mfcc14_var','mfcc15_mean','mfcc15_var','mfcc16_mean','mfcc16_var','mfcc17_mean','mfcc17_var','mfcc18_mean','mfcc18_var','mfcc19_mean','mfcc19_var','mfcc20_mean','mfcc20_var','label']
data = [filename,chromagram.mean(),chromagram.var(),rms.mean(),rms.var(),spectral_centroids.mean(),spectral_centroids.var(),spectral_bandwidth.mean(),spectral_bandwidth.var(),spectral_rolloff.mean(),spectral_rolloff.var(),zero_crossings.mean(),zero_crossings.var(),y_harm.mean(),y_harm.var(),y_perc.mean(),y_perc.var(),tempo,mfccs[0].mean(),mfccs[0].var(),mfccs[1].mean(),mfccs[1].var(),mfccs[2].mean(),mfccs[2].var(),mfccs[3].mean(),mfccs[3].var(),mfccs[4].mean(),mfccs[4].var(),mfccs[5].mean(),mfccs[5].var(),mfccs[6].mean(),mfccs[6].var(),mfccs[7].mean(),mfccs[7].var(),mfccs[8].mean(),mfccs[8].var(),mfccs[9].mean(),mfccs[9].var(),mfccs[10].mean(),mfccs[10].var(),mfccs[11].mean(),mfccs[11].var(),mfccs[12].mean(),mfccs[12].var(),mfccs[13].mean(),mfccs[13].var(),mfccs[14].mean(),mfccs[14].var(),mfccs[15].mean(),mfccs[15].var(),mfccs[16].mean(),mfccs[16].var(),mfccs[17].mean(),mfccs[17].var(),mfccs[18].mean(),mfccs[18].var(),mfccs[19].mean(),mfccs[19].var(),genre]

with open('features_rap.csv','a',encoding='UTF8',newline='') as f:
     writer = csv.writer(f)
     #writer.writerow(header)
     writer.writerow(data)
     

