import librosa
import csv

# Import
filename = 'KUNDIMAN/kundiman (last) (mp3cut.net) (11).wav'
genre = 'kundiman'
y, sr = librosa.load(filename)
# Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
audio_file, _ = librosa.effects.trim(y)

# Chromogram
chromagram = librosa.feature.chroma_stft(y=y, sr = sr, hop_length = 5000)
chromagram_mean = chromagram.mean()
chromagram_var = chromagram.var()
# Root mean square
rms = librosa.feature.rms(y=y)
rms_mean = rms.mean()
rms_var = rms.var()
# Zero_crossings
zero_crossings = librosa.zero_crossings(audio_file, pad=False)
zero_crossings_mean = zero_crossings.mean()
zero_crossings_var = zero_crossings.var()
# Spectral centroids
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_centroids_mean = spectral_centroids.mean()
spectral_centroids_var = spectral_centroids.var()
# Spectral bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
spectral_bandwidth_mean = spectral_bandwidth.mean()
spectral_bandwidth_var = spectral_bandwidth.var()
# Spectral rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
spectral_rolloff_mean = spectral_rolloff.mean()
spectral_rolloff_var = spectral_rolloff.var()
#Harmonic and Perceptrual
y_harm, y_perc = librosa.effects.hpss(audio_file)
y_harm_mean = y_harm.mean()
y_harm_var = y_harm.var()
y_perc_mean = y_perc.mean()
y_perc_var = y_perc.var()
# Tempo
tempo, _ = librosa.beat.beat_track(y=y, sr = sr)
#MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 20)
#mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
mfccs_0_mean = mfccs[0].mean()
mfccs_0_var = mfccs[0].var()
mfccs_1_mean = mfccs[1].mean()
mfccs_1_var = mfccs[1].var()
mfccs_2_mean = mfccs[2].mean()
mfccs_2_var = mfccs[2].var()
mfccs_3_mean = mfccs[3].mean()
mfccs_3_var = mfccs[3].var()
mfccs_4_mean = mfccs[4].mean()
mfccs_4_var = mfccs[4].var()
mfccs_5_mean = mfccs[5].mean()
mfccs_5_var = mfccs[5].var()
mfccs_6_mean = mfccs[6].mean()
mfccs_6_var = mfccs[6].var()
mfccs_7_mean = mfccs[7].mean()
mfccs_7_var = mfccs[7].var()
mfccs_8_mean = mfccs[8].mean()
mfccs_8_var = mfccs[8].var()
mfccs_9_mean = mfccs[9].mean()
mfccs_9_var = mfccs[9].var()
mfccs_10_mean = mfccs[10].mean()
mfccs_10_var = mfccs[10].var()
mfccs_11_mean = mfccs[11].mean()
mfccs_11_var = mfccs[11].var()
mfccs_12_mean = mfccs[12].mean()
mfccs_12_var = mfccs[12].var()
mfccs_13_mean = mfccs[13].mean()
mfccs_13_var = mfccs[13].var()
mfccs_14_mean = mfccs[14].mean()
mfccs_14_var = mfccs[14].var()
mfccs_15_mean = mfccs[15].mean()
mfccs_15_var = mfccs[15].var()
mfccs_16_mean = mfccs[16].mean()
mfccs_16_var = mfccs[16].var()
mfccs_17_mean = mfccs[17].mean()
mfccs_17_var = mfccs[17].var()
mfccs_18_mean = mfccs[18].mean()
mfccs_18_var = mfccs[18].var()
mfccs_19_mean = mfccs[19].mean()
mfccs_19_var = mfccs[19].var()

header = ['filename','chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','rolloff_mean','rolloff_var','zero_crossing_rate_mean','zero_crossing_rate_var','harmony_mean','harmony_var','perceptr_mean','perceptr_var','tempo','mfcc1_mean','mfcc1_var','mfcc2_mean','mfcc2_var','mfcc3_mean','mfcc3_var','mfcc4_mean','mfcc4_var','mfcc5_mean','mfcc5_var','mfcc6_mean','mfcc6_var','mfcc7_mean','mfcc7_var','mfcc8_mean','mfcc8_var','mfcc9_mean','mfcc9_var','mfcc10_mean','mfcc10_var','mfcc11_mean','mfcc11_var','mfcc12_mean','mfcc12_var','mfcc13_mean','mfcc13_var','mfcc14_mean','mfcc14_var','mfcc15_mean','mfcc15_var','mfcc16_mean','mfcc16_var','mfcc17_mean','mfcc17_var','mfcc18_mean','mfcc18_var','mfcc19_mean','mfcc19_var','mfcc20_mean','mfcc20_var','label']
data = [filename,chromagram_mean,chromagram_var,rms_mean,rms_var,spectral_centroids_mean,spectral_centroids_var,spectral_bandwidth_mean,
        spectral_bandwidth_var,spectral_rolloff_mean,spectral_rolloff_var,zero_crossings_mean,zero_crossings_var,y_harm_mean,y_harm_var,
        y_perc_mean,y_perc_var,tempo,mfccs_0_mean,mfccs_0_var,mfccs_1_mean,mfccs_1_var,mfccs_2_mean,mfccs_2_var,mfccs_3_mean,mfccs_3_var,
        mfccs_4_mean,mfccs_4_var,mfccs_5_mean,mfccs_5_var,mfccs_6_mean,mfccs_6_var,mfccs_7_mean,mfccs_7_var,mfccs_8_mean,mfccs_8_var,
        mfccs_9_mean,mfccs_9_var,mfccs_10_mean,mfccs_10_var,mfccs_11_mean,mfccs_11_var,mfccs_12_mean,mfccs_12_var,mfccs_13_mean,mfccs_13_var,
        mfccs_14_mean,mfccs_14_var,mfccs_15_mean,mfccs_15_var,mfccs_16_mean,mfccs_16_var,mfccs_17_mean,mfccs_17_var,mfccs_18_mean,mfccs_18_var,
        mfccs_19_mean,mfccs_19_var,genre]

with open('features_kundiman.csv','a',encoding='UTF8',newline='') as f:
     writer = csv.writer(f)
     #writer.writerow(header)
     writer.writerow(data)
     

