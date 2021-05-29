import matplotlib.pyplot as plt
import numpy as np
from pylab import plot
import essentia.standard as es

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Load audio file
audio = es.MonoLoader(filename='./Data/example.wav')()

# compute beat positions and BPM
rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

# Show extracted beat positions
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default
plot(audio)
for beat in beats:
    plt.axvline(x=beat*44100, color='red')
plt.title("Audio waveform and the estimated beat positions")

# Beat histograms
peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, histogram = es.BpmHistogramDescriptors()(beats_intervals)

print("Overall BPM (estimated before): %0.1f" % bpm)

fig, ax = plt.subplots()
ax.bar(range(len(histogram)), histogram, width=1)
ax.set_xlabel('BPM')
ax.set_ylabel('Frequency')
plt.title("BPM histogram")
ax.set_xticks([20 * x + 0.5 for x in range(int(len(histogram) / 20))])
ax.set_xticklabels([str(20 * x) for x in range(int(len(histogram) / 20))])


# Extract the pitch curve
# PitchMelodia takes the entire audio signal as input (no frame-wise processing is required)
pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
pitch_values, pitch_confidence = pitch_extractor(audio)

# Pitch is estimated on frames. Compute frame time positions
pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values) )

# Plot the estimated pitch contour and confidence over time
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title('estimated pitch [Hz]')
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title('pitch confidence')

plt.figure()
plt.subplot(211)
plt.plot(beats[1:],beats_intervals)
plt.xlabel('Time (sec)')
plt.ylabel('Beat Interval (sec)')
plt.subplot(212)
plt.plot(pitch_times,pitch_values)
plt.xlabel('Time (sec)')
plt.ylabel('Pitch (Hz)')

# Plot beat vs pitch
beat_pitch_arr = np.zeros([len(pitch_values), 2])
i = 0
for pitch_time,ind_pitch in zip(pitch_times,pitch_values):
    # find the nearest beat time from pitch_time
    beat_time, index = find_nearest(beats, pitch_time)
    ind_beatint = beats_intervals[index-1]

    if ind_pitch>0:
        beat_pitch_arr[i, 0] = ind_beatint
        beat_pitch_arr[i, 1] = ind_pitch
        i = i+1

win = 200
smooth_beat_time = moving_average(beat_pitch_arr[:,0], win)
smooth_pitch = moving_average(beat_pitch_arr[:,1], win)

plt.figure()
plt.plot(beat_pitch_arr[:,0],beat_pitch_arr[:,1],'.')
plt.plot(smooth_beat_time,smooth_pitch,'r-')
plt.xlim([0.4,0.7])
plt.xlabel('Beat Interval (sec)')
plt.ylabel('Pitch (Hz)')
plt.show()



