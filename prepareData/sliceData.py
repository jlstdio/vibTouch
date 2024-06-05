from dataAnalyzer.showPlot_Acc import showRawData
from util.util import *
from scipy.io import wavfile
from scipy.signal import resample
import os
from pydub import AudioSegment
import soundfile as sf


def downsample_audio(audio_path, target_sample_rate=16000):
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    sample_rate = audio.frame_rate

    # Check if downsampling is needed
    if sample_rate > target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)
        audio.export(audio_path, format="wav")
        print(f'Audio downsampled to {target_sample_rate} Hz')


def slicer(type='', location=0):
    if not (type.__contains__('tap') or type.__contains__('slide')):
        print('key wrong')
        return False

    folder_path = '../data/raw'

    if type == 'tap':
        dataAccDir = folder_path + '/' + type + '/' + str(location) + '/Accelerometer.csv'
        dataAudioDir = folder_path + '/' + type + '/' + str(location) + '/Microphone.mp4'
    else:
        dataAccDir = folder_path + '/' + type + '_' + str(location) + '/Accelerometer.csv'
        dataAudioDir = folder_path + '/' + type + '_' + str(location) + '/Microphone.mp4'

    showRawData(dataAccDir, True)

    shift = float(input('shift : '))
    interval = int(input('interval : '))

    # slice acc data
    shift *= 10 ** 9
    originalSet = 5.7 * 10 ** 9 + shift

    if type == 'tap':
        destDir = '../data/sliced/' + type + '/' + str(location) + '/'
    else:
        destDir = '../data/sliced/' + type + '_' + str(location) + '/'

    # Count the number of existing files in the destination directory
    acc_dir = os.path.join(destDir, 'acc')

    count = 0
    if os.path.exists(acc_dir):
        count += len([name for name in os.listdir(acc_dir) if os.path.isfile(os.path.join(acc_dir, name))])

    count -= 1
    print(f'starting count at {count}')

    # Downsample audio if needed
    downsample_audio(dataAudioDir)

    count = sliceData(dataAccDir, dataAudioDir, originalSet, interval, destDir, count)

    print(f'{count} data created')
    # ./data/sliced/tap/0/audio/audio_0.wav
    # ./data/sliced/tap/0/acc/acc_0.csv

    editedDir = destDir + '/acc/acc_0.csv'
    showRawData(editedDir, False)


print(f'starting')

type = 'tap'
# type = 'slide/3'
for location in range(0, 4):
    print(f'working on {type} location : {location}')
    slicer(type, location)

'''
# count = 0
for i in range(10, 14, 1):
    print(f'working on label : {i}')
    key = str(i)
    writeCount = extractFeaturesToCSV('keyboard_tap_full.csv', key, label=i)
    print(f'{writeCount} rows inserted')
'''
