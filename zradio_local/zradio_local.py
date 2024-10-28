#!/usr/bin/env python3

import argparse
import os
import sys
import threading
import queue
import time

from pydub import AudioSegment
from pydub.playback import play


import sounddevice as sd
import soundfile as sf

"""
Installation of audio libraries we are trying:

# Installing simpleaudio:  https://simpleaudio.readthedocs.io/en/latest/
sudo apt-get install -y python3-dev libasound2-dev
pip install simpleaudio

# Installing pydub:  https://github.com/jiaaro/pydub#installation
pip install pydub


# Installing pyaudio:  https://people.csail.mit.edu/hubert/pyaudio/docs/#
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio


# Install python-sounddevice:  https://python-sounddevice.readthedocs.io/en/0.5.1/installation.html
sudo apt install portaudio19-dev python3-pyaudio
pip install numpy
pip install soundfile
pip install sounddevice

# sounddevice is interesting because it includes a bunch of examples: 
https://github.com/spatialaudio/python-sounddevice/blob/master/examples/rec_unlimited.py




Interesting links:

A general intro to python audio libraries with examples (including how to record audio): 
- https://realpython.com/playing-and-recording-sound-python/




"""


def list_all_audio_devices() -> sd.DeviceList:
    """
    Returns an object of class sd.DeviceList representing a list with information about all available audio devices.
    It contains a dictionary for each available device, holding the keys described in `query_devices()`.
    If used as string it produces a user-friendly string representation of the list of devices.
    """
    return sd.query_devices()


# write a function that takes the path to an audio files (wav, mp3)
# and plays that 


def play_audio_pydub(file_path, format='wav'):
    
    sound = AudioSegment.from_file(file_path, format)
    play(sound)

    # play backwards
    backwards = sound.reverse()  # song is not modified
    play(backwards)

    # overlay 2 audios
    overlay = sound.overlay(backwards)
    play(overlay)


def play_audio_sounddevice(file_path, format=None):
    # show list of audio devices
    # print(sd.query_devices())

    # Play with soundfile
    # no sample rate, channels or file format need to be given,
    # information is obtained from file (except for RAW format)
    data, fs = sf.read(file_path, dtype='float32')
    sd.play(data, fs)
    status = sd.wait()  # Wait until file is done playing


def overlay_audios(*audio_files, format='wav', play_sideeffect=True):
    mixed = None
    for i, file in enumerate(audio_files):
        sound = AudioSegment.from_file(file, format)
        if i == 0:
            mixed = sound
        if i > 0:
            mixed = sound.overlay(mixed)
    if play_sideeffect:
        play(mixed)
    return mixed


class RecordAudio:
    """
    Record audio from the default microphone using python-sounddevice and soundfile.
    sources:
    1. https://github.com/spatialaudio/python-sounddevice/blob/master/examples/rec_unlimited.py
    2. https://github.com/spatialaudio/python-sounddevice/blob/master/examples/rec_gui.py
    """

    def __init__(self, device=None, channels=2, samplerate=None, file_path=None):
        # If *device* is None and *kind* is 'input' or 'output', a single dictionary is returned
        # with information about the default input or output device
        self.device = device  # input device (numeric ID or substring). If None
        if self.device is None:
            self.device = sd.default.device
        self.kind = 'input'
        # channels 1 for mono, 2 for stereo
        self.channels = channels
        self.samplerate = samplerate
        self.subtype = None  # sound file subtype (e.g. "PCM_24")

        self.file_path = file_path
        self.file_list = []
        if file_path:
            self.file_list.append(file_path)

        self.recording = False
        self.previously_recording = False
        self.q_audio = queue.Queue()

        self.thread_recording = None  # threading.Thread(target=self.start_recording)

    def get_device_info(self):
        device_info = sd.query_devices(self.device, kind='input')
        return device_info

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        if self.recording:
            self.q_audio.put(indata.copy())
            self.previously_recording = True
        else:
            if self.previously_recording:
                self.q_audio.put(None)
                self.previously_recording = False

    def start_recording(self, file_path=None):
        if self.recording:
            return
        if file_path and file_path not in self.file_list:
            self.file_path = file_path
            self.file_list.append(file_path)
        self._check_file(override=True)
        self.thread_recording = threading.Thread(target=self._create_stream_and_output_file)
        self.thread_recording.start()

    def stop_recording(self):
        self.recording = False
        self.thread_recording.join()

    def _create_stream_and_output_file(self):
        self.recording = True
        try:
            device_info = sd.query_devices(self.device, 'input')
            if self.samplerate is None:
                # soundfile expects an int, sounddevice provides a float
                self.samplerate = int(device_info['default_samplerate'])

            # Make sure the file is opened before recording anything:
            with sf.SoundFile(self.file_path, mode='x', samplerate=self.samplerate,
                              channels=self.channels, subtype=self.subtype) as file:
                with sd.InputStream(samplerate=self.samplerate, device=self.device,
                                    channels=self.channels, callback=self.callback):
                    while True:
                        data = self.q_audio.get()
                        if data is None:
                            break
                        file.write(data)
        finally:
            self.recording = False

    def _check_file(self, override=False):
        if os.path.exists(self.file_path):
            if override:
                os.remove(self.file_path)
            else:
                raise FileExistsError(f'File already exists: {self.file_path}')





def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'audio_file',
        nargs='?',
        metavar='AUDIO_FILE',
        help='Path to audio file to play',
    )

    # args, remaining = parser.parse_known_args()
    args = parser.parse_args(sys.argv[1:])

    print(args)





if __name__ == '__main__':

    print('Hello zombie world!')

    # show list of audio devices
    print(f'List of all audio devices detected:\n{sd.query_devices()}')

    # test recording
    print('\ntesting recording >>')

    # audio_file = 'test_recording.wav'

    rec = RecordAudio()
    print(f'selected recoding device:\n{rec.get_device_info()}\n')

    time.sleep(3)

    for i in range(2):
        print(f'recording #{i + 1} >>')
        audio_file = f'test_recording{i + 1}.wav'
        rec.start_recording(file_path=audio_file)
        print('recording started')
        time.sleep(10)
        rec.stop_recording()
        print('recording stopped >>')

        print('playing recorded audio in 3 seconds>>')
        time.sleep(3)
        # play_audio_pydub(audio_file, format='wav')
        play_audio_sounddevice(audio_file)

    print(f'rec.file_list={rec.file_list}')

    print('overlaying recorded audios in 3 seconds>>')
    time.sleep(3)
    overlay_audios(*rec.file_list)

    # audio_file = '/home/alfredo/workspace/hackTNT_202410/F5-TTS/tests/ref_audio/test_en_1_ref_short.wav'

    # print('playing audio with pydub >>')
    # play_audio_pydub(audio_file, format='wav')

    # print('playing audio with sounddevice >>')
    # play_audio_sounddevice(audio_file)
