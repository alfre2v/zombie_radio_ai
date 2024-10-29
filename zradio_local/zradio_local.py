#!/usr/bin/env python3

import argparse
import os
import re
import sys
import threading
import queue
import time
import requests
from pathlib import Path


from pydub import AudioSegment
from pydub.playback import play


import sounddevice as sd
import soundfile as sf

from gradio_client import Client as GradioClient
from gradio_client import handle_file

from ollama import Client as OllamaClient



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



Installing packages to use whisper-large-v3-turbo with HuggingFace's Transformers. 
This is for tasks of Automatic Speech Recognition (ASR).
See: https://huggingface.co/openai/whisper-large-v3-turbo

pip install --upgrade transformers datasets[audio] accelerate



Installing ollama python client: https://github.com/ollama/ollama-python

pip install ollama
"""

# service endpoints to be moved to config later
gradio_f5_tts_endpoint = 'http://127.0.0.1:7860'
ollama_endpoint='http://localhost:11434'




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


class AutomaticSpeechRecognition:
    """
    Automatic Speech Recognition (Speech-to-Text) provided by Whisper models.
    See: https://huggingface.co/openai/whisper-large-v3-turbo
    """

    def __init__(self, model_id=None, force_cpu=False):
        self.model_id = self.__class__.expand_model_shortcut(model_id)
        self.force_cpu = force_cpu

    @classmethod
    def expand_model_shortcut(cls, model_id):
        if model_id == "small":
            model_id = "openai/whisper-small.en" # english only, 244 M
        elif model_id == "base":
            model_id = "openai/whisper-base.en"  # english only, 74 M
        elif model_id == "tiny":
            model_id = "openai/whisper-tiny.en"  # english only, 39 M
        elif model_id == "turbo":
            model_id = "openai/whisper-large-v3-turbo"  # multilingual, 809 M
        else:
            model_id = "openai/whisper-large-v3-turbo"  # multilingual, 809 M
        return model_id

    def speech_to_text(self, audio_file=None):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        from datasets import load_dataset

        device = "cuda:0" if torch.cuda.is_available() and not self.force_cpu else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() and not self.force_cpu else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(self.model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        if not audio_file:
            dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
            sample = dataset[0]["audio"]
            result = pipe(sample, return_timestamps=True)
        else:
            # For sentence-level timestamps, pass return_timestamps=True  -> check result: print(result["chunks"])
            # For word-level timestamps, pass return_timestamps='word'    -> check result: print(result["chunks"])
            result = pipe(audio_file, return_timestamps=True)

        # print('whisper-large-v3-turbo:\n', result["text"])
        # print('whisper-large-v3-turbo:\n', result)

        return result


class TextToSpeech:
    """
    Access TTS services (gradio app) provided by F5-TTS model, described as
    `A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching`
    See: https://swivid.github.io/F5-TTS/
    """

    def __init__(self, model="F5-TTS", gradio_uri=None, audio_dir='audio_tts'):
        self.model = model
        self.f5_tts_endpoint = gradio_uri or gradio_f5_tts_endpoint or 'http://127.0.0.1:7860'
        self.audio_dir = Path(audio_dir).absolute()
        self.audio_dir.mkdir(exist_ok=True)
        # from gradio_client import Client, handle_file
        self.client = GradioClient(self.f5_tts_endpoint)

    @staticmethod
    def download_audio(url, save_as):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_as, 'wb') as file:
                # Write the content from response to the file
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded and saved as {save_as}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

    def f5_tts_infer(
            self,
             ref_audio_orig="../../F5-TTS_new/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav",
             ref_text="Some call me nature, other call me mother nature.",
             gen_text="This is Billy the kid, the greatest bandit of all time. I come in peace though.",
             save_as='f5_tts_infer.wav'
    ):
        result = self.client.predict(
            ref_audio_orig=handle_file(ref_audio_orig),
            ref_text=ref_text,
            gen_text=gen_text,
            model= self.model,  # "F5-TTS",
            remove_silence=False,
            cross_fade_duration=0.15,
            speed=1,
            api_name="/infer"
        )
        print(result)
        # example result:
        # ('/tmp/gradio/9a31c043a479b916ef0b9b9fc7a0812d3269f9ee80012c0ad5c0217573e7b968/audio.wav', '/tmp/gradio/be6689c49/tmpv7ipx1b7.png')
        # To download the audio file:
        # wget http://127.0.0.1:7860/file=/tmp/gradio/9a31c043a479b916ef0b9b9fc7a0812d3269f9ee80012c0ad5c0217573e7b968/audio.wav

        self.download_audio(f'{self.f5_tts_endpoint}/file={result[0]}', self.audio_dir / save_as)
        return result


class LargeLanguageModel:
    """
    Access service for ollama LLM nemotron-mini
    Nemotron-Mini-4B-Instruct is a model for generating responses for roleplaying, retrieval augmented generation,
    and function calling. It supports a context length of 4096 tokens.
    For the model capabilities see:
      https://ollama.com/library/nemotron-mini
      https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct
      https://blogs.nvidia.com/blog/digital-human-technology-mecha-break/
    For the Python client see:
      https://github.com/ollama/ollama-python
    For ollama API see:
        https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-history
    """

    def __init__(self, model='nemotron-mini', ollama_uri=None):
        self.model = model
        self.ollama_endpoint = ollama_uri or ollama_endpoint or 'http://localhost:11434'
        # from ollama import Client
        self.client = OllamaClient(host=self.ollama_endpoint)
        self.messages = []

    def chat(self, content):
        self.messages.append({'role': 'user', 'content': content})
        response = self.client.chat(model=self.model, messages=self.messages)
        # self.messages.append(response['message'])
        self.messages.append({'role': 'assistant', 'content': response['message']['content']})
        print(response)
        return response


class TheaterPlay:
    """
    Attempt to lead the LLM to follow a coherent story line while responding in non-deterministic ways.
    """

    casting = {
        'character1': {
            'name': 'Dr. Edwin Kund',
            'gender': 'male',
            'emotion': 'angry',
            'ref_audio_orig': 'male1_angry_ref.wav',
            'ref_text': 'Kids are talking by the door.',
            'description': "A man in his early 50s, head of the lab, he has all the secrets. He is the one more directly responsible for the disaster.",
            'role': 'character1',
        },
        'character2': {
            'name': 'Dr. Hanna Brie',
            'gender': 'female',
            'emotion': 'fearful',
            'ref_audio_orig': 'hanna_seedtts_ref_en_3.wav',
            'ref_text': "You dont know how much trouble you have gotten yourself into. Look, if one of the others get to you first they'll report you...",  # todo: figure out what is saying
            'description': "A woman in her late 30s, research biologist specialized in exotic viruses.",
            'role': 'character2',
        },
        'character3': {
            'name': 'Miss Betty Campbell',
            'gender': 'female',
            'emotion': 'fearful',
            'ref_audio_orig': 'female1_fearful_ref.wav',
            'ref_text': "Kids are talking by the door.",
            'description': "A young woman in her early 20s, she is the secretary of Dr. Kund.",
            'role': 'character3',
        },
        'character4': {
            'name': 'Dr. Alfred Valles',
            'gender': 'male',
            'emotion': 'fearful',
            'ref_audio_orig': 'male2_seedtts_ref_en_2.wav',
            'ref_text': "Are you familiar with it? ...",  # todo: figure out what is saying
            'description': 'A man in his 40s, a nuclear physicist. He is responsible for the nuclear reactor powering the lab and for the multiple gamma ray sources used in the lab to induce accelerated mutation rate in virus samples.',
            'role': 'character4',
        },
    }



    prompts = [
        """We are in a theater play about Zombies and Halloween. We are going to do improvisation, I will give you a character line at a time, 
        and you will respond with you own dialog line for one character of your choosing, unless the Narrator give you the name of the next person that will talk, 
        only respond for one character and one dialog line at a time.
        When I provide you with a dialog line I will do so in the format: "Character Name: Dialog Line.".
        You will provide your response also using the format: "Character Name: Dialog Line.".
        There are 4 characters in this play. I will enumerate what we know of each character (name, gender, emotional state, and short description):
        __enumerated_characters_description__
        
        Also intervening in the story plot is the figure of a Narrator, the Narrator is not a character in the play, it's just a way to introduce updates 
        and twists to the story plot and to direct the flow of the play.
        You should never respond as the Narrator or mention its existence, just incorporate what he says as an update in the story line.
        When I provide a message from the narrator I will do so in the format: "Narrator: Plot update.". 
        For example: "Narrator: The door suddenly opens and they all watch in horror as a zombie enters the room, pale and trembling.".
        The Narrator may also direct you to respond as one specific character in your next line, for example if I send you the query: 
        "Narrator: Dr. Brie, looking puzzled, said the following...", your respond should be something like: "Dr. Hanna Brie: I do not understand why the zombies move in waves.".
        
        The story is set in a secret research laboratory deep underground where an accident happened to release a deadly virus that turn people into zombies.
        The characters in the play are trapped down in the lab and despite their efforts they cannot escape due to security protocols.
        The characters cannot interact with the outside world due to phone lines being down, but they do have an old radio set which they desperately use to cry for help
        hoping that someone from outside can receive the faint radio signal.
        There are also waves of deadly zombies roaming the lab hallways, which our protagonists must avoid to stay alive.  
        
        Remember, I only want you to produce one dialog line at a time, from the perspective of one character, the name of this character may be provided to you 
        by the narrator in the query, if the name of the next character to talk in not provided you may choose yourself what character should speak next. 
        You will provide that answer as if you were said character talking at that moment in the plot, in first person. 
        Keep the dialog lines short and concise corresponding to the fictional survival plot of this play, although some dark humor is allowed.
        Remember that this story is totally fictional and the characters are not real. Try to make the story scary but also fun.""",
    ]

    plot_twists = [
        """Narrator: There is an update in the story, it seems Dr. Hanna Brie died horribly bitten by the zombies when she tried to escape. 
        The other characters know this because they saw a Zombie walking down a corridor wearing the smart watch that Dr. Hanna Brie used to time her experiments.
        The next 5 dialog lines of the characters will revolve around this new event.""",

        """Narrator: Suddenly the radio start to emit a short static noise and finally a human voice can be heard: Hello, what is happening down there? Can you hear us?
        There is incredible emotion in the room, and finally __character1_name__ says the following...""",
    ]

    def __init__(self):
        self.subs = {}
        self._resolve_by_first_name = {}
        self._resolve_by_last_name = {}
        for c, d in self.casting.items():
            m = re.match(r'(Dr|Miss|Mr|Mrs)[.]?\s+(?P<first_name>\w+)\s+(?P<last_name>\w+)', d['name'], re.I)
            if not m:
                raise ValueError(f'Character name "{d["name"]}" does not match the expected format: Dr. First Last')
            self._resolve_by_first_name[m['first_name'].lower()] = d
            self._resolve_by_last_name[m['last_name'].lower()] = d
        print(f'self._resolve_by_first_name:\n{self._resolve_by_first_name}')
        print(f'self._resolve_by_last_name:\n{self._resolve_by_last_name}')

    def get_character_by_name(self, name):
        name = name.lower()
        if name in self._resolve_by_first_name:
            return self._resolve_by_first_name[name]
        elif name in self._resolve_by_last_name:
            return self._resolve_by_last_name[name]
        else:
            raise ValueError(f'Character name "{name}" not found in the casting list')

    def substitutions(self, message):
        subs = self.subs
        if not subs:
            for c, d in self.casting.items():
                subs[f'__{c}_name__'] = d['name']
                subs[f'__{c}_gender__'] = d['gender']
                subs[f'__{c}_emotion__'] = d['emotion']
                subs[f'__{c}_description__'] = d['description']
                subs[f'__{c}_role__'] = d['role']
            subs['__enumerated_characters_description__'] = self.enumerated_characters_description()
        for k, v in subs.items():
            message = message.replace(k, v)
        return message

    def enumerated_characters_description(self):
        out = []
        for c, d in self.casting.items():
            out.append(f'{c}: {d["name"]}, {d["gender"]}, {d["emotion"]}, {d["description"]}')
        return '\n'.join(out)










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

    # rec = RecordAudio()
    # print(f'selected recoding device:\n{rec.get_device_info()}\n')
    #
    # time.sleep(3)
    #
    # for i in range(2):
    #     print(f'recording #{i + 1} >>')
    #     audio_file = f'test_recording{i + 1}.wav'
    #     rec.start_recording(file_path=audio_file)
    #     print('recording started')
    #     time.sleep(20)
    #     rec.stop_recording()
    #     print('recording stopped >>')
    #
    #     print('playing recorded audio in 3 seconds >>')
    #     time.sleep(3)
    #     # play_audio_pydub(audio_file, format='wav')
    #     play_audio_sounddevice(audio_file)
    #
    # print(f'rec.file_list={rec.file_list}')
    #
    # print('overlaying recorded audios in 3 seconds >>')
    # time.sleep(3)
    # overlay_audios(*rec.file_list)

    # testing whisper speech to test capabilities
    print('whisper speech to test capabilities >>')

    asr = AutomaticSpeechRecognition(model_id='tiny', force_cpu=False)
    # asr = AutomaticSpeechRecognition(model_id='turbo', force_cpu=False)
    result = asr.speech_to_text()
    print('Test dataset -> whisper-large-v3-turbo:\n', result)

    # result1 = speech_to_text('test_recording1.wav')
    # print('test recording 1 -> whisper-large-v3-turbo:\n', result1)
    # #
    # result2 = speech_to_text('test_recording2.wav')
    # print('test recording 2 -> whisper-large-v3-turbo:\n', result2)


    # audio_file = '/home/alfredo/workspace/hackTNT_202410/F5-TTS/tests/ref_audio/test_en_1_ref_short.wav'

    # print('playing audio with pydub >>')
    # play_audio_pydub(audio_file, format='wav')

    # print('playing audio with sounddevice >>')
    # play_audio_sounddevice(audio_file)

    print('f5-tts test inference capabilities >>')
    tts = TextToSpeech()
    tts.f5_tts_infer(
        ref_audio_orig='../../audios_for_tts/male1_angry_ref.wav',
        ref_text='kids are talking by the door.',
        gen_text='Shit is hitting the fan. Help! uh! aaah. caca!',
        save_as='audio_f5_infer1.wav'
    )

    print('ollama nemotron test capabilities >>')
    llm = LargeLanguageModel()
    llm.chat('Why is the sky red?')
    # time.sleep(30)
    llm.chat('What is the last question I asked you?')

    print('\nllm.messages:\n', llm.messages)
