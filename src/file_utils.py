import numpy as np
import io
from scipy.io.wavfile import read, write
import soundfile as sf

def save_audio_file_from_numpy(filename, audio_numpy, sample_rate):
    sf.write(filename, audio_numpy, sample_rate, format="WAV")


def audiobyte_to_numpy(audiobyte):
    return np.frombuffer(audiobyte, dtype=np.int16) / (2 ** 15)

def save_audio_file_from_bytestring(audio_byte_string, filename):
    """
    Save the audio byte string to a file with the given filename.

    Parameters:
    audio_byte_string (bytes): The byte string representing the audio data.
    filename (str): The desired filename for the audio file.
    """
    with open(filename, "wb") as file:
        file.write(audio_byte_string)