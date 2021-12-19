# source: https://python-sounddevice.readthedocs.io/en/0.4.2/examples.html#recording-with-arbitrary-duration

import queue

from einops import rearrange
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio


CUSTOM_SAMPLERATE = 16000
CUSTOM_CHANNELS = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
asr_model, decoder = None, None


def callback(queue, indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    queue.put(indata.copy())
    
def record_audio():
    device_idx = sd.default.device[0]
    device_info = sd.query_devices(device_idx, 'input')
    n_channels = int(device_info['max_input_channels']) \
      if CUSTOM_CHANNELS <= 0 \
      else CUSTOM_CHANNELS
    samplerate = int(device_info['default_samplerate']) \
      if CUSTOM_SAMPLERATE <= 0 \
      else CUSTOM_SAMPLERATE
    audio_queue = queue.Queue()
    audio_buffer = []
    audio_callback = lambda *args: callback(audio_queue, *args)

    try:
        with sd.InputStream(samplerate=samplerate, device=device_idx,
                            channels=n_channels, callback=audio_callback):
            print('#' * 80)
            print('press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
              audio_buffer.append(audio_queue.get())
    except KeyboardInterrupt:
        pass

    audio_data = np.array(audio_buffer)
    audio_data = rearrange(audio_data, 'b f c -> c (b f)')
    
    return audio_data, samplerate
  
def format_audio(data, samplerate, device, target_samplerate=16000):
  data = torch.tensor(data, device=device)
  
  if data.size(0) > 1:
    data = data.mean(dim=0, keepdim=True)

  if samplerate != target_samplerate:
    transform = torchaudio.transforms.Resample(
      orig_freq=samplerate, new_freq=target_samplerate)
    data = transform(data)

  return data
  
def perform_asr(audio_input, samplerate):
    global asr_model, decoder
    
    if asr_model is None or decoder is None:
        asr_model, decoder, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_stt',
        language='en',
        device=device)
  
    audio_data = format_audio(audio_input, samplerate, device)
    
    output = asr_model(audio_data)
    output_text = decoder(output[0].cpu())
    
    return output_text
    
if __name__ == '__main__':
    audio_input, samplerate = record_audio()
    output_text = perform_asr(audio_input, samplerate)
    print('\n ASR OUTPUT:', output_text)