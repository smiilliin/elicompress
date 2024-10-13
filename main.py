import librosa
from elicompress import (
    compress,
    decompress,
    print_progress,
    get_comprsesed_size,
    save,
    load,
)
import soundfile as sf
import time
import threading


data, sr = librosa.load("ylangylang.mp3", sr=None)
data = data[: round(sr * 2)]
scale = 6
done = False


def show_progress():
    while not done:
        print_progress()
        time.sleep(1)
    print_progress(end="\n")


thread = threading.Thread(target=show_progress, daemon=True)
thread.start()

data_len, a_array, z2_array = compress(data, scale)
done = True

thread.join()

save("out.npz", a_array, z2_array, data_len, scale)

print(f"Compressed {data_len} to {get_comprsesed_size(a_array,z2_array)}")

a_array, z2_array, data_len, scale = load("out.npz")

data = decompress(a_array, z2_array, data_len, scale)

sf.write("out.wav", data, sr)
