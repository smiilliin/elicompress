import librosa
import numpy as np
import soundfile as sf
import threading


def elimf(a, z, x):
    y = np.zeros_like(x, dtype=float)
    l = len(z)
    for i in range(l):
        former = 1
        latter = np.ones_like(x, dtype=float)
        for j in range(1, l):
            former *= x - z[(i + j) % l].real
            latter *= z[i].real - z[(i + j) % l].real
        latter = a / l * (x - z[i].real) + z[i].imag / latter
        y += former * latter

    return y


def grad_a(z1, a, z2):
    grad = 0
    for z in z1:
        grad_t = 2 * (elimf(a, z2, z.real) - z.imag) * np.prod(z.real - z2.real)
        grad += grad_t
    grad /= len(z1)

    if np.abs(grad) > 5:
        return 5 * np.abs(grad) / grad

    return grad


def errf(z1, a, z2):
    err = 0
    for z in z1:
        err += (elimf(a, z2, z.real) - z.imag) ** 2

    return err / len(z1)


def update_z2(z1, a, z2):
    updated_z2 = z2.copy()

    for i, z in enumerate(z2):
        left = np.where(z1.real < z.real, z.real - z1.real, np.inf)
        right = np.where(z1.real > z.real, z1.real - z.real, np.inf)

        if left.size == 0 or right.size == 0:
            continue

        li = left.argmin()
        ri = right.argmin()

        lz2 = z2.copy()
        rz2 = z2.copy()

        if not np.isin(updated_z2, z1[li]).any() and not np.isin(z1, z1[li]).any():
            lz2[i] = z1[li]
        if not np.isin(updated_z2, z1[ri]).any() and not np.isin(z1, z1[ri]).any():
            rz2[i] = z1[ri]

        erri = np.array([errf(z1, a, z2), errf(z1, a, lz2), errf(z1, a, rz2)]).argmin()

        if erri == 0:
            updated_z2[i] = z
        elif erri == 1:
            updated_z2[i] = lz2[i]
        elif erri == 2:
            updated_z2[i] = rz2[i]

    return updated_z2


data, sr = librosa.load("ylangylang.mp3", sr=None)

data = data[: sr * 2]
data = data[: (len(data) // 100) * 100]
data_len = len(data)
data = np.split(data, data.size // 100)


def elicompress(chunk, scale):
    a = 0
    r = 0.0001

    epochs = 5

    z1 = np.array([i * scale / len(chunk) + z * 1j for i, z in enumerate(chunk)])

    I = np.linspace(0, len(z1) - 1, 30, dtype=int)
    I2 = np.linspace(0, len(z1) - 1, 10, dtype=int)

    z2 = z1[I2]
    z1 = z1[I]

    for i in range(epochs):
        # print(i, a, errf(z1, a, z2))

        new_z2 = update_z2(z1, a, z2)
        new_a = a - r * grad_a(z1, a, z2)

        z2 = new_z2
        a = new_a

    return a, z2


scale = 6

a_array = [None for _ in range(len(data))]
z2_array = [None for _ in range(len(data))]
thread_n = 8
threads = []
end = 0


def job(n, start, chunks):
    print(f"Started a thread job {n}")
    for i, chunk in enumerate(chunks):
        a, z2 = elicompress(chunk, scale)
        a_array[start + i] = a
        z2_array[start + i] = z2
        print(f"Done {start+i}, {round((i / (len(chunks)-1)) * 100, 2)}%")

    print(f"Done a thread job {n}")


n = len(data)
gap = n // thread_n
start = 0

for i in range(thread_n):
    if i == thread_n - 1:
        chunks = data[start:]
    else:
        chunks = data[start : start + gap]

    thread = threading.Thread(target=job, args=(i, start, chunks), daemon=True)
    threads.append(thread)
    start += gap

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print(f"Compressed {data_len} to {len(a_array) + len(z2_array) + 1}")

new_y = np.array([])
size = data_len // len(a_array)

for i in range(len(a_array)):
    a = a_array[i]
    z2 = z2_array[i]

    x = np.linspace(0, scale, size)
    new_y = np.append(new_y, elimf(a, z2, x))

sf.write("out.wav", new_y, sr)
