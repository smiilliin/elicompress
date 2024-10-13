import numpy as np
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


def elicompress(chunk, scale):
    a = 0
    r = 0.0001

    epochs = 5

    z1 = np.array([i * scale / len(chunk) + z * 1j for i, z in enumerate(chunk)])

    I = np.linspace(0, len(z1) - 1, 30, dtype=int)
    I2 = np.linspace(0, len(z1) - 1, 10, dtype=int)

    z2 = z1[I2]
    z1 = z1[I]

    for _ in range(epochs):
        new_z2 = update_z2(z1, a, z2)
        new_a = a - r * grad_a(z1, a, z2)

        z2 = new_z2
        a = new_a

    return a, z2


a_array = []
z2_array = []
progress = []


def job(n, offset, chunks, scale):
    global a_array, z2_array, progress

    # print(f"Started a thread job {n}")
    for i, chunk in enumerate(chunks):
        a, z2 = elicompress(chunk, scale)
        a_array[offset + i] = a
        z2_array[offset + i] = z2
        progress[n] = i / (len(chunks) - 1)

    progress[n] = 1

    # print(f"Done a thread job {n}")


def compress(data, scale, thread_n=5, chunksize=100):
    global a_array, z2_array, progress

    data = data[: (len(data) // chunksize) * chunksize]
    data_len = len(data)
    chunks = np.split(data, data.size // chunksize)

    a_array = [None for _ in range(len(chunks))]
    z2_array = [None for _ in range(len(chunks))]
    progress = [0 for _ in range(thread_n)]

    chunks_n = len(chunks)
    gap = chunks_n // thread_n
    offset = 0

    threads = []

    for i in range(thread_n):
        if i == thread_n - 1:
            _chunks = chunks[offset:]
        else:
            _chunks = chunks[offset : offset + gap]

        thread = threading.Thread(
            target=job, args=(i, offset, _chunks, scale), daemon=True
        )
        threads.append(thread)
        offset += gap

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return data_len, a_array, z2_array


def print_progress(end=""):
    global progress
    if len(progress) == 0:
        return 0.0

    p = round((sum(progress) / len(progress)) * 100, 2)

    print(
        f"\rCompression progress: {p}% ({progress.count(1)}/{len(progress)})", end=end
    )


def get_comprsesed_size(a_array, z2_array):
    # a + z2 + data_len + scale
    return len(a_array) + len(z2_array) + 2


def save(name, a_array, z2_array, data_len, scale):
    np.savez(name, a=a_array, z=z2_array, d=data_len, s=scale)


def load(name):
    loaded = np.load(name)
    a_array = loaded["a"]
    z2_array = loaded["z"]
    data_len = loaded["d"]
    scale = loaded["s"]
    return a_array, z2_array, data_len, scale


def decompress(a_array, z2_array, data_len, scale):
    y = np.array([])
    size = data_len // len(a_array)

    for i in range(len(a_array)):
        a = a_array[i]
        z2 = z2_array[i]

        x = np.linspace(0, scale, size)
        y = np.append(y, elimf(a, z2, x))

    return y
