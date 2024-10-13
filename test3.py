import pygame as pg
import sys
import numpy as np
import matplotlib.pyplot as plt


pg.init()
width = 400
height = 400
screen = pg.display.set_mode((width, height))
pg.display.set_caption("elicompress")


drawing = False
started = False
last_mouse_pos = None
ratio = max(width / 2, height / 2) / 20
clock = pg.time.Clock()


def z_to_position(z):
    z = z.conjugate() * ratio + (width / 2 + 1j * height / 2)
    return (z.real, z.imag)


def position_to_z(z):
    z = z[0] + 1j * z[1]
    z = (z - (width / 2 + 1j * height / 2)) / ratio
    return z.conjugate()


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


zs = np.array([])
thetas = np.array([])

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                zs = np.array([])
                thetas = np.array([])
                drawing = True
        if event.type == pg.MOUSEBUTTONUP:
            if event.button == 1:
                started = True
                drawing = False
                rs = np.abs(zs)
                i = np.argsort(thetas)
                thetas = thetas[i]
                rs = rs[i]
                z1 = thetas + rs * 1j

    screen.fill((0, 0, 0))

    pg.draw.circle(screen, (255, 255, 255), z_to_position(0), 3)
    if drawing:
        mouse_pos = pg.mouse.get_pos()

        z = position_to_z(mouse_pos)
        theta = np.mod(np.angle(z), np.pi * 2)

        if not (theta in thetas):
            thetas = np.append(thetas, theta)
            zs = np.append(zs, z)

        if len(zs) > 1:
            pg.draw.lines(
                screen,
                (100, 100, 200),
                False,
                [z_to_position(s) for s in zs],
                3,
            )
    elif started:
        a = 0
        r = 0.0001

        epochs = 150

        I = np.linspace(0, len(z1) - 1, 30, dtype=int)
        I2 = np.linspace(0, len(z1) - 1, 10, dtype=int)

        z2 = z1[I2]
        z1 = z1[I]
        for i in range(epochs):
            print(i, a, errf(z1, a, z2))

            new_z2 = update_z2(z1, a, z2)
            new_a = a - r * grad_a(z1, a, z2)

            z2 = new_z2
            a = new_a

        plt.ylim((0, 10))
        plt.plot(thetas, rs)

        x = np.arange(0, 2 * np.pi, 0.1)
        y = elimf(a, z2, x)
        plt.plot(x, y)

        plt.show()
        started = False

        zs2 = np.exp(1j * x) * np.maximum(y, 0)
    else:
        if len(zs) > 1:
            pg.draw.lines(
                screen,
                (100, 100, 200),
                False,
                [z_to_position(s) for s in zs],
                3,
            )
            pg.draw.lines(
                screen,
                (200, 100, 200),
                False,
                [z_to_position(s) for s in zs2],
                3,
            )

    pg.display.flip()
    clock.tick(60)
