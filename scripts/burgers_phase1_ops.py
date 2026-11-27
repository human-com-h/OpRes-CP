from functools import lru_cache

import numpy as np
from scipy.fft import irfft, rfft
from scipy.ndimage import gaussian_filter1d

from burgers_phase1_data import phi


@lru_cache
def modes(n):
    return np.arange(n//2+1, dtype=float)


def gradient(u):
    k = modes(u.shape[-1]).copy()
    k[-1] = 0
    return irfft(1j*k*rfft(u), n=u.shape[-1])


def divergence_of_square(u):
    n = u.shape[-1]
    fourier = rfft(u, norm="forward")
    fourier[..., -1] = 0
    padded_n = 3*n//2
    up = irfft(fourier, n=padded_n, norm="forward")
    flux = rfft(0.5*up*up, norm="forward")[..., :n//2+1]
    flux[..., -1] = 0
    return irfft(1j*modes(n)*flux, n=n, norm="forward")


def trajectory_residual(u, nu, config):
    """Residual on [t_j,t_{j+1}], using both endpoints of this interval only."""
    n = u.shape[-1]
    mid = (u[:, 1:]+u[:, :-1])*0.5
    lap = irfft(-modes(n)**2*rfft(mid), n=n)
    return (u[:, 1:]-u[:, :-1])/config["frame_dt"] + divergence_of_square(mid) - nu[:, None, None]*lap


def face_departures(velocity, dt):
    n = velocity.shape[-1]
    faces = (velocity+np.roll(velocity, 1, axis=-1))*0.5
    departure = np.arange(n)[None]-dt*faces*n/(2*np.pi)
    jacobian = np.roll(departure, -1, axis=-1)-departure
    jacobian[:, -1] += n
    if jacobian.min() <= 0:
        raise ValueError(f"Folded conservative transport departure map: min={jacobian.min():.4g}")
    cell = np.floor(departure).astype(int)
    return cell % n, cell//n, departure-cell


def transport(z, departure):
    indices, periods, frac = departure
    cumulative = np.cumsum(z, axis=1)
    mass = cumulative[:, -1:]
    primitive = np.take_along_axis(cumulative-z, indices, axis=1)
    primitive += periods*mass+frac*np.take_along_axis(z, indices, axis=1)
    updated = np.roll(primitive, -1, axis=1)-primitive
    updated[:, -1:] += mass
    return updated


def lift(inverse, u, residual, nu, config):
    n = residual.shape[-1]
    substeps = config["advection_substeps"] if inverse == "adv_diff" else 1
    dt = config["frame_dt"]/substeps
    exponent = -dt*(nu[:, None]*modes(n)**2+config["gamma"])
    exp = np.exp(exponent)
    force_coef = dt*phi(exponent)[0]
    forcing = rfft(residual, axis=-1)*force_coef[:, None]
    z = np.zeros_like(residual[:, 0])
    states = []
    zhat = rfft(z)
    for frame in range(residual.shape[1]):
        if inverse == "adv_diff":
            mid = (u[:, frame]+u[:, frame+1])*0.5
            depart = face_departures(mid, dt)
            for _ in range(substeps):
                advected = transport(z, depart)
                z = irfft(exp*rfft(advected)+forcing[:, frame], n=n)
        else:
            zhat = exp*zhat+forcing[:, frame]
            z = irfft(zhat, n=n)
        states.append(z)
    return np.stack(states, axis=1)


def rms(x):
    return np.sqrt(np.mean(x*x, axis=(-2, -1), keepdims=True))


def envelope(z, config):
    width = config["spatial_radius"]*z.shape[-1]/(2*np.pi)
    relative = config["envelope_relative_floor"]*rms(z)
    smooth = gaussian_filter1d(z*z, width, axis=-1, mode="wrap")
    return np.sqrt(smooth+relative*relative)


def mixed(s, lam):
    return np.sqrt((1-lam)*rms(s)**2+lam*s*s)
