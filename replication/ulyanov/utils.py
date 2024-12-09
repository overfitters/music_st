from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile
import torch
from numpy.typing import NDArray
from torch import Tensor


def writefile(out_path: Path, x: NDArray, sr: int):
    """
    write audio file to disk.

    ### Parameters
    out_path (Path): the file to write to
    x (NDArray): the data to write
    sr (int | float): sampling rate

    ### Returns
    None
    """
    soundfile.write(str(out_path), x, sr)


def readfile(filepath: Path) -> tuple[NDArray, int]:
    """
    read an audio file

    ### Parameters
    filepath (Path): the file to read
    ### Returns
    tuple[NDArray, int]: the audio wave as an NDArray and the sampling rate
    """
    if not (filepath.exists() and filepath.is_file()):
        raise FileNotFoundError(f"{filepath} does not exist")

    x, sr = librosa.load(filepath)
    return x, int(sr)


def stft_wave_to_spectrum(
    in_wave: NDArray, n_fft: int = 512
) -> tuple[NDArray, NDArray]:
    """
    convert an audio wave to a spectrum

    ### Parameters
    in_wave (NDArray): the input wave
    n_fft (int): length of windowed signal after padding with zeros

    ### Returns
    tuple[NDArray, NDArray]: the spectrum and the phase
    """
    spectrum = librosa.stft(in_wave, n_fft=n_fft)
    phase = np.angle(spectrum)

    spectrum = np.log1p(np.abs(spectrum))

    return spectrum, phase


def stft_spectrum_to_wave(
    spectrum: NDArray,
    n_fft: int = 512,
    keep_phase: Union[NDArray, None] = None,
) -> NDArray:
    """
    convert a spectrum to an audio wave

    ### Parameters
    spectrum (NDArray): the spectrum - a numpy ndarray
    n_fft (int): length of windowed signal after padding with zeros
    keep_phase (NDArray | None): ...

    ### Returns
    NDArray: the audio wave
    """
    spectrum_exp = np.exp(spectrum) - 1
    phase = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    if keep_phase is not None:
        phase = keep_phase

    out_wave = np.array([])
    for _ in range(50):
        s = spectrum_exp * np.exp(1j * phase)
        out_wave = librosa.istft(s)
        phase = np.angle(librosa.stft(out_wave, n_fft=n_fft))

    return out_wave


def gram(mat: Tensor) -> Tensor:
    """
    ### Parameters
    mat (Tensor): tensor of shape (n_c, n_l)

    ### Returns
    Tensor: Gram matrix of shape (n_c, n_c)
    """
    gram_matrix = torch.matmul(mat, mat.t())
    return gram_matrix


def gram_over_time(mat: Tensor) -> Tensor:
    """
    ### Parameters
    mat (Tensor): tensor of shape (1, n_c, n_h, n_w)

    ### Returns
    Tensor: Gram matrix of mat along time axis, of shape (n_c, n_c)
    """
    m, n_c, n_h, n_w = mat.shape

    mat_unrolled = mat.view(m * n_c * n_h * n_w)
    gram_matrix = torch.matmul(mat_unrolled, mat_unrolled.t())

    return gram_matrix


def compute_content_loss(mat_c: Tensor, mat_g: Tensor) -> Tensor:
    """
    computes the content loss

    ### Parameters
    mat_c (Tensor): tensor of dimension (1, n_c, n_h, n_w)
    mat_g (Tensor): tensor of dimension (1, n_c, n_h, n_w)

    ### Returns
    Tensor: a tensor representing a scalar content loss
    """
    m, n_c, n_h, n_w = mat_g.shape

    g_unrolled = mat_g.view(m * n_c * n_h * n_w)
    c_unrolled = mat_c.view(m * n_c * n_h * n_w)

    cost_content = (
        1.0 / (4 * m * n_c * n_h * n_w) * torch.sum((c_unrolled - g_unrolled) ** 2)
    )

    return cost_content


def compute_layer_style_loss(mat_s: Tensor, mat_g: Tensor) -> Tensor:
    """
    computes the style loss

    ### Parameters
    mat_s (Tensor): tensor of dimension (1, n_c, n_h, n_w)
    mat_g (Tensor): tensor of dimension (1, n_c, n_h, n_w)

    ### Returns
    Tensor: a tensor representing a scalar style cost
    """
    _, n_c, n_h, n_w = mat_g.shape

    gs = gram_over_time(mat_s)
    gg = gram_over_time(mat_g)

    cost_style_layer = 1.0 / (4 * (n_c**2) * (n_h * n_w)) * torch.sum((gs - gg) ** 2)

    return cost_style_layer


def mel_wave_to_spectrum(in_wave: NDArray, sr: int, **kwargs) -> NDArray:
    """
    convert an audio wave to a mel spectogram

    ### Parameters
    in_wave (NDArray): the input wave
    sr (int): the sampling rate of the input wave
    kwargs: any other arguments to be passed to librosa.feature.melspectogram

    ### Returns
    NDArray: the spectrum and the phase
    """
    mel_signal = librosa.feature.melspectrogram(y=in_wave, sr=sr, **kwargs)
    spectrogram = np.abs(mel_signal)
    return spectrogram


def mel_spectrum_to_wave(spectrum: NDArray, sr: int, **kwargs) -> NDArray:
    """
    convert a mel spectrum to an audio wave

    ### Parameters
    spectrum (NDArray): the spectrum - a numpy ndarray
    sr (int): sampling rate
    kwargs: any other arguments to be passed to librosa.feature.inverse.mel_to_audio

    ### Returns
    NDArray: the audio wave
    """
    return librosa.feature.inverse.mel_to_audio(spectrum, sr=sr, **kwargs)

def cqt_wave_to_spectrum(in_wave: NDArray, sr: int, **kwargs) -> NDArray:
    """
    convert an audio wave to a cqt spectogram

    ### Parameters
    in_wave (NDArray): the input wave
    sr (int): the sampling rate of the input wave
    kwargs: any other arguments to be passed to librosa.cqt

    ### Returns
    NDArray: the spectrum and the phase
    """
    cqt_signal = librosa.cqt(y=in_wave, sr=sr, **kwargs)
    cqt_spec = np.abs(cqt_signal)
    return cqt_spec


def cqt_spectrum_to_wave(spectrum: NDArray, sr: int, **kwargs) -> NDArray:
    """
    convert a cqt spectrum to an audio wave

    ### Parameters
    spectrum (NDArray): the spectrum - a numpy ndarray
    sr (int): sampling rate
    kwargs: any other arguments to be passed to librosa.icqt

    ### Returns
    NDArray: the audio wave
    """
    return librosa.icqt(spectrum, sr=sr, **kwargs)
