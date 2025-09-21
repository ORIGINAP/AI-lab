!pip install resampy
!pip install boto3

import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

import math
import timeit

import librosa
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import resampy
from IPython.display import Audio

#I/O
SAMPLE_WAV = [
    "/content/asg1_audiodata/095522039.m4a",
    "/content/asg1_audiodata/095522040.m4a",
    "/content/asg1_audiodata/095522041.m4a",
    "/content/asg1_audiodata/095522042.m4a"
    ]

def _hide_seek(obj):
    class _wrapper:
        def __init__(self, obj):
            self.obj = obj

        def read(self, n):
            return self.obj.read(n)

    return _wrapper(obj)


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

DEFAULT_OFFSET = 201


def _get_log_freq(sample_rate, max_sweep_rate, offset):
    start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
    return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset


def _get_inverse_log_freq(freq, sample_rate, offset):
    half = sample_rate // 2
    return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))


def _get_freq_ticks(sample_rate, offset, f_max):
    times, freq = [], []
    for exp in range(2, 5):
        for v in range(1, 10):
            f = v * 10**exp
            if f < sample_rate // 2:
                t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
                times.append(t)
                freq.append(f)
    t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
    times.append(t_max)
    freq.append(f_max)
    return times, freq


def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
    max_sweep_rate = sample_rate
    freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
    delta = 2 * math.pi * freq / sample_rate
    cummulative = torch.cumsum(delta, dim=0)
    signal = torch.sin(cummulative).unsqueeze(dim=0)
    return signal


def plot_sweep(
    waveform,
    sample_rate,
    title,
    max_sweep_rate=48000,
    offset=DEFAULT_OFFSET,
):
    x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
    y_ticks = [1000, 5000, 10000, 20000, sample_rate // 2]

    time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
    freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
    freq_y = [f for f in freq if f in y_ticks and 1000 <= f <= sample_rate // 2]

    figure, axis = plt.subplots(1, 1)
    _, _, _, cax = axis.specgram(waveform[0].numpy(), Fs=sample_rate)
    plt.xticks(time, freq_x)
    plt.yticks(freq_y, freq_y)
    axis.set_xlabel("Original Signal Frequency (Hz, log scale)")
    axis.set_ylabel("Waveform Frequency (Hz)")
    axis.xaxis.grid(True, alpha=0.67)
    axis.yaxis.grid(True, alpha=0.67)
    figure.suptitle(f"{title} (sample rate: {sample_rate} Hz)")
    plt.colorbar(cax)

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")


#리샘플링의 의의  = 다운샘플링 경우 크기 축소 또는 음성인식 최적화. 업샘플링의 경우 특정 오디오 처리 모델 요건 맞추는데 사용

#======== audio1 ============
waveform, sample_rate = torchaudio.load(SAMPLE_WAV[0])
Audio(waveform.numpy()[0], rate=sample_rate)

resample_rate = 32000
resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=128)

plot_sweep(resampled_waveform, resample_rate, title="44100>32000 Down_Spectogram")
plot_waveform(waveform, sample_rate)
Audio(resampled_waveform.numpy()[0], rate=resample_rate)

#======== audio2 ============
waveform, sample_rate = torchaudio.load(SAMPLE_WAV[1])
Audio(waveform.numpy()[0], rate=sample_rate)

resample_rate = 32000
resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=128)

plot_sweep(resampled_waveform, resample_rate, title="44100>32000 Down_Spectogram")
plot_waveform(waveform, sample_rate)
Audio(resampled_waveform.numpy()[0], rate=resample_rate)

#======== audio3 ============
waveform, sample_rate = torchaudio.load(SAMPLE_WAV[2])
Audio(waveform.numpy()[0], rate=sample_rate)

resample_rate = 48000
resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=128)

plot_sweep(resampled_waveform, resample_rate, title="44100>48000 Up_Spectogram")
plot_waveform(waveform, sample_rate)
Audio(resampled_waveform.numpy()[0], rate=resample_rate)

#======== audio4 ============
waveform, sample_rate = torchaudio.load(SAMPLE_WAV[3])
Audio(waveform.numpy()[0], rate=sample_rate)

resample_rate = 48000
resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=128)

plot_sweep(resampled_waveform, resample_rate, title="44100>48000 Up_Spectogram")
plot_waveform(waveform, sample_rate)
Audio(resampled_waveform.numpy()[0], rate=resample_rate)