import streamlit as st
import numpy as np
from scipy.fftpack import dct
from scipy.signal import get_window
import plotly.graph_objs as go
import plotly.express as px
import os
import parselmouth


st.title("Espectrograma de Mel modificado")
st.write("¡Bienvenido! Por favor, sube tu archivo de audio.")

# Widget para subir archivo
uploaded_file = st.file_uploader("Selecciona un archivo de audio", type=['wav'])

if uploaded_file is not None:
    # Guardar temporalmente el archivo subido
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Archivo cargado: {uploaded_file.name}")
    st.audio("temp_audio.wav")
else:
    st.warning("Por favor, sube un archivo de audio (.wav)")



def preenfasis(signal, pre_emphasis_coeff=0.97):
    """Aplica un filtro de preénfasis a la señal de audio."""
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])
    return emphasized_signal

def segmentacion(audio, frame_size, hop_size, sample_rate=None):
        """
        Divide la señal en frames usando el mismo método que HTK.
        Args:
            audio: señal de audio
            frame_size: tamaño del frame en muestras
            hop_size: salto entre frames en muestras
            sample_rate: frecuencia de muestreo (opcional, no usado)
        Returns:
            frames: matriz de frames
        """
        # Calcular número de frames como HTK
        signal_length = len(audio)
        num_frames = int(np.ceil(float(signal_length - frame_size + hop_size) / hop_size))
        
        # Crear matriz de frames con padding si es necesario
        pad_length = (num_frames - 1) * hop_size + frame_size
        if pad_length > signal_length:
            pad_signal = np.append(audio, np.zeros(pad_length - signal_length))
        else:
            pad_signal = audio
            
        frames = np.zeros((num_frames, frame_size))
        for i in range(num_frames):
            start = i * hop_size
            frames[i] = pad_signal[start:start + frame_size]
            
        return frames

def ventaneo(signal, frame_size, hop_size, window_type='hamming'):
        """Divide la señal en frames con ventana aplicada."""
        num_frames = 1 + int((len(signal) - frame_size) / hop_size)
        frames = np.zeros((num_frames, frame_size))
        window = get_window(window_type, frame_size, fftbins=True)
        for i in range(num_frames):
            start = i * hop_size
            frames[i] = signal[start:start + frame_size] * window
        return frames

def met_to_freq(mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def Melk(k, fres):
    return 1127.0 * np.log(1.0 + (k - 1) * fres)

def calcular_mfcc(data, fs, n_mfcc=12, win_len=0.025, hop_len=0.01):
    """
    Calcula los coeficientes MFCC de una señal de audio sin librerías externas.
    Args:
        data: señal de audio (numpy array)
        fs: frecuencia de muestreo
        n_mfcc: número de coeficientes MFCC (por defecto 12 como HTK)
        win_len: longitud de ventana en segundos (por defecto 25ms como HTK)
        hop_len: salto entre ventanas en segundos (por defecto 10ms como HTK)
    Returns:
        mfccs: matriz de coeficientes MFCC (ventanas x coeficientes)
        t_mfcc: vector de tiempo de cada ventana
        filter_banks: matriz de filter banks (ventanas x filtros)
    """
    # 1. Pre-emphasis
    emphasized = preenfasis(data, pre_emphasis_coeff=0.97)
    # 2. Framentación
    frame_len = int(win_len * fs)  # 25ms * fs
    frame_step = int(hop_len * fs)  # 10ms * fs
    frames = segmentacion(emphasized, frame_len, frame_step, fs)
    # 3. Ventanamiento
    frames *= ventaneo(np.ones(frame_len), frame_len, frame_len, window_type='hamming')[0]
    # 4. FFT y Power Spectrum
    NFFT = 256  # Como HTK
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

    # 5. Crear y aplicar filtros Mel
    low_freq_mel = freq_to_mel(0)
    high_freq_mel = freq_to_mel(fs/2)
    n_filt = 29  # Como HTK
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt + 2)
    hz_points = met_to_freq(mel_points)
    bin_points = np.floor((NFFT + 1) * hz_points / fs).astype(int)

    # Crear matriz de filter banks
    fbank = np.zeros((n_filt, NFFT // 2 + 1))
    # Construir los filtros triangulares
    for i in range(n_filt):
        for j in range(int(bin_points[i]), int(bin_points[i + 1])):
            fbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
        for j in range(int(bin_points[i + 1]), int(bin_points[i + 2])):
            fbank[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
    # Normalizar los filtros
    fbank = fbank / np.maximum(np.sum(fbank, axis=1)[:, np.newaxis], 1e-8)
    # Aplicar los filter banks
    filter_banks = np.dot(pow_frames, fbank.T)
    # Convertir a dB y manejar valores pequeños
    filter_banks = np.where(filter_banks <= 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # 6. Coeficientes DCT para obtener MFCC (sin transponer)
    mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    # 7. Aplicar la función Melk para el cálculo de los filtros de Mel

    n_filters = 40  # Número de filtros en la escala de Mel
    fres = fs / n_filters  # Resolución en frecuencia
    mel_filters = np.zeros((n_filters, int(win_len * fs)))

    # Crear la matriz de filtros de Mel usando la función Melk
    for k in range(1, n_filters + 1):
        mel_filters[k - 1] = Melk(k, fres)  # Calcula los valores Mel para cada filtro

    # Aplicar los filtros de Mel a la señal
    filter_banks = np.dot(mel_filters, frames.T)  # Multiplicación matricial para aplicar los filtros

    # 8. Calcular los coeficientes MFCC mediante DCT
    mfccs = dct(np.log(filter_banks), type=2, axis=0, norm='ortho')[:n_mfcc]

    # 9. Calcular el vector de tiempo para los MFCC
    t_mfcc = np.arange(0, len(mfccs)) * hop_len

    # Imprimir información de debug
    print('##############MFCC Custom calculated###############')
    print('MFCC Custom shape:', mfccs.shape)
    print('MFCC Custom (frames):\n', mfccs)
    print('Filter Banks Custom shape:', filter_banks.shape)
    print('Filter Banks Custom (frames):\n', filter_banks)

    return mfccs, t_mfcc, filter_banks

def draw_spectrogram(spectrogram, vmin, vmax):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    # Plot with plotly
    data = [go.Heatmap(x=X, y=Y, z=sg_db, zmin=vmin, zmax= vmax, colorscale=colours,)]
    layout = go.Layout(
        title='Espectrograma Praat',
        yaxis=dict(title='Frecuencia (Hz)'),
        xaxis=dict(title='Tiempo (s)'),
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)

def plot_mfcc_spectrogram(mfccs, t_mfcc, filter_banks, colours):
    # Visualizar los filter banks
    fig_banks = go.Figure(data=go.Heatmap(
        z=filter_banks,
        x=t_mfcc,
        colorscale=colours,
    ))
    fig_banks.update_layout(
        title='Espectrograma de Mel (Filter Banks)',
        xaxis_title='Tiempo (s)',
        yaxis_title='Filtros Mel',
    )
    st.plotly_chart(fig_banks)

    # Visualizar los coeficientes MFCC
    fig_mfcc = go.Figure(data=go.Heatmap(
        z=mfccs,
        x=t_mfcc,
        colorscale=colours,
    ))
    fig_mfcc.update_layout(
        title='Coeficientes MFCC',
        xaxis_title='Tiempo (s)',
        yaxis_title='Coeficiente MFCC',
    )
    st.plotly_chart(fig_mfcc)


if uploaded_file is not None:
    # Side Bar #######################################################
    sound = parselmouth.Sound("temp_audio.wav")
    signal = sound.values[0]  # Obtener los valores del audio
    fs = sound.sampling_frequency  # Obtener la frecuencia de muestreo
    
    nyquist_frequency = int(fs/2)
    maximum_frequency = st.sidebar.slider('Frecuencia máxima (Hz)', 5000, nyquist_frequency, 5500)

    named_colorscales = px.colors.named_colorscales()
    default_ix = named_colorscales.index('turbo')
    colours = st.sidebar.selectbox(('Elige la paleta de colores'), named_colorscales, index=default_ix)
    dynamic_range = st.sidebar.slider('Rango Dinámico (dB)', 10, 100, 75)
    window_length = st.sidebar.slider('Longitud de ventana (s)', 0.005, 0.05, 0.05)
    n_mfcc = st.sidebar.slider('Número de coeficientes MFCC', 12, 40, 12)

    st.header("Espectrogramas")
    
    # Mostrar espectrograma tradicional
    spectrogram = sound.to_spectrogram(window_length=window_length, maximum_frequency=maximum_frequency)
    sg_db = 10 * np.log10(spectrogram.values)
    vmin = sg_db.max() - dynamic_range
    vmax = sg_db.max()
    draw_spectrogram(spectrogram, vmin, vmax)

    # Calcular y mostrar MFCC y filter banks
    mfccs, t_mfcc, filter_banks = calcular_mfcc(signal, fs, n_mfcc=n_mfcc, 
                                               win_len=window_length, 
                                               hop_len=window_length/4)
    
    plot_mfcc_spectrogram(mfccs, t_mfcc, filter_banks, colours)