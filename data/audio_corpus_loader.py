import torch
import torchaudio

class AudioSpectrogramExtractor:
    """
    Implements Eq. (14)-(15): 16 kHz resampling, 25 ms window (N_w=400),
    10 ms stride (N_s=160), and globally normalized 80-channel log-Mel spectrogram.
    """
    def __init__(self, sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80):
        self.sample_rate = sample_rate
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )

    def __call__(self, waveform: torch.Tensor, orig_sr: int = 16000) -> torch.Tensor:
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)
        if waveform.dim() > 1 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Eq. 15: log10 Mel energy
        mel = self.mel_spec(waveform)
        log_mel = torch.log10(torch.clamp(mel, min=1e-10))
        # Global normalization
        s_mel_tilde = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return s_mel_tilde