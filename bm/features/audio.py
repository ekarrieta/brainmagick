# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""All the supported audio features."""

import logging
import math
import os
import typing as tp
import warnings
from pathlib import Path
from typing import Union

import julius
import torch
import torchaudio
import numpy as np
from bm import events
from bm.cache import Cache, MemoryCache
from bm.lib.pitch_calc.yin import compute_yin
from bm.utils import CaptureInit, Frequency
from torch.nn import functional as F

from . import base

logger = logging.getLogger(__name__)


class MelSpectrum(base.Feature, CaptureInit):
    """Outputs the sound waves with the features frequency
    """
    event_kind = "sound"

    def __init__(self, sample_rate: Frequency, n_mels=40, n_fft=512, in_sampling=16_000,
                 normalized=True, use_log_scale=True, log_scale_eps=1e-5,
                 norm_audio: bool = True) -> None:
        super().__init__(sample_rate)
        self.dimension = n_mels
        kwargs = self._init_kwargs
        kwargs.pop('sample_rate')
        self.cache = Cache(self.__class__.__name__, kwargs)

        self.in_sampling = in_sampling
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.use_log_scale = use_log_scale
        self.log_scale_eps = log_scale_eps
        self.normalized = normalized
        self.norm_audio = norm_audio
        self.trans = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.in_sampling, n_mels=self.n_mels,
            n_fft=n_fft, hop_length=self.hop_length, normalized=normalized
        )

        if use_log_scale:
            self.default_value = math.log10(log_scale_eps)

    def _compute(self, filepath: Path, start: float, stop: float) -> torch.Tensor:
        wav, sr = _extract_wav_part(filepath, start, stop)
        wav = torch.mean(wav, dim=0)  # stereo to mono
        if self.norm_audio:
            wav = (wav - wav.mean()) / (1e-8 + wav.std())
        wav = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=self.in_sampling)(wav)

        # Two UserWarnings thrown internally by torch here: "stft will require the return_complex
        # parameter be explicitly" and "The function torch.rfft is deprecated". Remove this once
        # torch library updates to fix this
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            melspec = self.trans(wav)
        if self.use_log_scale:
            melspec = torch.log10(melspec + self.log_scale_eps)
        return melspec

    def get(self, event: events.Sound) -> torch.Tensor:
        melspec = self.cache.get(
            self._compute, filepath=event.filepath,
            start=event.offset, stop=event.offset + event.duration)
        feature_samples = self.sample_rate.to_ind(event.stop - event.start)
        return F.interpolate(melspec[None], feature_samples)[0]


class Pitch(base.Feature, CaptureInit):
    """Pitch from the waveform.
    """

    event_kind = "sound"

    def __init__(self, sample_rate: Frequency, min_f0=100.0, max_f0=350.0, harmonic_thresh=0.1,
                 frame_length_in_samples=256, frame_space_in_samples=64) -> None:
        super().__init__(sample_rate)
        kwargs = self._init_kwargs
        kwargs.pop('sample_rate')
        self.cache = Cache(self.__class__.__name__, kwargs)

        self.frame_length_in_samples = frame_length_in_samples
        self.frame_space_in_samples = frame_space_in_samples
        self.harmonic_thresh = harmonic_thresh
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.in_sampling = 16_000

    @property
    def _cache_params(self):
        return self._init_args_kwargs

    def _compute(self, filepath: Path, start: float, stop: float) -> torch.Tensor:
        wav_stereo, sr = _extract_wav_part(filepath, start, stop)
        wav = torch.mean(wav_stereo, axis=0)  # Stereo to mono
        wav = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=self.in_sampling)(wav)

        pitches, harmonic_rates, argmins, times = compute_yin(
            sig=wav.numpy(),
            sr=self.in_sampling,
            w_len=self.frame_length_in_samples,
            w_step=self.frame_space_in_samples,
            harmo_thresh=self.harmonic_thresh,
            f0_min=self.min_f0,
            f0_max=self.max_f0)
        out = torch.FloatTensor(pitches)
        return out

    def get(self, event: events.Sound) -> torch.Tensor:
        pitches = self.cache.get(
            self._compute, filepath=event.filepath,
            start=event.offset, stop=event.offset + event.duration)
        feature_samples = self.sample_rate.to_ind(event.stop - event.start)
        out = F.interpolate(pitches[None, None], feature_samples)[0, 0]
        return out[None]


class _BaseWav2Vec(base.Feature, CaptureInit):
    """
    Parent class for Wav2VecTr and Wav2VecConv
    """

    event_kind = "sound"
    model_name = "facebook/wav2vec2-large-xlsr-53"

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True, random: bool = False,
                 device: str = "cpu") -> None:
        super().__init__(sample_rate)
        args: tp.Any = self.model_name
        if random:
            args = (self.model_name, random)
        self.cache = Cache("Wav2VecEmbedding", args, mode="memmap")
        self.normalized = normalized
        self.device = device
        self.random = random
        # Huggingface logging
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
        self._model_cache = MemoryCache("Wav2VecEmbedding", "model")
        self._extractor_cache = MemoryCache("Wav2VecEmbedding", "extractor")

    @property
    def model(self) -> tp.Any:
        from transformers import Wav2Vec2Model
        if self.random:
            return self._model_cache.get(self._get_random_model)
        else:
            return self._model_cache.get(Wav2Vec2Model.from_pretrained, self.model_name)

    def _get_random_model(self):
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(self.model_name)
        return Wav2Vec2Model(config)

    @property
    def feature_extractor(self) -> tp.Any:
        from transformers import Wav2Vec2FeatureExtractor
        return self._extractor_cache.get(Wav2Vec2FeatureExtractor.from_pretrained, self.model_name)

    def _preprocess_wav(self, filepath: Union[Path, str],
                        start: float, stop: float) -> torch.Tensor:
        wav, sr = _extract_wav_part(filepath, start, stop)
        logger.debug(
            "Preprocessing Wav on %s, start %.1f, stop %.1f, duration %.1f",
            filepath, start, stop, stop - start)
        wav = torch.mean(wav, dim=0)  # stereo to mono
        model_sr = self.feature_extractor.sampling_rate
        wav = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=model_sr)(wav)

        # [1, T]
        out = self.feature_extractor(wav,
                                     return_tensors="pt",
                                     sampling_rate=model_sr,
                                     do_normalize=self.normalized).input_values
        return out

    def _compute_hidden_states(
            self, name: str, filepath: Path, start: float, stop: float,
            layers: tp.Optional[tp.List[int]] = None) -> torch.Tensor:
        input_values = self._preprocess_wav(filepath=filepath, start=start, stop=stop)

        self.model.to(self.device)
        self.model.eval()  # needs to be in eval mode
        with torch.no_grad():
            outputs = self.model(input_values.to(self.device), output_hidden_states=True)
        out: tp.Any = outputs.get(name)
        if isinstance(out, tuple):
            out = torch.stack(out)
        if layers is not None:
            out = out[layers].mean(0)
        return out.detach().cpu().clone().numpy()

    def _get_cached_tensor(
            self, event: events.Sound, overlap: events.DataSlice, name: str,
            layers: tp.Optional[tp.List[int]] = None,
    ) -> torch.Tensor:
        outputs = self.cache.get(
            self._compute_hidden_states, start=event.offset, stop=event.offset + event.duration,
            filepath=event.filepath, name=name, layers=layers)
        embd_sr = outputs.shape[-2] / event.duration
        # safety, to make sure we extract the right dim... but maybe slow
        if event.duration >= 0.5:
            assert 42 < embd_sr < 52, (f"Unexpected sampling rate for embedding {embd_sr}",
                                       event.duration, outputs.shape[-2])
        # if the above assert fails, event duration may be inconsistent with actual wav duration
        # or the wav2vec output sampling rate has changed.
        # we'd need to either find a way to get the embedding sampling rate independently, or
        # figure out the duration in another way
        sr = Frequency(embd_sr)
        start, stop = [sr.to_ind(x - event.start) for x in (overlap.start, overlap.stop)]
        start = min(start, outputs.shape[-2] - 1)
        stop = max(start + 1, stop)
        chunk = outputs[..., start: stop, :]
        # load into memory (probably unnecessary, but lets avoid weird issues)
        chunk = np.array(chunk, copy=True)
        return torch.from_numpy(chunk)

    def get(self, event: events.Sound) -> torch.Tensor:
        raise RuntimeError(f"Only get_on_overlap is available for {self.__class__.__name__}")


class Wav2VecTransformer(_BaseWav2Vec):
    """Outputs the Wav2Vec transformer layers
    """
    event_kind = "sound"
    dimension = 1024

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 layers: tp.Tuple[int, ...] = (14, 15, 16, 17, 18),
                 random: bool = False,
                 device: str = "cpu") -> None:
        super().__init__(sample_rate=sample_rate, normalized=normalized,
                         device=device, random=random)
        self.layers = layers

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(
            event, overlap=overlap,
            name="hidden_states", layers=list(self.layers))
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        return F.interpolate(outputs[None], overlap.duration_ind)[0]


class Wav2VecConvolution(_BaseWav2Vec):
    """Outputs the Wav2Vec convolutional layers
    """
    event_kind = "sound"
    dimension = 512

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(event, overlap=overlap, name="extract_features")
        # [1, T, D] -> [T, D] -> [D, T]
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        out = F.interpolate(outputs[None], overlap.duration_ind)[0]
        return out


class Wav2VecChunk(_BaseWav2Vec):
    """Outputs a chunk of the waveform compatible to be an input of Wav2Vec Model"""

    event_kind = "sound"
    dimension = 1
    model_name = "facebook/wav2vec2-large-xlsr-53"
    normalizable = False

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 random: bool = False,
                 device: str = "cpu") -> None:
        # Forcing the SR to 16k for this feature (base::FeaturesBuilder()
        # doesn't handle multiple SRs)
        super().__init__(sample_rate=Frequency(16000), normalized=normalized,
                         device=device, random=random)

    @property
    def feature_extractor(self) -> tp.Any:
        from transformers import Wav2Vec2FeatureExtractor

        return self._extractor_cache.get(
            Wav2Vec2FeatureExtractor.from_pretrained, self.model_name
        )

    def get(self, event: events.Sound) -> torch.Tensor:
        # Possible improv.: add cache here to read full .wav once (small time reduction expected)
        wav = self._preprocess_wav(
            filepath=event.filepath,
            start=event.offset,
            stop=event.offset + event.duration,
        )
        return wav


class _BaseHubert(_BaseWav2Vec):
    # The Hubert Base pretrained model from Hugging Face
    model_name = "facebook/hubert-base-ls960"

    @property
    def model(self) -> tp.Any:
        from transformers import HubertModel
        if self.random:
            return self._model_cache.get(self._get_random_model)
        else:
            # Load the HubertModel instead of Wav2Vec2Model
            return self._model_cache.get(HubertModel.from_pretrained, self.model_name)

    def _get_random_model(self):
        from transformers import HubertModel, HubertConfig
        # Use HubertConfig to configure the random model
        config = HubertConfig.from_pretrained(self.model_name)
        return HubertModel(config)

    def _compute_hidden_states(
            self, name: str, filepath: Path, start: float, stop: float,
            layers: tp.Optional[tp.List[int]] = None) -> torch.Tensor:
        input_values = self._preprocess_wav(filepath=filepath, start=start, stop=stop)

        self.model.to(self.device)
        self.model.eval()  # needs to be in eval mode
        with torch.no_grad():
            outputs = self.model(input_values.to(self.device), output_hidden_states=True)
        out: tp.Any = outputs.get(name)
        if isinstance(out, tuple):
            out = torch.stack(out)
        if layers is not None:
            out = out[layers].mean(0)
        return out.detach().cpu().clone().numpy()


class HubertTransformer(_BaseHubert):
    """Outputs the Hubert Base transformer layers"""
    dimension = 768

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 layers: tp.Tuple[int, ...] = (7, 8, 9),
                 random: bool = False,
                 device: str = "cpu") -> None:
        super().__init__(sample_rate=sample_rate, normalized=normalized,
                         device=device, random=random)
        self.layers = layers

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(
            event, overlap=overlap,
            name="hidden_states", layers=list(self.layers))
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        return F.interpolate(outputs[None], overlap.duration_ind)[0]


class HubertConvolution(_BaseHubert):
    """Outputs the Hubert Base convolutional layers"""
    event_kind = "sound"
    dimension = 768

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(event, overlap=overlap, name="last_hidden_state")
        # [1, T, D] -> [T, D] -> [D, T]
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        out = F.interpolate(outputs[None], overlap.duration_ind)[0]
        return out


class HubertChunk(_BaseHubert):
    """Outputs a chunk of the waveform compatible to be an input of the Hubert Base Model"""

    dimension = 1  # This may remain the same as it refers to raw waveforms
    model_name = "facebook/hubert-large-ll60k"

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 random: bool = False,
                 device: str = "cpu") -> None:
        # Forcing the SR to 16k for this feature (base::FeaturesBuilder()
        # doesn't handle multiple SRs)
        super().__init__(sample_rate=Frequency(16000), normalized=normalized,
                         device=device, random=random)

    @property
    def feature_extractor(self) -> tp.Any:
        from transformers import Wav2Vec2FeatureExtractor

        return self._extractor_cache.get(
            Wav2Vec2FeatureExtractor.from_pretrained, self.model_name
        )

    def get(self, event: events.Sound) -> torch.Tensor:
        # Possible improv.: add cache here to read full .wav once (small time reduction expected)
        wav = self._preprocess_wav(
            filepath=event.filepath,
            start=event.offset,
            stop=event.offset + event.duration,
        )
        return wav


class _BaseHubertLarge(_BaseWav2Vec):
    # The Hubert Large pretrained model from Hugging Face
    model_name = "facebook/hubert-large-ls960-ft"

    @property
    def model(self) -> tp.Any:
        from transformers import HubertModel
        if self.random:
            return self._model_cache.get(self._get_random_model)
        else:
            # Load the HubertModel instead of Wav2Vec2Model
            return self._model_cache.get(HubertModel.from_pretrained, self.model_name)

    def _get_random_model(self):
        from transformers import HubertModel, HubertConfig
        # Use HubertConfig to configure the random model
        config = HubertConfig.from_pretrained(self.model_name)
        return HubertModel(config)

    def _compute_hidden_states(
            self, name: str, filepath: Path, start: float, stop: float,
            layers: tp.Optional[tp.List[int]] = None) -> torch.Tensor:
        input_values = self._preprocess_wav(filepath=filepath, start=start, stop=stop)

        self.model.to(self.device)
        self.model.eval()  # needs to be in eval mode
        with torch.no_grad():
            outputs = self.model(input_values.to(self.device), output_hidden_states=True)
        out: tp.Any = outputs.get(name)
        if isinstance(out, tuple):
            out = torch.stack(out)
        if layers is not None:
            out = out[layers].mean(0)
        return out.detach().cpu().clone().numpy()


class HubertTransformerLarge(_BaseHubertLarge):
    """Outputs the Hubert Large transformer layers"""
    dimension = 1024

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 layers: tp.Tuple[int, ...] = (14, 15, 16, 17, 18),
                 random: bool = False,
                 device: str = "cpu") -> None:
        super().__init__(sample_rate=sample_rate, normalized=normalized,
                         device=device, random=random)
        self.layers = layers

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(
            event, overlap=overlap,
            name="hidden_states", layers=list(self.layers))
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        return F.interpolate(outputs[None], overlap.duration_ind)[0]


class HubertConvolutionLarge(_BaseHubertLarge):
    """Outputs the Hubert Large convolutional layers"""
    event_kind = "sound"
    dimension = 1024

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(event, overlap=overlap, name="last_hidden_state")
        # [1, T, D] -> [T, D] -> [D, T]
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        out = F.interpolate(outputs[None], overlap.duration_ind)[0]
        return out


class HubertChunkLarge(_BaseHubertLarge):
    """Outputs a chunk of the waveform compatible to be an input of the Hubert Large Model"""

    dimension = 1  # This may remain the same as it refers to raw waveforms
    model_name = "facebook/hubert-large-ls960-ft"

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 random: bool = False,
                 device: str = "cpu") -> None:
        # Forcing the SR to 16k for this feature (base::FeaturesBuilder()
        # doesn't handle multiple SRs)
        super().__init__(sample_rate=Frequency(16000), normalized=normalized,
                         device=device, random=random)

    @property
    def feature_extractor(self) -> tp.Any:
        from transformers import Wav2Vec2FeatureExtractor

        return self._extractor_cache.get(
            Wav2Vec2FeatureExtractor.from_pretrained, self.model_name
        )

    def get(self, event: events.Sound) -> torch.Tensor:
        # Possible improv.: add cache here to read full .wav once (small time reduction expected)
        wav = self._preprocess_wav(
            filepath=event.filepath,
            start=event.offset,
            stop=event.offset + event.duration,
        )
        return wav


class _BaseHubertXLarge(_BaseWav2Vec):
    # The Hubert X-Large pretrained model from Hugging Face
    model_name = "facebook/hubert-xlarge-ll60k"

    @property
    def model(self) -> tp.Any:
        from transformers import HubertModel
        if self.random:
            return self._model_cache.get(self._get_random_model)
        else:
            # Load the HubertModel instead of Wav2Vec2Model
            return self._model_cache.get(HubertModel.from_pretrained, self.model_name)

    def _get_random_model(self):
        from transformers import HubertModel, HubertConfig
        # Use HubertConfig to configure the random model
        config = HubertConfig.from_pretrained(self.model_name)
        return HubertModel(config)

    def _compute_hidden_states(
            self, name: str, filepath: Path, start: float, stop: float,
            layers: tp.Optional[tp.List[int]] = None) -> torch.Tensor:
        input_values = self._preprocess_wav(filepath=filepath, start=start, stop=stop)

        self.model.to(self.device)
        self.model.eval()  # needs to be in eval mode
        with torch.no_grad():
            outputs = self.model(input_values.to(self.device), output_hidden_states=True)
        out: tp.Any = outputs.get(name)
        if isinstance(out, tuple):
            out = torch.stack(out)
        if layers is not None:
            out = out[layers].mean(0)
        return out.detach().cpu().clone().numpy()


class HubertTransformerXLarge(_BaseHubertXLarge):
    """Outputs the Hubert XLarge transformer layers"""
    dimension = 1280

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 layers: tp.Tuple[int, ...] = (28, 30, 32, 34, 36),
                 random: bool = False,
                 device: str = "cpu") -> None:
        super().__init__(sample_rate=sample_rate, normalized=normalized,
                         device=device, random=random)
        self.layers = layers

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(
            event, overlap=overlap,
            name="hidden_states", layers=list(self.layers))
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        return F.interpolate(outputs[None], overlap.duration_ind)[0]


class HubertConvolutionXLarge(_BaseHubertXLarge):
    """Outputs the Hubert XLarge convolutional layers"""
    event_kind = "sound"
    dimension = 1280

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(event, overlap=overlap, name="last_hidden_state")
        # [1, T, D] -> [T, D] -> [D, T]
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        out = F.interpolate(outputs[None], overlap.duration_ind)[0]
        return out


class HubertChunkXLarge(_BaseHubertXLarge):
    """Outputs a chunk of the waveform compatible to be an input of the Hubert XLarge Model"""

    dimension = 1  # This may remain the same as it refers to raw waveforms
    model_name = "facebook/hubert-xlarge-ll60k"

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 random: bool = False,
                 device: str = "cpu") -> None:
        # Forcing the SR to 16k for this feature (base::FeaturesBuilder()
        # doesn't handle multiple SRs)
        super().__init__(sample_rate=Frequency(16000), normalized=normalized,
                         device=device, random=random)

    @property
    def feature_extractor(self) -> tp.Any:
        from transformers import Wav2Vec2FeatureExtractor

        return self._extractor_cache.get(
            Wav2Vec2FeatureExtractor.from_pretrained, self.model_name
        )

    def get(self, event: events.Sound) -> torch.Tensor:
        # Possible improv.: add cache here to read full .wav once (small time reduction expected)
        wav = self._preprocess_wav(
            filepath=event.filepath,
            start=event.offset,
            stop=event.offset + event.duration,
        )
        return wav


class _BaseHubertXLargeFT_LS960(_BaseHubertXLarge):
    # The Hubert X-Large fientuned in Librispeech model from Hugging Face
    model_name = "facebook/hubert-xlarge-ls960-ft"


class HubertTransformerXLargeFT_LS960(HubertTransformerXLarge):
    """Outputs the Hubert XLarge FT Librispeech transformer layers"""


class HubertConvolutionXLargeFT_LS960(HubertConvolutionXLarge):
    """Outputs the Hubert XLarge FT Librispeech convolutional layers"""


class HubertChunkXLargeFT_LS960(HubertChunkXLarge):
    """Outputs a chunk of the waveform compatible to be an input of the Hubert
    XLarge FT Librispeech Model
    """
    model_name = "facebook/hubert-xlarge-ls960-ft"


class _BaseXLSR300m(_BaseWav2Vec):
    # The XLS-R 300m pretrained model from Hugging Face
    model_name = "facebook/wav2vec2-xls-r-300m"

    @property
    def model(self) -> tp.Any:
        from transformers import Wav2Vec2Model
        if self.random:
            return self._model_cache.get(self._get_random_model)
        else:
            return self._model_cache.get(Wav2Vec2Model.from_pretrained, self.model_name)

    def _get_random_model(self):
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(self.model_name)
        return Wav2Vec2Model(config)

    def _compute_hidden_states(
            self, name: str, filepath: Path, start: float, stop: float,
            layers: tp.Optional[tp.List[int]] = None) -> torch.Tensor:
        input_values = self._preprocess_wav(filepath=filepath, start=start, stop=stop)

        self.model.to(self.device)
        self.model.eval()  # needs to be in eval mode
        with torch.no_grad():
            outputs = self.model(input_values.to(self.device), output_hidden_states=True)
        out: tp.Any = outputs.get(name)
        if isinstance(out, tuple):
            out = torch.stack(out)
        if layers is not None:
            out = out[layers].mean(0)
        return out.detach().cpu().clone().numpy()


class XLSRTransformer300m(_BaseXLSR300m):
    """Outputs the XLS-R 300m transformer layers"""
    dimension = 1024

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 layers: tp.Tuple[int, ...] = (14, 15, 16, 17, 18),
                 random: bool = False,
                 device: str = "cpu") -> None:
        super().__init__(sample_rate=sample_rate, normalized=normalized,
                         device=device, random=random)
        self.layers = layers

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(
            event, overlap=overlap,
            name="hidden_states", layers=list(self.layers))
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        return F.interpolate(outputs[None], overlap.duration_ind)[0]


class XLSRConvolution300m(_BaseXLSR300m):
    """Outputs the XLS-R 300m convolutional layers"""
    event_kind = "sound"
    dimension = 1024

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(event, overlap=overlap, name="last_hidden_state")
        # [1, T, D] -> [T, D] -> [D, T]
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        out = F.interpolate(outputs[None], overlap.duration_ind)[0]
        return out


class XLSRChunk300m(_BaseXLSR300m):
    """Outputs a chunk of the waveform compatible to be an input of the XLS-R 300m Model"""

    dimension = 1  # This may remain the same as it refers to raw waveforms
    model_name = "facebook/wav2vec2-xls-r-300m"

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 random: bool = False,
                 device: str = "cpu") -> None:
        # Forcing the SR to 16k for this feature (base::FeaturesBuilder()
        # doesn't handle multiple SRs)
        super().__init__(sample_rate=Frequency(16000), normalized=normalized,
                         device=device, random=random)

    @property
    def feature_extractor(self) -> tp.Any:
        from transformers import Wav2Vec2FeatureExtractor

        return self._extractor_cache.get(
            Wav2Vec2FeatureExtractor.from_pretrained, self.model_name
        )

    def get(self, event: events.Sound) -> torch.Tensor:
        # Possible improv.: add cache here to read full .wav once (small time reduction expected)
        wav = self._preprocess_wav(
            filepath=event.filepath,
            start=event.offset,
            stop=event.offset + event.duration,
        )
        return wav


class _BaseXLSR1b(_BaseWav2Vec):
    # The XLS-R 1b pretrained model from Hugging Face
    model_name = "facebook/wav2vec2-xls-r-1b"

    @property
    def model(self) -> tp.Any:
        from transformers import Wav2Vec2Model
        if self.random:
            return self._model_cache.get(self._get_random_model)
        else:
            return self._model_cache.get(Wav2Vec2Model.from_pretrained, self.model_name)

    def _get_random_model(self):
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(self.model_name)
        return Wav2Vec2Model(config)

    def _compute_hidden_states(
            self, name: str, filepath: Path, start: float, stop: float,
            layers: tp.Optional[tp.List[int]] = None) -> torch.Tensor:
        input_values = self._preprocess_wav(filepath=filepath, start=start, stop=stop)

        self.model.to(self.device)
        self.model.eval()  # needs to be in eval mode
        with torch.no_grad():
            outputs = self.model(input_values.to(self.device), output_hidden_states=True)
        out: tp.Any = outputs.get(name)
        if isinstance(out, tuple):
            out = torch.stack(out)
        if layers is not None:
            out = out[layers].mean(0)
        return out.detach().cpu().clone().numpy()


class XLSRTransformer1b(_BaseXLSR1b):
    """Outputs the XLS-R 1b transformer layers"""
    dimension = 1280

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 layers: tp.Tuple[int, ...] = (28, 30, 32, 34, 36),
                 random: bool = False,
                 device: str = "cpu") -> None:
        super().__init__(sample_rate=sample_rate, normalized=normalized,
                         device=device, random=random)
        self.layers = layers

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(
            event, overlap=overlap,
            name="hidden_states", layers=list(self.layers))
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        return F.interpolate(outputs[None], overlap.duration_ind)[0]


class XLSRConvolution1b(_BaseXLSR1b):
    """Outputs the XLS-R 1b convolutional layers"""
    event_kind = "sound"
    dimension = 1280

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(event, overlap=overlap, name="last_hidden_state")
        # [1, T, D] -> [T, D] -> [D, T]
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        out = F.interpolate(outputs[None], overlap.duration_ind)[0]
        return out


class XLSRChunk1b(_BaseXLSR1b):
    """Outputs a chunk of the waveform compatible to be an input of the XLS-R 1b Model"""

    dimension = 1  # This may remain the same as it refers to raw waveforms
    model_name = "facebook/wav2vec2-xls-r-1b"

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 random: bool = False,
                 device: str = "cpu") -> None:
        # Forcing the SR to 16k for this feature (base::FeaturesBuilder()
        # doesn't handle multiple SRs)
        super().__init__(sample_rate=Frequency(16000), normalized=normalized,
                         device=device, random=random)

    @property
    def feature_extractor(self) -> tp.Any:
        from transformers import Wav2Vec2FeatureExtractor

        return self._extractor_cache.get(
            Wav2Vec2FeatureExtractor.from_pretrained, self.model_name
        )

    def get(self, event: events.Sound) -> torch.Tensor:
        # Possible improv.: add cache here to read full .wav once (small time reduction expected)
        wav = self._preprocess_wav(
            filepath=event.filepath,
            start=event.offset,
            stop=event.offset + event.duration,
        )
        return wav


class _BaseXLSR2b(_BaseWav2Vec):
    # The XLS-R 2b pretrained model from Hugging Face
    model_name = "facebook/wav2vec2-xls-r-2b"

    @property
    def model(self) -> tp.Any:
        from transformers import Wav2Vec2Model
        if self.random:
            return self._model_cache.get(self._get_random_model)
        else:
            return self._model_cache.get(Wav2Vec2Model.from_pretrained, self.model_name)

    def _get_random_model(self):
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(self.model_name)
        return Wav2Vec2Model(config)

    def _compute_hidden_states(
            self, name: str, filepath: Path, start: float, stop: float,
            layers: tp.Optional[tp.List[int]] = None) -> torch.Tensor:
        input_values = self._preprocess_wav(filepath=filepath, start=start, stop=stop)

        self.model.to(self.device)
        self.model.eval()  # needs to be in eval mode
        with torch.no_grad():
            outputs = self.model(input_values.to(self.device), output_hidden_states=True)
        out: tp.Any = outputs.get(name)
        if isinstance(out, tuple):
            out = torch.stack(out)
        if layers is not None:
            out = out[layers].mean(0)
        return out.detach().cpu().clone().numpy()


class XLSRTransformer2b(_BaseXLSR2b):
    """Outputs the XLS-R 2b transformer layers"""
    dimension = 1920

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 layers: tp.Tuple[int, ...] = (28, 30, 32, 34, 36),
                 random: bool = False,
                 device: str = "cpu") -> None:
        super().__init__(sample_rate=sample_rate, normalized=normalized,
                         device=device, random=random)
        self.layers = layers

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(
            event, overlap=overlap,
            name="hidden_states", layers=list(self.layers))
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        return F.interpolate(outputs[None], overlap.duration_ind)[0]


class XLSRConvolution2b(_BaseXLSR2b):
    """Outputs the XLS-R 2b convolutional layers"""
    event_kind = "sound"
    dimension = 1920

    def get_on_overlap(self, event: events.Sound, overlap: events.DataSlice) -> torch.Tensor:
        outputs = self._get_cached_tensor(event, overlap=overlap, name="last_hidden_state")
        # [1, T, D] -> [T, D] -> [D, T]
        outputs = outputs[0].transpose(0, 1)  # [1, T, D] -> [T, D] -> [D, T]
        out = F.interpolate(outputs[None], overlap.duration_ind)[0]
        return out


class XLSRChunk2b(_BaseXLSR2b):
    """Outputs a chunk of the waveform compatible to be an input of the XLS-R 2b Model"""

    dimension = 1  # This may remain the same as it refers to raw waveforms
    model_name = "facebook/wav2vec2-xls-r-2b"

    def __init__(self, sample_rate: Frequency,
                 normalized: bool = True,
                 random: bool = False,
                 device: str = "cpu") -> None:
        # Forcing the SR to 16k for this feature (base::FeaturesBuilder()
        # doesn't handle multiple SRs)
        super().__init__(sample_rate=Frequency(16000), normalized=normalized,
                         device=device, random=random)

    @property
    def feature_extractor(self) -> tp.Any:
        from transformers import Wav2Vec2FeatureExtractor

        return self._extractor_cache.get(
            Wav2Vec2FeatureExtractor.from_pretrained, self.model_name
        )

    def get(self, event: events.Sound) -> torch.Tensor:
        # Possible improv.: add cache here to read full .wav once (small time reduction expected)
        wav = self._preprocess_wav(
            filepath=event.filepath,
            start=event.offset,
            stop=event.offset + event.duration,
        )
        return wav


def _extract_wav_part(
    filepath: Union[Path, str], onset: float, offset: float
) -> tp.Tuple[torch.Tensor, Frequency]:
    """Extract a chunk of a wave file based on onset and offset in seconds
    """
    info = torchaudio.info(str(filepath))
    sr = Frequency(info.sample_rate)
    wav = torchaudio.load(
        filepath, frame_offset=sr.to_ind(onset), num_frames=sr.to_ind(offset - onset))[0]
    delta = abs(wav.shape[-1] / sr - offset + onset)
    assert delta <= 0.1, (delta, filepath, onset, offset, onset - offset)
    return wav, sr
