# Copyright (c) HiTZ Zentroa, BCBL and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# pylint: disable=too-many-lines
"""
Neocortical activity tracks the hierarchical linguistic structures of
---------------------------------------------------------------------
self-produced speech during reading aloud
-----------------------------------------

- Paper: https://www.sciencedirect.com/science/article/pii/S1053811920302755
- DOI: https://doi.org/10.1016/j.neuroimage.2020.116788
- Authors: Mathieu Bourguignon, Nicola Molinaro, Mikel Lizarazu, Samu Taulu,
  Veikko Jousmäki, Marie Lallier, Manuel Carreiras, Xavier De Tiège.

Dataset where 18 subjects did around 15 mins of speech each, with three tasks:
producing speech, listening to another speech, and playing back their own
speech, being each of the tasks around 5 mins.

There are two stories distributed randomnly between the subjects, where they
pronounced one of the stories, and listened to the other story from another
person.

Each task is in a different MEG file in fif format (MEGIN Elekta Oy) and an
external wav file with the audio. Additionally, transcriptions (by-hand) and
TextGrid files have been generated with MFA with Word and Phone annotations.
This two are not present in the original dataset, but can be share on demand.
To synchronize the WAV and MEG recordings, a realignment file exists in MATLAB
with `dec` (time shift applied to align the MEG signal, in negative) and `tds`
(array of indices used for downsampling the signal) values.

Most of this code has been based in `gwilliams2022.py`, `broderick2019.py` and
`schoffelen2019/` dataset examples in the original code from Défossez (2023).

Installation
------------

The dataset needs to be put in `data/bourguignon2020/download` subdirectory,
for example:

```shell
$ mkdir -p data/bourguignon2020
$ ln -s ~/datasets/bourguignon2020 data/bourguignon2020/download
```

Tests
-----

The source here passes the style tests of `black`, `pylint` and `flake8`:

```shell
$ black --check bm/bourguignon2020.py bm/studies/test_bourguignon2020.py
$ pylint bm/bourguignon2020.py bm/studies/test_bourguignon2020.py
$ flake8 bm/bourguignon2020.py bm/studies/test_bourguignon2020.py
```

Integration and unit tests:

```shell
$ pytest bm/studies/test_bourguignon2020.py
```
"""

from itertools import product
import os
from pathlib import Path
import typing as tp
import warnings
import regex as re

import mne
import spacy
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.wavfile import read as read_wav

from bm.lib import textgrid
from bm.lib.textgrid import Entry

from . import api
from . import utils
from ..events import extract_sequence_info


SPACY_MODEL = "es_core_news_sm"
TASKS = ("listen", "playback", "self")


def get_subjects(path: Path) -> list:
    """
    Get the list of subjects from `selfpINT` directories.

    Parameters
    ----------
    path : Path
        The path of the MEG directory of the dataset.

    Returns
    -------
    list
        The list of subjects as `str` with the full subject name:
        `"selfp1"`, `"selfp2"`, ...
    """
    subjects = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        matches = re.match(r"selfp(\d+)", entry)
        if os.path.isdir(full_path) and matches is not None:
            subject = matches.group(0)
            subjects.append(subject)
    return subjects


class StudyPaths(utils.StudyPaths):
    """
    A class to manage and handle file paths used in the Bourguignon2020 study.

    Attributes
    ----------
    megs : Path
        A Path object pointing to the directory containing MEG data.

    Methods
    -------
    __init__()
        Initializes the StudyPaths object for Bourguignon2020Recording study.
    """

    def __init__(self) -> None:
        """
        Initialize the StudyPaths object for the Bourguignon2020Recording
        study.
        """
        super().__init__(Bourguignon2020Recording.study_name())
        self.megs = self.download / "meg"


class RecordingPaths:
    """
    A class to manage the file paths associated with a specific recording in a
    study.

    Attributes
    ----------
    subject : int
        An identifier for the subject in the study.
    task : str
        The task associated with the recording (e.g., 'playback', 'self').
    wav_task : str
        Adjusted task name used for wav file naming. Set to 'self' if task is
        'playback'.
    path : Path
        The base path to the directory containing the MEG data.

    Methods
    -------
    subject_dir()
        Returns the directory path for the subject.
    raw_path()
        Returns the path to the raw fif file for the recording.
    realign_path()
        Returns the path to the realignment data file (MATLAB format) for the
        recording.
    wav_path()
        Returns the path to the normalized wav file for the recording.
    txt_path()
        Returns the path to the corresponding text file for the recording.
    textgrid_path()
        Returns the path to the TextGrid file for the recording.
    validate()
        Checks if all essential files for the recording exist.
    """

    def __init__(self, subject: int, task: str) -> None:
        """
        Initializes the RecordingPaths object with a subject identifier and
        task name.

        Parameters
        ----------
        subject : int
            An identifier for the subject in the study.
        task : str
            The task associated with the recording.
        """
        self.subject = subject
        self.task = task
        self.wav_task = "self" if task == "playback" else task
        self.path = StudyPaths().megs

    def subject_dir(self):
        """
        Returns the directory path for the subject.

        Returns
        -------
        Path
            The path to the subject's directory.
        """
        path = self.path / self.subject
        return path

    def raw_path(self) -> Path:
        """
        Returns the path to the raw fif file for the recording.

        Returns
        -------
        Path
            The path to the raw fif file.
        """
        path = self.subject_dir() / f"{self.subject}_{self.task}_tsss.fif"
        return path

    def realign_path(self) -> Path:
        """
        Returns the path to the realignment data file (MATLAB format) for the
        recording.

        Returns
        -------
        Path
            The path to the realignmnet file.
        """
        path = self.subject_dir() / f"{self.subject}_{self.task}_realign.mat"
        return path

    def wav_path(self) -> Path:
        """
        Returns the path to the normalized wav file for the recording.

        Returns
        -------
        Path
            The path to the wav file.
        """
        path = self.subject_dir() / f"{self.subject}_{self.wav_task}_norm.wav"
        return path

    def txt_path(self) -> Path:
        """
        Returns the path to the corresponding transcription file for the
        recording.

        Returns
        -------
        Path
            The path to the txt file.
        """
        path = self.subject_dir() / f"{self.subject}_{self.wav_task}_norm.txt"
        return path

    def textgrid_path(self) -> Path:
        """
        Returns the path to the TextGrid file for the recording.

        Returns
        -------
        Path
            The path to the TextGrid file.
        """
        path = self.subject_dir() / f"{self.subject}_{self.wav_task}_norm.TextGrid"
        return path

    def validate(self) -> bool:
        """
        Checks if all essential files for the recording exist.

        Returns
        -------
        bool
            True if all files exist, False otherwise.
        """
        return (
            self.raw_path().exists()
            and self.realign_path().exists()
            and self.wav_path().exists()
            and self.txt_path().exists()
            and self.textgrid_path().exists()
        )


def find_nearest_meg_sample(wav_sample, array, start=0):
    """
    Function to find the nearest MEG sample number to a given WAV sample
    number.

    Parameters
    ----------
    wav_sample : int
        The WAV sample number to find in the array.
    array : list
        The array where each index is a MEG sample number and value is the WAV
        sample number.
    start : int
        The index to start searching from within the array.

    Returns
    -------
    int
        The index (MEG sample number) corresponding to the nearest WAV sample
        number.
    """
    prev_diff = float("inf")

    for meg_sample in range(start, len(array)):
        wav_sample_in_array = array[meg_sample]
        diff = abs(wav_sample_in_array - wav_sample)

        if diff > prev_diff:
            # The difference started increasing, so the previous index was the
            # closest
            return meg_sample - 1

        prev_diff = diff

    # If we reached the end without an increasing difference, return the last
    # index
    return len(array) - 1


class _BourguignonMetadata:
    """
    A class for handling and parsing metadata associated with the Bourguignon
    dataset.

    This class provides methods to parse transcription text files and MEG
    alignment data from MATLAB matrices.

    Attributes
    ----------
    raw_data : type
        MEG recording Raw signal data.
    raw_sample_rate : type
        MEG Raw signal sample rate in Hz.
    wav_sample_rate : type
        WAV audio signal sample rate in Hz.
    nlp : type
        spaCy NLP model to parse the text corpora into sentences and words
        (in Spanish).
    _cache : type
        Caches pre-processed events metadata.

    Methods
    -------
    parse_txt(subject: str, task: str)
        Parses transcription text files into a DataFrame.
    parse_realign(subject: str, task: str)
        Parses MEG alignment data from MATLAB matrices.
    load_raw(self, subject: str, task: str)
        Loads and returns the raw MEG data for a given subject and task.
    raw_sr(self, subject: str, task: str)
        Returns the sample rate of the raw MEG data.
    load_wav(self, subject: str, task: str)
        Loads and returns the wave (wav) audio data for a given subject and
        task.
    wav_sr(self, subject: str, task: str)
        Returns the sample rate of the wave (wav) audio data.
    realign_textgrid(self, parts: dict, realign: dict, subject: str, task: str)
        Realigns the TextGrid annotation data with the MEG data.
    parse_textgrid(self, subject: str, task: str)
        Parses a TextGrid file and aligns it with MEG data for the specified
        subject and task.
    lazy_init(self)
        Initializes resources required by the metadata processor lazily.
    __call__(self, subject: str, task: str)
        Processes and retrieves metadata for a given subject and task.
    process(self, subject: str, task: str)
        Generates metadata by merging transcriptions and TextGrid information.
    """

    def __init__(self):
        """
        Initializes the _BourguignonMetadata object with default values.
        """
        self.raw_data = None
        self.raw_sample_rate = None
        self.wav_sample_rate = None
        self.nlp = None
        self._cache = None

    def parse_txt(self, subject: str, task: str) -> pd.DataFrame:
        """
        Parse transcriptions to word and sentences into a DataFrame.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing parsed words and sentences from the
            transcription text.
        """
        # read text
        paths = RecordingPaths(subject, task)
        txt_path = paths.txt_path()
        with open(txt_path, "r", encoding="utf-8") as file_handle:
            txt = file_handle.read()

        # tokenize text
        doc = self.nlp(txt)

        # retrieve word and sentences
        df = []
        for sequence_id, sent in enumerate(doc.sents):
            seq_uid = str(sent).replace("\n", "")
            for word_id, word in enumerate(sent):
                word_ = re.sub(r"\W+", "", str(word))
                if len(word_) == 0:
                    continue
                df.append(
                    {
                        "word": word_,
                        "original_word": str(word),
                        "word_id": word_id,
                        "sequence_id": sequence_id,
                        "sequence_uid": seq_uid,
                    }
                )
        df = pd.DataFrame(df)
        return df

    def parse_realign(self, subject: str, task: str) -> pd.DataFrame:
        """
        Parse MEG alignment data from MATLAB matrices file.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        dict
            A dictionary containing 'tds' and 'dec' keys with MEG alignment
            data.
        """
        paths = RecordingPaths(subject, task)
        realign_path = paths.realign_path()
        mat = loadmat(realign_path)
        realign = {
            "tds": list(map(int, mat["tds"][0])),
            "dec": int(mat["dec"][0][0]),
        }
        return realign

    def load_raw(self, subject: str, task: str) -> mne.io.RawArray:
        """
        Loads and returns the raw MEG data for a given subject and task.

        If the raw data is already loaded, it returns the cached data.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        mne.io.RawArray
            The raw MEG data as an MNE RawArray object.
        """
        if self.raw_data is not None:
            return self.raw_data

        paths = RecordingPaths(subject, task)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*does not conform to MNE naming conventions.*",
                category=RuntimeWarning,
            )
            raw = mne.io.read_raw_fif(paths.raw_path())
        self.raw_sample_rate = raw.info["sfreq"]
        picks = {
            "meg": True,
            "eeg": False,
            "stim": False,
            "eog": False,
            "ecg": False,
            "misc": False,
        }
        self.raw_data = raw.pick_types(**picks)
        return self.raw_data

    def raw_sr(self, subject: str, task: str) -> int:
        """
        Returns the sample rate of the raw MEG data.

        Loads the raw data if not already loaded.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        int
            The sample rate of the raw MEG data.
        """
        _ = self.load_raw(subject, task)
        return self.raw_sample_rate

    def load_wav(self, subject: str, task: str) -> np.array:
        """
        Loads and returns the wave (wav) audio data for a given subject and
        task.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        np.array
            The wave audio data.
        """
        paths = RecordingPaths(subject, task)
        sfreq, wav = read_wav(paths.wav_path())
        self.wav_sample_rate = sfreq
        return wav

    def wav_sr(self, subject: str, task: str) -> int:
        """
        Returns the sample rate of the wave (wav) audio data.

        Loads the wave data if not already loaded.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        int
            The sample rate of the wave audio data.
        """
        if self.wav_sample_rate is not None:
            return self.wav_sample_rate

        _ = self.load_wav(subject, task)
        return self.wav_sample_rate

    def realign_textgrid(
        self, parts: dict, realign: dict, subject: str, task: str
    ) -> pd.DataFrame:
        """
        Realigns the TextGrid annotation data with the MEG data.

        Parameters
        ----------
        parts : dict
            Dictionary containing the TextGrid annotation data.
        realign : dict
            Dictionary containing realignment data from MEG and audio
            recordings.
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        pd.DataFrame
            DataFrame with realigned TextGrid annotation data.
        """
        tds, dec = realign["tds"], realign["dec"]
        raw_sr = self.raw_sr(subject, task)
        wav_sr = self.wav_sr(subject, task)

        parts_aligned: tp.Dict[str, tp.Any] = {}
        for tier, rows in parts.items():
            # Make sure they are sorted by start time, just in case:
            rows.sort(key=lambda x: float(x.start))

            parts_aligned[tier] = []
            prev_onset = 0
            for row in rows:
                # Calculate the sample from event time:
                onset_wav = row.start * wav_sr
                offset_wav = row.stop * wav_sr

                # Realign the event from the wav sample to the MEG sample:
                onset_meg = find_nearest_meg_sample(onset_wav, tds, prev_onset)

                # Optimization to avoid traversing from 0 for next samples:
                prev_onset = onset_meg
                offset_meg = find_nearest_meg_sample(offset_wav, tds, prev_onset)
                prev_onset = offset_meg

                # Calculation event time in seconds from sample number:
                onset = float(onset_meg + dec) / raw_sr
                offset = float(offset_meg + dec) / raw_sr
                parts_aligned[tier].append(
                    Entry(start=onset, stop=offset, name=row.name, tier=row.tier)
                )

            # Fix stop times if too long due to rounding:
            for i in range(len(parts_aligned[tier]) - 1):
                if parts_aligned[tier][i].stop > parts_aligned[tier][i + 1].start:
                    parts_aligned[tier][i].stop = parts_aligned[tier][i + 1].start

        return parts_aligned

    def parse_textgrid(self, subject: str, task: str) -> pd.DataFrame:
        """
        Parses a TextGrid file and aligns it with MEG data for the specified
        subject and task.

        This method reads the TextGrid annotations for words and phonemes
        (phones) and realigns them with the MEG recordings. It handles the
        timing differences between WAV and MEG recordings and creates a unified
        DataFrame with both orthographic (word) and phonetic (phoneme) data.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the parsed and aligned TextGrid data,
            including timing, word, and phoneme information.

        Notes
        -----
        - The method assumes that the words and phonemes are present in the
          tiers named 'words' and 'phones' in the TextGrid file.
        - The method also filters out empty entries and special tags like
          '<p:>' and 'spn'.
        """
        # Read TextGrid
        paths = RecordingPaths(subject, task)
        textgrid_path = paths.textgrid_path()
        tgrid = textgrid.read_textgrid(str(textgrid_path))
        parts: tp.Dict[str, tp.Any] = {}
        for p in tgrid:
            if p.name not in ["", "<p:>", "spn"]:  # Remove empty entries
                parts.setdefault(p.tier, []).append(p)

        # Realign WAV time stamps to MEG recordings:
        realign = self.parse_realign(subject, task)
        parts = self.realign_textgrid(parts, realign, subject, task)

        # Separate orthographics, phonetics, and phonemes
        words = parts["words"]
        phonemes = parts["phones"]  # FIXME phonemes != phones

        # Def concatenate orthographics and phonetics
        rows: tp.List[tp.Dict[str, tp.Any]] = []
        for word_index, word in enumerate(words):
            rows.append(
                {
                    "kind": "word",
                    "start": word.start,
                    "stop": word.stop,
                    "word_index": word_index,
                    "word": word.name,
                    "modality": "audio",
                }
            )

        # Add timing of individual phonemes
        starts = np.array([i["start"] for i in rows])
        # Phonemes and starts are both ordered so this could be further
        # optimized if need be
        for phoneme in phonemes:
            idx = np.where(phoneme.start < starts)[0]
            idx = idx[0] - 1 if idx.size else len(rows) - 1
            row = rows[idx]
            rows.append(
                {
                    "kind": "phoneme",
                    "start": phoneme.start,
                    "stop": phoneme.stop,
                    "word_index": row["word_index"],
                    "word": row["word"],
                    "phoneme": phoneme.name,
                    "modality": "audio",
                }
            )
        # Sorting is needed for later optimization to align with the MEG
        rows.sort(key=lambda x: float(x["start"]))
        df = pd.DataFrame(rows)
        return df

    def lazy_init(self) -> None:
        """
        Initializes resources required by the metadata processor lazily.

        This includes loading the SpaCy model for natural language processing
        and initializing the events cache.
        """
        if self.nlp is None:
            if not spacy.util.is_package(SPACY_MODEL):
                spacy.cli.download(SPACY_MODEL)
            self.nlp = spacy.load(SPACY_MODEL)
            self._cache: dict = {}

    def __call__(self, subject: str, task: str) -> pd.DataFrame:
        """
        Processes and retrieves metadata for a given subject and task.

        Lazily initializes necessary resources and caches processed data for
        reuse.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing merged and processed metadata.
        """
        self.lazy_init()
        recording_uid = f"{subject}_task{task}"
        if recording_uid not in self._cache:
            self._cache[recording_uid] = self.process(subject, task)
        return self._cache[recording_uid].copy()

    def process(self, subject: str, task: str) -> pd.DataFrame:
        """
        Generates metadata by merging transcriptions and TextGrid information.

        Processes text transcriptions and TextGrid data, aligning them with MEG
        data. The resulting DataFrame contains detailed event information.

        Returned columns:

        - kind: word, phoneme.
        - modality: audio
        - start: event onset in seconds.
        - stop: event offset in seconds.
        - word_id: word index in the sentence.
        - word: the normalized word.
        - original_word: the word before normalization.
        - phoneme_id
        - phoneme
        - sequence_id: sentence number.
        - sequence_uid: sentence.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns such as 'kind', 'modality', 'start',
            'stop', 'word_id', 'word', 'original_word', 'phoneme_id',
            'phoneme', 'sequence_id', and 'sequence_uid'.
        """
        # Read text transcriptions and parse with spacy
        text = self.parse_txt(subject, task)

        # Read textgrid, and realign it with MEG
        txtgrid = self.parse_textgrid(subject, task)
        txtgrid.drop(columns=["word_index"], inplace=True)  # incorrect index

        # Separate word and phoneme data from TextGrid
        txtgrid_words = txtgrid[txtgrid["kind"] == "word"]
        txtgrid_phonemes = txtgrid[txtgrid["kind"] == "phoneme"]

        # Align words using the match_list function
        i, j = utils.match_list(
            txtgrid_words["word"].str.lower(), text["word"].str.lower()
        )

        # Use indices i and j to align rows, instead of merging on "word"
        aligned_txtgrid = txtgrid_words.iloc[i].reset_index(drop=True)
        aligned_text = text.iloc[j].reset_index(drop=True)

        # Reset word_id per sequence_id
        aligned_text["word_id"] = aligned_text.groupby("sequence_id").cumcount()

        # Merge TextGrid word data with transcription word data
        aligned_text.drop(columns=["word"], inplace=True)  # also in txtgrid
        merged_words = pd.concat([aligned_txtgrid, aligned_text], axis=1)
        # merged_words.rename({"word_id": "word_index"}, inplace=True)

        # Now reintegrate phoneme data iterate over merged words, and for each
        # word, find the corresponding phonemes
        columns = ["word_id", "original_word", "sequence_id", "sequence_uid"]
        # Create a copy of the slice to avoid the SettingWithCopyWarning
        txtgrid_phonemes = txtgrid_phonemes.copy()
        for _, word_row in merged_words.iterrows():
            # Find phonemes that start before this word and after the previous
            # word
            phonemes_for_word = txtgrid_phonemes[
                (txtgrid_phonemes["start"] >= word_row["start"])
                & (txtgrid_phonemes["start"] < word_row["stop"])
            ]

            # Add word metadata to these phonemes
            for phoneme_index, _ in phonemes_for_word.iterrows():
                for col in columns:
                    txtgrid_phonemes.at[phoneme_index, col] = word_row[col]

        # Concatenate merged word data with phoneme data, and sort it
        merged_df = pd.concat([merged_words, txtgrid_phonemes]).sort_values(
            by=["start", "kind"], ascending=[True, False]
        )

        # Add sound event
        dec = self.parse_realign(subject, task)["dec"]
        paths = RecordingPaths(subject, task)
        raw_sr = self.raw_sr(subject, task)
        start = -float(dec) / raw_sr
        wav_sr = self.wav_sr(subject, task)
        stop = start + (len(self.load_wav(subject, task)) / wav_sr)
        sound = {
            "start": start,
            "stop": stop,
            "kind": "sound",
            "filepath": paths.wav_path(),
        }
        merged_df = pd.concat([pd.DataFrame([sound]), merged_df], ignore_index=True)

        return merged_df


class Bourguignon2020Recording(api.Recording):
    """
    A class representing recordings from the Bourguignon2020 study.

    This class extends `api.Recording` and provides specific functionalities
    for handling and processing recordings from the Bourguignon2020 dataset.

    Attributes
    ----------
    data_url : str
        URL for the dataset (if publicly available).
    paper_url : str
        URL of the scientific paper related to the dataset.
    doi : str
        Digital Object Identifier (DOI) of the related paper.
    licence : str
        License under which the dataset is distributed.
    modality : str
        Modality of the recording: audio.
    language : str
        Language of the audio recordings.
    device : str
        Device used for the recording (e.g., MEG).
    description : str
        Brief description of the study and dataset.
    _metadata : _BourguignonMetadata
        Instance of _BourguignonMetadata for metadata processing.

    Methods
    -------
    download()
        Placeholder method for dataset download functionality.
    iter(modality='all')
        Class method to iterate over all recordings in the dataset.
    __init__(subject, task)
        Initializes a recording with the specified subject and task.
    _load_raw()
        Loads and returns the raw MEG data.
    _load_events()
        Loads and processes events metadata.
    """

    data_url = ""
    paper_url = "https://www.sciencedirect.com/science/article/pii/S1053811920302755"
    doi = "https://doi.org/10.1016/j.neuroimage.2020.116788"
    licence = ""
    modality = "audio"
    language = "spanish"
    device = "meg"
    description = "18 subjects produced and listened to 5min stories."
    _metadata = _BourguignonMetadata()

    @classmethod
    def download(cls) -> None:
        """
        Placeholder method for dataset download functionality.

        As the dataset is non-public and must be already downloaded; this
        method does nothing.
        """

    # pytest: disable=arguments-differ
    @classmethod
    def iter(  # pylint: disable=arguments-differ
        cls, modality: str = "all"
    ) -> tp.Iterator["Bourguignon2020Recording"]:
        """
        Returns a generator that iterates over all recordings in the dataset.

        Parameters
        ----------
        modality : str, optional
            The modality to filter the recordings, by default 'all'. Other
            possible values refer to the task: 'listen', 'playback', 'self'.
            The tasks can also be combined with an underscore ('_'), for
            example: 'listen_playback'.

        Yields
        ------
        Bourguignon2020Recording
            An instance of Bourguignon2020Recording for each valid recording.
        """
        # download, extract, organize
        cls.download()

        # List all recordings: depends on study structure
        paths = StudyPaths()
        subjects = get_subjects(paths.megs)

        # Filter the tasks to read if modality set:
        tasks = modality.split("_") if modality in TASKS else TASKS

        for subject, task in product(subjects, tasks):
            rec_paths = RecordingPaths(subject, task)
            if not rec_paths.validate():
                continue

            recording = cls(subject=subject, task=task)
            yield recording

    def __init__(self, subject: str, task: str) -> None:
        """
        Initializes a Bourguignon2020Recording with the specified subject and
        task.

        Parameters
        ----------
        subject : str
            The subject identifier.
        task : str
            The task identifier.
        """
        self.raw_data = None
        self.raw_sample_rate = None
        recording_uid = f"{subject}_task{task}"
        super().__init__(subject_uid=subject, recording_uid=recording_uid)
        self.subject = subject
        self.task = task

    def _load_raw(self) -> mne.io.RawArray:
        """
        Loads and returns the raw MEG data for the recording.

        Returns
        -------
        mne.io.RawArray
            The raw MEG data.
        """
        # FIXME here raw is read at least twice (the metadata class and here)
        if self.raw_data is not None:
            return self.raw_data
        self.raw_data = self._metadata.load_raw(
            self.subject, self.task
        )  # pylint disable=
        self.raw_sample_rate = self._metadata.raw_sample_rate
        return self.raw_data

    def _load_events(self) -> pd.DataFrame:
        """
        Loads and processes events metadata for the recording.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing processed event metadata.
        """
        events = self._metadata(self.subject, self.task)
        events["duration"] = events["stop"] - events["start"]
        events[["language", "modality"]] = self.language, self.modality
        events["condition"] = "sentence"
        events["word_index"] = events["word_id"]
        events = extract_sequence_info(events, phoneme=True)

        # Remove all the events starting before the MEG recording
        events = events[(events["start"] >= 0) | (events["kind"] == "sound")]

        # Regenerate word indexes, in case some have been removed above
        word_indices = (
            events[events["kind"] == "word"].groupby(["sequence_id"]).cumcount()
        )
        events.loc[events["kind"] == "word", "word_index"] = word_indices

        # Create blocks from between sentences user word indexes
        events = events.event.create_blocks(groupby="sentence")

        return events
