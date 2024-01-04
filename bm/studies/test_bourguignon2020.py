# Copyright (c) HiTZ Zentroa, BCBL and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for the Bourguignon2020 study classes.

Most of them are integration tests, so the original dataset is required. This
can be used to check if the dataset has been installed correctly in the
`data/bourguignon2020` directory. It also checks if the source code is still
working.

Check the `bourguignon2020.py` file for more installation details.
"""

from itertools import product

import mne
import pandas as pd
from bm.lib.textgrid import Entry

from . import bourguignon2020


def test_metadata_parse_txt() -> None:
    """
    Test the parsing of transcription text into a DataFrame.

    This test checks if the `_BourguignonMetadata.parse_txt` method correctly
    parses transcriptions for a given subject and task into a pandas DataFrame.
    """
    meta = bourguignon2020._BourguignonMetadata()
    meta.lazy_init()
    text = meta.parse_txt("selfp1", "self")
    assert isinstance(text, pd.DataFrame)


def test_metadata_parse_textgrid() -> None:
    """
    Test the parsing of TextGrid data into a DataFrame.

    This test verifies that the `_BourguignonMetadata.parse_textgrid` method
    successfully parses TextGrid files for a specific subject and task into a
    DataFrame.
    """
    meta = bourguignon2020._BourguignonMetadata()
    meta.lazy_init()
    txtgrid = meta.parse_textgrid("selfp1", "self")
    assert isinstance(txtgrid, pd.DataFrame)


def test_metadata_process() -> None:
    """
    Test the processing of metadata for a specific subject and task.

    This test ensures that the `_BourguignonMetadata.process` method correctly
    processes and merges transcription and TextGrid data into a unified
    DataFrame.
    """
    meta = bourguignon2020._BourguignonMetadata()
    meta.lazy_init()
    data = meta.process("selfp1", "self")
    assert isinstance(data, pd.DataFrame)


def test_constructor() -> None:
    """
    Test the construction of a Bourguignon2020Recording object.

    This test verifies that an instance of Bourguignon2020Recording can be
    successfully created with a given subject and task.
    """
    rec = bourguignon2020.Bourguignon2020Recording("selfp1", "self")
    assert isinstance(rec, bourguignon2020.Bourguignon2020Recording)


def test_download() -> None:
    """
    Test the download functionality of Bourguignon2020Recording.

    This test ensures that the `download` method of Bourguignon2020Recording
    works as expected. It is a placeholder test as the dataset is non-public
    and already downloaded.
    """
    rec = bourguignon2020.Bourguignon2020Recording("selfp1", "self")
    rec.download()


def test_iter() -> None:
    """
    Test the iterator functionality of Bourguignon2020Recording.

    This test checks if `Bourguignon2020Recording.iter` correctly iterates over
    all recordings in the dataset and yields instances of
    Bourguignon2020Recording.
    """
    recs = []
    for rec in bourguignon2020.Bourguignon2020Recording.iter():
        recs.append(rec)
    assert len(recs) > 0
    assert isinstance(recs[0], bourguignon2020.Bourguignon2020Recording)


def test_load_raw() -> None:
    """
    Test the loading of raw MEG data in Bourguignon2020Recording.

    This test verifies that the `_load_raw` method correctly loads and returns
    the raw MEG data as an MNE RawArray object.
    """
    rec = bourguignon2020.Bourguignon2020Recording("selfp1", "self")
    raw = rec._load_raw()
    assert isinstance(raw, mne.io.Raw)


def test_load_event() -> None:
    """
    Test the loading of event data in Bourguignon2020Recording.

    This test checks if the `_load_events` method successfully loads and
    processes events metadata into a DataFrame with expected columns and
    values.
    """
    rec = bourguignon2020.Bourguignon2020Recording("selfp1", "self")
    events = rec._load_events()
    assert isinstance(events, pd.DataFrame)
    columns = [
        "start",
        "duration",
        "kind",
        "modality",
        "language",
        "word",
        "word_id",
        "sequence_id",
        "condition",
        "filepath",
    ]
    for col in columns:
        assert col in events, f"Column `{col}` missing in events."


def test_load_event_subject12() -> None:
    """
    Test the loading of event data for a specific subject in
    Bourguignon2020Recording.

    This test is similar to `test_load_event` but focuses on a different
    subject to ensure consistency across different subjects in the dataset.
    """
    rec = bourguignon2020.Bourguignon2020Recording("selfp12", "self")
    events = rec._load_events()
    assert isinstance(events, pd.DataFrame)
    columns = [
        "start",
        "duration",
        "kind",
        "modality",
        "language",
        "word",
        "word_id",
        "sequence_id",
        "condition",
        "filepath",
    ]
    for col in columns:
        assert col in events, f"Column `{col}` missing in events."


def test_validate_event() -> None:
    """
    Validates the event data of the first recording in the Bourguignon2020
    dataset.

    This test ensures that event data loaded by `_load_events` is valid and
    that blocks of events can be successfully created based on sentences.

    It can also generate event plots (currently commented at the end).

    Based in the documentation here:
    https://github.com/facebookresearch/brainmagick/blob/main/doc/recordings_and_events.md#events
    """
    rec = bourguignon2020.Bourguignon2020Recording("selfp1", "self")
    events = rec._load_events()
    # Make sure events are valid
    events = events.event.validate()
    # Create blocks from existing events
    events = events.event.create_blocks(groupby="sentence")
    # Plot events for sanity check (it takes time)
    # fig, ax = events.event.plot()
    # fig.savefig("{rec.subject}_{rec.task}.png", format="png")


def test_number_of_recordings() -> None:
    """
    Test that the number of recordings found is correct.
    """
    n_subjects = 18
    n_tasks = 3

    recs = list(bourguignon2020.Bourguignon2020Recording.iter())
    assert len(recs) == n_subjects * n_tasks


def test_number_of_recordings_by_modality() -> None:
    """
    Test that the number of recordings found is correct.
    """
    modality = "self"
    n_subjects = 18

    recs = list(bourguignon2020.Bourguignon2020Recording.iter(modality=modality))
    assert len(recs) == n_subjects


def test_validate_all_paths() -> None:
    """
    Validates all the paths in the Bourguignon2020 dataset.

    This test iterates over all subjects and tasks to ensure that the paths
    for recordings are valid and accessible.
    """
    paths = bourguignon2020.StudyPaths()  # pylint: disable=unreachable
    subjects = bourguignon2020.get_subjects(paths.megs)
    tasks = bourguignon2020.TASKS
    for subject, task in product(subjects, tasks):
        if subject == "selfp17":  # no MEG recordings present for subject 17
            continue
        paths = bourguignon2020.RecordingPaths(subject, task)
        assert paths.validate() is True, f"Cannot validate: {subject}, {task}"


def test_validate_all_events() -> None:
    """
    Validates all the events data in the Bourguignon2020 dataset.

    This test iterates over all recordings using the Bourguignon2020Recording
    iterator and checks if the events for each recording can be successfully
    validated and processed into blocks based on sentences.

    This test is a little slow and may take a few minutes to complete.
    """
    for rec in bourguignon2020.Bourguignon2020Recording.iter():
        events = rec._load_events()
        events = events.event.validate()
        events.event.create_blocks(groupby="sentence")


def test_realign_textgrid_positive_dec() -> None:
    """
    Tests the WAV/MEG realignment algorithm in the `realign_textgrid` method.
    """
    subject = "selfp1"
    task = "listen"
    parts = {
        "words": [
            Entry(
                start=0.0,
                stop=0.5,
                name="foo",
                tier="word",
            ),
            Entry(
                start=0.5,
                stop=1.0,
                name="bar",
                tier="word",
            ),
        ],
    }
    meta = bourguignon2020._BourguignonMetadata()
    realign = meta.parse_realign(subject, task)
    parts = meta.realign_textgrid(parts, realign, subject, task)
    assert parts["words"][0].start == 6.528


def test_realign_textgrid_negative_dec() -> None:
    """
    Tests the WAV/MEG realignment algorithm with negative delays.

    Here we are going to ensure that the first event is correct when the
    `dec` (MEG signal delay) is negative.
    """
    subject = "selfp1"
    task = "self"
    parts = {
        "words": [
            Entry(
                start=0.0,
                stop=0.5,
                name="foo",
                tier="word",
            ),
            Entry(
                start=0.5,
                stop=1.0,
                name="bar",
                tier="word",
            ),
        ],
    }
    meta = bourguignon2020._BourguignonMetadata()
    realign = meta.parse_realign(subject, task)
    parts = meta.realign_textgrid(parts, realign, subject, task)
    assert parts["words"][0].start == -1.534
