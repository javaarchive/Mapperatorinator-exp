from datetime import timedelta
from decimal import Decimal

from .osu_parser import OsuParser
from .mmrs_dataset import BeatmapDatasetIterable, LABEL_IGNORE_ID
from .data_utils import load_audio_file, merge_events, remove_events_of_type, get_song_length, speed_events

from ..tokenizer import Tokenizer
from ..config import DataConfig, TrainConfig
from ..event import ContextType, Event, EventType

from torch.utils.data import IterableDataset

from pathlib import Path

from typing import Optional, Callable

import os
import random
import glob

import simfile

import torch
import numpy as np

from simfile.timing import Beat, TimingData, BeatValue
from simfile.timing.engine import TimingEngine
from simfile.notes import NoteData, NoteType
from simfile.notes.timed import time_notes

from multiprocessing.managers import Namespace
from dataclasses import dataclass

@dataclass
class SimfileSubchart:
    file: simfile.types.Simfile
    chart: simfile.types.Chart
    timing_data: TimingData
    note_data: NoteData
    timing_engine: TimingEngine
    columns: int
    @staticmethod
    def create(sfile: simfile.types.Simfile, chart: simfile.types.Chart):
        timing_data = TimingData(sfile, chart)
        timing_engine = TimingEngine(timing_data)
        note_data = NoteData(chart)
        columns = note_data.columns # aka circle size
        return SimfileSubchart(sfile, chart, timing_data, note_data, timing_engine, columns)

# timing point compatibility class
@dataclass
class SMTimingPoint:
    offset: timedelta
    ms_per_beat: float
    # unusued I think?
    volume: int = 100
    kiai_mode: bool = False
    meter: int = 4

    bpm_override: Optional[float] = None

    @property
    def bpm(self):
        if self.bpm_override is not None:
            return self.bpm_override
        return 60000.0 / self.ms_per_beat

EPSILON_MS = 2
EPSILON = 0.002

class ChartParser(OsuParser):
    def __init__(self, args: TrainConfig, tokenizer: Tokenizer) -> None:
        super().__init__(args, tokenizer)

    @staticmethod
    def uninherited_point_at(time: timedelta, beatmap: SimfileSubchart):
        # compat method
        for bpm_change in reversed(beatmap.timing_data.bpms):
            bpm_change: BeatValue = bpm_change
            time_start = beatmap.timing_engine.time_at(bpm_change.beat)
            time_start_timedelta = timedelta(milliseconds = time_start * 1000.0)
            # compare ms level
            if (time_start * 1000 - EPSILON_MS) <= (time / timedelta(milliseconds=1)):
                ms_per_beat = float((Decimal("60.0") * Decimal("1000.0")) / bpm_change.value)
                return SMTimingPoint(offset = time_start_timedelta, ms_per_beat = ms_per_beat)

    def get_timing_points(beatmap: SimfileSubchart) -> list[SMTimingPoint]:
        timing_data = beatmap.timing_data
        timing_engine = beatmap.timing_engine
        timing_points = []
        for bpm_change in timing_data.bpms:
            bpm_change: BeatValue = bpm_change
            time_start = timing_engine.time_at(bpm_change.beat)
            ms_per_beat = float((Decimal("60.0") * Decimal("1000.0")) / bpm_change.value)
            timing_points.append(SMTimingPoint(offset = timedelta(milliseconds = time_start * 1000.0), ms_per_beat = ms_per_beat, bpm_override = float(bpm_change.value)))
        return timing_points

    def get_duration(self, beatmap_subchart: SimfileSubchart) -> float:
        last_time = max(timed_note.time for timed_note in time_notes(beatmap_subchart.note_data, beatmap_subchart.timing_data))
        sfile = beatmap_subchart.file
        if getattr(sfile, "lastsecondhint", None):
            last_time = max(last_time, float(sfile.lastsecondhint))
        if getattr(sfile, "musiclength", None):
            last_time = max(last_time, float(sfile.musiclength))
        return last_time

    def get_hold_note_ratio(self, beatmap_subchart: SimfileSubchart) -> float:
        total = 0
        hold_note_count = 0
        for timed_note in time_notes(beatmap_subchart.note_data, beatmap_subchart.timing_data):
            note = timed_note.note
            if note.note_type == NoteType.HOLD_HEAD or note.note_type == NoteType.ROLL_HEAD:
                hold_note_count += 1
            total += 1
        
        assert total > 0, "No notes found in chart."
        return hold_note_count / total

    def parse(
            self,
            simfile_subchart: SimfileSubchart,
            speed: float = 1.0,
            song_length: Optional[float] = None,
            flip: tuple[bool, bool] = (False, False),
    ) -> tuple[list[Event], list[int]]:
        sfile = simfile_subchart.file
        chart = simfile_subchart.chart

        note_data = NoteData(chart)
        timing_data = TimingData(sfile, chart)
        timing_engine = TimingEngine(timing_data)
        columns_count = note_data.columns # aka circle size
        
        events = []
        event_times = []

        for timed_note in time_notes(note_data, timing_data):
            note = timed_note.note
            time_ms = timed_note.time * 1000.0
            time_delta = timedelta(milliseconds = time_ms)

            if not note.note_type in [
                NoteType.TAP,
                NoteType.HOLD_HEAD,
                NoteType.ROLL_HEAD,
                NoteType.TAIL,
            ] :
                continue
            # set column
            column = note.column
            # column = (columns_count - 1 - column) if flip[0] else column
            if self.types_first:
                events.append(Event(EventType.MANIA_COLUMN, column))
                event_times.append(time_ms)
            self._add_time_event(
                time_delta,
                simfile_subchart,
                events,
                event_times,
            )
            # we don't need to do complicated calculation because the column is already an integer.
            if not self.types_first:
                events.append(Event(EventType.MANIA_COLUMN, column))
                event_times.append(time_ms)

            if note.note_type == NoteType.TAP:
                events.append(Event(EventType.CIRCLE))
                event_times.append(time_ms)
            elif note.note_type == NoteType.HOLD_HEAD or note.note_type == NoteType.ROLL_HEAD:
                # pretend that roll is the same thing as hold (for now).
                events.append(Event(EventType.HOLD_NOTE))
                event_times.append(time_ms)
            elif note.note_type == NoteType.TAIL:
                events.append(Event(EventType.HOLD_NOTE_END))
                event_times.append(time_ms)

        # Sort events by time
        if len(events) > 0:
            # noinspection PyArgumentList
            events, event_times = zip(*sorted(zip(events, event_times), key=lambda x: x[1]))
        result = list(events), list(event_times)

        if self.add_timing:
            timing_events = self.parse_timing(ChartParser.get_timing_points(simfile_subchart), song_length = self.get_duration(simfile_subchart) * 1000.0)
            result = merge_events(timing_events, result)

        if speed != 1.0:
            result = speed_events(result, speed)

        return result

class ChartDatasetIterable(BeatmapDatasetIterable):
    @staticmethod
    def _load_metadata(track_path: Path) -> dict:
         return {}
    
    def _get_difficulty(self, beatmap_metadata, speed: float = 1.0) -> float:
        return 1.0 # stub
    
    def _get_next_beatmaps(self):
        for beatmap_idx, beatmap_path in enumerate(self.metadata): # data storage hack
            for sample in self._get_next_beatmap(beatmap_path, {"beatmap_idx": beatmap_idx}, 1.0):
                yield sample

    def _get_next_tracks(self):
        # aliased for now.
        return self._get_next_beatmaps()

    def _get_next_beatmap(self, beatmap_path: Path, metadata: dict, speed: float, flip: tuple[bool, bool] = (False, False)):
        
        root_folder = beatmap_path.parent

        chart_simfile = simfile.open(beatmap_path)

        music_path = root_folder / chart_simfile.music

        # fix case sensitivity if not using ..
        if not ".." in chart_simfile.music:
            filename_matches = [filename for filename in os.listdir(root_folder) if filename.lower() == chart_simfile.music.lower()]
            assert len(filename_matches) > 0, f"Music file {chart_simfile.music} not found in {root_folder}"
            actual_music_filename = filename_matches[0]
            music_path = root_folder / actual_music_filename
        
        # context
        context_info = None
        if len(self.args.context_types) > 0:
            # Randomly select a context type with probabilities of context_weights
            context_info = random.choices(self.args.context_types, weights=self.args.context_weights)[0]

            if isinstance(context_info, str):
                context_info = {"out": "map", "in": [context_info]}
            else:
                # It's important to copy the context_info because we will modify it, and we don't want to permanently change the config
                context_info = context_info.copy()

        # load audio and other data

        audio_samples = load_audio_file(music_path, self.args.sample_rate, speed, self.args.normalize_audio)
        
        frames, frame_times = self._get_frames(audio_samples)

        
        # {"extra": {"context_type": context, "add_type": add_type, "id": identifier + '_' + context.value}}

        for chart_idx, chart in enumerate(chart_simfile.charts):
            data = {"extra":{"context_type": ContextType.MAP, "add_type": True, "id": f"{chart_idx}_{beatmap_path.stem}_map"}}#, "context_type": "map"}
            
            chart: simfile.types.Chart = chart

            generated_id = 1000 * metadata["beatmap_idx"] + chart_idx  

            simfile_subchart = SimfileSubchart.create(
                sfile = chart_simfile,
                chart = chart
            )

            print("parsing", chart_simfile.title, " -> ", chart.difficulty, chart.meter, beatmap_path, simfile_subchart.note_data.columns, " cols")

            data["events"], data["event_times"] = self.parser.parse(simfile_subchart, speed, None, flip)

            extra_data = {
                "beatmap_idx": torch.tensor(generated_id, dtype=torch.long),
                "mapper_idx": torch.tensor(1, dtype=torch.long),
                "difficulty": torch.tensor(float(chart.meter), dtype=torch.float32),
                "special": {
                    "hitsounded": False,
                    "song_length": get_song_length(audio_samples, self.args.sample_rate),
                    "keycount": simfile_subchart.note_data.columns,
                    "hold_note_ratio": self.parser.get_hold_note_ratio(simfile_subchart),
                    # can't use this in inference
                    #"circle_size": simfile_subchart.note_data.columns,
                    "year": 2077,
                    "gamemode": 3, # mania id
                    "beatmap_id": torch.tensor(generated_id, dtype=torch.long),
                    "difficulty": torch.tensor(float(chart.meter), dtype=torch.float32),
                    # "beatmap_id": 
                    # TODO: calculate hold note ratio.
                },
            }

            data_none_type = {
                **data,
                "events": [],
                "event_times": [],
                "extra": {
                    **data["extra"],
                    "context_type": ContextType.NONE,
                    "add_type": True,
                    "id": f"{chart_idx}_{beatmap_path.stem}_none",
                },
                # "context_type": ContextType.NONE,
            }

            print("events generated for", data["extra"]["id"], " count ", len(data["events"]))

            sequences = self._create_sequences(
                frames,
                frame_times,
                #out_context,
                #in_context,
                [data_none_type],
                [data],
                extra_data
                #extra_data,
            )

            #for sequence in sequences:
            #    self.maybe_change_dataset()
            #    sequence = self._normalize_time_shifts(sequence)
            #    sequence = self._tokenize_sequence(sequence)
            #    sequence = self._pad_frame_sequence(sequence)
            #    sequence = self._pad_and_split_token_sequence(sequence)
            #    if not self.add_empty_sequences and ((sequence["labels"] == self.tokenizer.eos_id) | (
            #            sequence["labels"] == LABEL_IGNORE_ID)).all():
            #        continue
            #    # if sequence["decoder_input_ids"][self.pre_token_len - 1] != self.tokenizer.pad_id:
            #    #     continue
            #    yield sequence

            yield from self.process_sequences(sequences, beatmap_path)

    

def discover_simfiles(path: Path) -> list[Path]:
    discover_simfile_paths = []
    for dirpath, dirnames, filepaths in os.walk(path):
        if any([
            filename.endswith(".sm") or filename.endswith(".ssc") for filename in filepaths
        ]):
            # add
            simfile_filename = sorted([filename for filename in filepaths if filename.endswith(".sm") or filename.endswith(".ssc")])[-1]
            discover_simfile_paths.append(path / dirpath / simfile_filename) # prefer ssc over sm
    
    return discover_simfile_paths
        

# stepmania sim file parser
class StepmaniaDataset(IterableDataset):

    def __init__(
            self,
            args: DataConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            beatmap_files: Optional[list[Path]] = None,
            test: bool = False,
            shared: Namespace = None,
    ):
        super().__init__()
        self.simfile_paths = beatmap_files
        self.path = args.test_dataset_path if test else args.train_dataset_path
        self.start = args.test_dataset_start if test else args.train_dataset_start
        self.end = args.test_dataset_end if test else args.train_dataset_end
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared

    def _get_simfile_paths(self) -> list[Path]:
        if self.simfile_paths is not None:
            return self.simfile_paths
        
        self.simfile_paths = discover_simfiles(Path(self.path))

        return self.simfile_paths
    
    def __iter__(self):
        untrunc_paths = self._get_simfile_paths()[:]

        if not self.test:
            random.shuffle(untrunc_paths)
        else:
            random.shuffle(untrunc_paths)
        simfile_paths = untrunc_paths[self.start:self.end]

        

        if self.args.cycle_length > 1 and not self.test:
            return InterleavingStepmaniaDatasetIterable(
                simfile_paths,
                self._iterable_factory,
                self.args.cycle_length,
            )

        return self._iterable_factory(simfile_paths).__iter__()

    def _iterable_factory(self, beatmap_files: list[Path]):
        return ChartDatasetIterable(
            beatmap_files, # metadata is wrong type
            self.args,
            Path(self.path),
            self.parser,
            self.tokenizer,
            self.test,
            self.shared,
            None,
        )


class InterleavingStepmaniaDatasetIterable:
    __slots__ = ("workers", "cycle_length", "index")

    def __init__(
            self,
            simfile_paths: list[Path],
            iterable_factory: Callable,
            cycle_length: int,
    ):
        per_worker = int(np.ceil(len(simfile_paths) / float(cycle_length)))
        self.workers = [
            iterable_factory(
                simfile_paths[
                i * per_worker: min(len(simfile_paths), (i + 1) * per_worker)
                ]
            ).__iter__()
            for i in range(cycle_length)
        ]
        self.cycle_length = cycle_length
        self.index = 0

    def __iter__(self) -> "InterleavingStepmaniaDatasetIterable":
        return self

    def __next__(self) -> tuple[any, int]:
        num = len(self.workers)
        for _ in range(num):
            try:
                self.index = self.index % len(self.workers)
                item = self.workers[self.index].__next__()
                self.index += 1
                return item
            except StopIteration:
                self.workers.remove(self.workers[self.index])
        raise StopIteration
    
if __name__ == "__main__":
    print(discover_simfiles(Path(".")))