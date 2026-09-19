# SPDX-License-Identifier: AGPL-3.0-only

"""Keep externally generated progress output concise and diagnostic output intact.

The tracking stages invoke third-party tools through ``batch.py``.  Several of
those tools redraw a tqdm progress bar many times per second with carriage
returns.  A redirected stdout stream turns every redraw into a permanent log
record, which makes the GUI, Colab output, and ``log.txt`` unnecessarily large.

``CompactChildOutput`` consumes that stream and emits only completed progress
states.  Training keeps one completed line per epoch; other stages keep one
completed line per progress bar.  Ordinary text, warnings, errors, and complete
tracebacks are never summarized.
"""

from __future__ import annotations

import re


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TQDM_PERCENT_RE = re.compile(r"(?P<pct>\d{1,3})%\|")
_EPOCH_PREFIX_RE = re.compile(r"^\s*(?P<epoch>\d+)\s*/\s*(?P<epochs>\d+)\b")
_TQDM_COUNT_RE = re.compile(r"(?P<count>\d+)\s*/\s*(?P<total>\d+)(?=\s|\[|$)")
_IMPORTANT_RE = re.compile(
    r"\b(?:error|exception|traceback|failed|fatal|critical|oom|out of memory|"
    r"segmentation fault|assertion failed|aborted)\b",
    re.IGNORECASE,
)
_DIAGNOSTIC_PREFIX_RE = re.compile(
    r"^\s*\[(?:WARN(?:ING)?|ERROR|FATAL|CRITICAL|RECOVERY)\]", re.IGNORECASE
)
_TRAINING_TABLE_RE = re.compile(
    r"^\s*(?:Epoch\s+GPU_mem\b|Class\s+Images\s+Instances\b|all\s+\d+\s+\d+\s+)",
    re.IGNORECASE,
)


class CompactChildOutput:
    """Reduce dynamic child-process progress output to stable log records.

    ``feed`` accepts arbitrary decoded chunks, including chunks split in the
    middle of a UTF-8 stream or a CRLF delimiter.  It returns text that is safe
    to write verbatim to both an interactive console and a persistent log.
    """

    def __init__(self, *, training: bool = False) -> None:
        self.training = bool(training)
        self._record: list[str] = []
        self._last_delimiter_was_cr = False
        self._emitted_progress: set[tuple[object, ...]] = set()
        self._omitted_progress_updates = 0
        self._omitted_perf_samples = 0
        self._omitted_training_rows = 0

    @staticmethod
    def _clean_record(record: str) -> str:
        return _ANSI_ESCAPE_RE.sub("", record).rstrip()

    def _parse_progress(self, line: str) -> tuple[tuple[object, ...] | None, int, str | None] | None:
        percent_match = _TQDM_PERCENT_RE.search(line)
        if percent_match is None:
            return None
        try:
            pct = max(0, min(100, int(percent_match.group("pct"))))
        except (TypeError, ValueError):
            return None

        epoch_match = _EPOCH_PREFIX_RE.match(line)
        if epoch_match is not None:
            epoch = int(epoch_match.group("epoch"))
            epochs = int(epoch_match.group("epochs"))
            return (
                ("epoch", epoch, epochs),
                pct,
                f"[PROGRESS] training epoch {epoch}/{epochs} complete",
            )

        # During training, Ultralytics also shows validation and data-loader
        # bars.  The completed epoch line above is the useful concise status;
        # keeping the nested bars would make the training log longer again.
        if self.training:
            return (None, pct, None)

        # Non-training stages retain one concise completion line for a regular
        # tqdm bar.  The final N/N count follows the percentage portion.
        counts = list(_TQDM_COUNT_RE.finditer(line[percent_match.end() :]))
        if not counts:
            return (None, pct, None)
        count_match = counts[-1]
        count = int(count_match.group("count"))
        total = int(count_match.group("total"))
        description = " ".join(line[: percent_match.start()].strip(" :").split())
        if not description:
            description = "progress"
        if len(description) > 72:
            description = description[:69].rstrip() + "..."
        return (
            ("bar", description, total),
            pct,
            f"[PROGRESS] {description}: {count}/{total} complete",
        )

    def _consume_record(self, record: str) -> str:
        line = self._clean_record(record)
        if not line.strip():
            return ""

        # A traceback or an error must win over every progress heuristic, even
        # when a library happened to write it while a progress bar was active.
        if _IMPORTANT_RE.search(line) or _DIAGNOSTIC_PREFIX_RE.match(line):
            return line + "\n"

        if line.lstrip().startswith("[PERF]"):
            self._omitted_perf_samples += 1
            return ""

        if self.training and _TRAINING_TABLE_RE.match(line):
            self._omitted_training_rows += 1
            return ""

        progress = self._parse_progress(line)
        if progress is None:
            return line + "\n"

        key, pct, summary = progress
        if pct < 100 or key is None or summary is None or key in self._emitted_progress:
            self._omitted_progress_updates += 1
            return ""

        self._emitted_progress.add(key)
        return summary + "\n"

    def feed(self, text: str) -> str:
        """Consume a decoded stream chunk and return compacted complete records."""
        if not text:
            return ""

        output: list[str] = []
        for char in text:
            if char in ("\r", "\n"):
                # Treat CRLF as one delimiter even when its two characters are
                # received in separate pipe reads.
                if char == "\n" and self._last_delimiter_was_cr and not self._record:
                    self._last_delimiter_was_cr = False
                    continue
                output.append(self._consume_record("".join(self._record)))
                self._record.clear()
                self._last_delimiter_was_cr = char == "\r"
            else:
                self._record.append(char)
                self._last_delimiter_was_cr = False
        return "".join(output)

    def flush(self) -> str:
        """Return a final non-progress record left without a newline."""
        if not self._record:
            return ""
        record = "".join(self._record)
        self._record.clear()
        self._last_delimiter_was_cr = False
        return self._consume_record(record)

    def summary_line(self) -> str:
        """Describe suppressed routine output once, without hiding diagnostics."""
        parts: list[str] = []
        if self._omitted_progress_updates:
            parts.append(f"{self._omitted_progress_updates} intermediate progress update(s)")
        if self._omitted_training_rows:
            parts.append(f"{self._omitted_training_rows} repeated training table row(s)")
        if self._omitted_perf_samples:
            parts.append(f"{self._omitted_perf_samples} routine [PERF] sample(s)")
        if not parts:
            return ""
        return "[INFO] compacted progress output: omitted " + ", ".join(parts) + ".\n"


class GuiProgressPassthrough:
    """Forward raw progress bars to a GUI while dropping routine ``[PERF]`` lines.

    The Easy Tracking, Advanced Tracking, and multi-config GUIs use raw tqdm
    redraws to update their own progress widgets and live loss charts.  Their
    terminal-like widgets already render ``\r`` as an overwrite, so this small
    adapter preserves that private GUI protocol while the persistent log uses
    :class:`CompactChildOutput`.
    """

    def __init__(self) -> None:
        self._record: list[str] = []
        self._last_delimiter_was_cr = False
        self._last_cr_was_emitted = False

    @staticmethod
    def _should_emit(record: str) -> bool:
        cleaned = _ANSI_ESCAPE_RE.sub("", record)
        if not cleaned.lstrip().startswith("[PERF]"):
            return True
        return bool(_IMPORTANT_RE.search(cleaned) or _DIAGNOSTIC_PREFIX_RE.match(cleaned))

    def feed(self, text: str) -> str:
        """Return raw output suitable for GUI progress parsing."""
        if not text:
            return ""

        output: list[str] = []
        for char in text:
            if char == "\n" and self._last_delimiter_was_cr and not self._record:
                if self._last_cr_was_emitted:
                    output.append("\n")
                self._last_delimiter_was_cr = False
                self._last_cr_was_emitted = False
                continue

            if char in ("\r", "\n"):
                record = "".join(self._record)
                emitted = self._should_emit(record)
                if emitted:
                    output.append(record + char)
                self._record.clear()
                self._last_delimiter_was_cr = char == "\r"
                self._last_cr_was_emitted = emitted and char == "\r"
            else:
                self._record.append(char)
                self._last_delimiter_was_cr = False
                self._last_cr_was_emitted = False
        return "".join(output)

    def flush(self) -> str:
        """Return a final raw record that did not end in a delimiter."""
        if not self._record:
            return ""
        record = "".join(self._record)
        self._record.clear()
        self._last_delimiter_was_cr = False
        self._last_cr_was_emitted = False
        return record if self._should_emit(record) else ""
