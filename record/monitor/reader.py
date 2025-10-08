import os
from pathlib import Path
from typing import List, Optional


class TailReader:
    """
    Simple tail that keeps a file handle and last position, recovers when file truncated.
    - If the file doesn't exist yet, read_new_lines() returns [] and the reader will try again
      next time (no exception).
    - The initial `from_start` is honoured when the file is first opened. Subsequent automatic
      reopens (due to truncation/rotation) open at the file start.
    """

    def __init__(self, path: str | Path, from_start: bool = True):
        self.path = Path(path)
        self.from_start = bool(from_start)
        self.f: Optional[object] = None
        self.pos: int = 0

        if self.path.exists():
            try:
                self._open(self.from_start)
            except Exception:
                self.f = None
                self.pos = 0

    def _open(self, from_start: bool) -> None:
        """Open the file and set the initial position."""
        self.f = self.path.open("r", encoding="utf-8", errors="ignore")
        if not from_start:
            self.f.seek(0, os.SEEK_END)
        self.pos = self.f.tell()

    def read_new_lines(self) -> List[str]:
        """Non-blocking: return newly appended lines, or [] if none / file missing."""
        if self.f is None:
            if not self.path.exists():
                return []
            try:
                self._open(self.from_start)
            except FileNotFoundError:
                self.f = None
                self.pos = 0
                return []
            except Exception:
                self.f = None
                self.pos = 0
                return []

        try:
            self.f.seek(self.pos)
        except Exception:
            try:
                self.f.close()
            except Exception:
                pass
            try:
                self._open(False)
            except FileNotFoundError:
                self.f = None
                self.pos = 0
                return []
            except Exception:
                self.f = None
                self.pos = 0
                return []

        lines: List[str] = []
        while True:
            line = self.f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n\r"))

        try:
            self.pos = self.f.tell()
        except Exception:
            self.pos = 0

        try:
            st_size = self.path.stat().st_size
            if st_size < self.pos:
                try:
                    self.f.close()
                except Exception:
                    pass
                try:
                    self._open(True)
                except Exception:
                    self.f = None
                    self.pos = 0
        except FileNotFoundError:
            try:
                self.f.close()
            except Exception:
                pass
            self.f = None
            self.pos = 0

        return lines
