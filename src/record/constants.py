from dataclasses import dataclass
import os
from typing import Dict


@dataclass(frozen=True)
class ConstantsSpec:
    CLICK_GAP_THRESHOLD: float
    MOVE_GAP_THRESHOLD: float
    SCROLL_GAP_THRESHOLD: float
    KEY_GAP_THRESHOLD: float
    CLICK_TOTAL_THRESHOLD: float
    MOVE_TOTAL_THRESHOLD: float
    SCROLL_TOTAL_THRESHOLD: float
    KEY_TOTAL_THRESHOLD: float
    PADDING_BEFORE: int
    PADDING_AFTER: int
    FINAL_PADDING: int

    @property
    def max_total_threshold(self) -> float:
        """Calculate the maximum total threshold across all event types."""
        return max(
            self.CLICK_TOTAL_THRESHOLD,
            self.MOVE_TOTAL_THRESHOLD,
            self.SCROLL_TOTAL_THRESHOLD,
            self.KEY_TOTAL_THRESHOLD,
        )


PRESETS: Dict[str, ConstantsSpec] = {
    "accurate": ConstantsSpec(
        CLICK_GAP_THRESHOLD=0.2,
        MOVE_GAP_THRESHOLD=0.5,
        SCROLL_GAP_THRESHOLD=0.5,
        KEY_GAP_THRESHOLD=0.5,
        CLICK_TOTAL_THRESHOLD=0.3,
        MOVE_TOTAL_THRESHOLD=4.0,
        SCROLL_TOTAL_THRESHOLD=3.0,
        KEY_TOTAL_THRESHOLD=6.0,
        PADDING_BEFORE=75,
        PADDING_AFTER=75,
        FINAL_PADDING=100,
    ),
    "rough": ConstantsSpec(
        CLICK_GAP_THRESHOLD=0.8,
        MOVE_GAP_THRESHOLD=2.0,
        SCROLL_GAP_THRESHOLD=2.0,
        KEY_GAP_THRESHOLD=2.0,
        CLICK_TOTAL_THRESHOLD=1.2,
        MOVE_TOTAL_THRESHOLD=16.0,
        SCROLL_TOTAL_THRESHOLD=12.0,
        KEY_TOTAL_THRESHOLD=24.0,
        PADDING_BEFORE=75,
        PADDING_AFTER=75,
        FINAL_PADDING=100,
    ),
}


class ConstantsManager:
    """Singleton manager for constants that can be updated at runtime."""

    _instance = None
    _current_preset: str = "accurate"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            preset = os.getenv("CAPTURE_PRECISION", "accurate").lower()
            self.set_preset(preset, verbose=False)

    def set_preset(self, preset_name: str, verbose: bool = True) -> None:
        """Set the current preset by name."""
        preset_name = preset_name.lower()

        if preset_name not in PRESETS:
            available = ", ".join(PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )

        self._current_preset = preset_name

        if verbose:
            print(f"Setting constants to '{preset_name}' precision.")
            print(f"Constants: {self.get()}")

    def get(self) -> ConstantsSpec:
        """Get the current constants specification."""
        return PRESETS[self._current_preset]

    def get_preset_name(self) -> str:
        """Get the name of the current preset."""
        return self._current_preset

    @property
    def max_total_threshold(self) -> float:
        """Get the maximum total threshold for the current preset."""
        return self.get().max_total_threshold


constants_manager = ConstantsManager()


def get_constants() -> ConstantsSpec:
    """Get the current constants specification."""
    return constants_manager.get()
