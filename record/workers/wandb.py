import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from typing import List, Optional
from collections import deque
from record.models.event import InputEvent
from record.models.aggregation import AggregationRequest


class WandBLogger:
    """Logger for tracking events and aggregations in WandB."""

    def __init__(self, project_name: str = "screen-recorder", session_name: Optional[str] = None):
        """
        Initialize WandB logger.

        Args:
            project_name: WandB project name
            session_name: Optional session name, auto-generated if None
        """
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        wandb.init(
            project=project_name,
            name=session_name,
            config={
                "event_types": ["click", "move", "scroll", "key"],
                "visualization": "real-time"
            }
        )

        self.colors = {
            'click': '#FF6B6B',
            'move': '#4ECDC4',
            'scroll': '#FFE66D',
            'key': '#95E1D3'
        }

        self.event_y_positions = {
            'click': 3,
            'move': 2,
            'scroll': 1,
            'key': 0
        }

        self.aggregation_history = deque(maxlen=1000)
        self.event_history = deque(maxlen=5000)
        self.start_time = None

        self.total_events = 0
        self.total_aggregations = 0

    def log_event(self, event: InputEvent, event_type: str):
        """
        Log an individual input event.

        Args:
            event: InputEvent object
            event_type: Type string ('click', 'move', 'scroll', 'key')
        """
        if self.start_time is None:
            self.start_time = event.timestamp

        self.event_history.append({
            'timestamp': event.timestamp,
            'relative_time': event.timestamp - self.start_time,
            'event_type': event_type
        })

        self.total_events += 1

        wandb.log({
            f'events/{event_type}': 1,
            'events/total': self.total_events,
            'timestamp': event.timestamp
        })

    def log_aggregations(self, requests: List[AggregationRequest]):
        """
        Log aggregation requests and create visualization.

        Args:
            requests: List of AggregationRequest objects
        """
        if not requests:
            return

        if self.start_time is None:
            self.start_time = requests[0].timestamp

        # Group requests into start/end pairs for each burst
        bursts = self._group_into_bursts(requests)

        # Store bursts for visualization
        for burst in bursts:
            self.aggregation_history.append(burst)
            self.total_aggregations += 1

        # Create and log visualizations
        self._log_aggregation_plot()
        self._log_individual_events_plot()

        # Log statistics
        wandb.log({
            'aggregations/total': self.total_aggregations,
            'aggregations/in_batch': len(bursts)
        })

    def _group_into_bursts(self, requests: List[AggregationRequest]) -> List[dict]:
        """
        Group start/end requests into burst objects.

        Args:
            requests: List of AggregationRequest objects

        Returns:
            List of burst dictionaries with start_time, end_time, and event_type
        """
        bursts = []
        pending_starts = {}

        for req in sorted(requests, key=lambda r: r.timestamp):
            if req.is_start:
                # Store the start request
                key = f"{req.event_type}_{req.timestamp}"
                pending_starts[key] = req
            else:
                # Find matching start request
                # Look for starts of the same type that haven't been matched yet
                matching_start = None
                for key, start_req in list(pending_starts.items()):
                    if start_req.event_type == req.event_type:
                        matching_start = start_req
                        del pending_starts[key]
                        break

                if matching_start:
                    bursts.append({
                        'event_type': req.event_type,
                        'start_time': matching_start.timestamp,
                        'end_time': req.timestamp,
                        'start_relative': matching_start.timestamp - self.start_time,
                        'end_relative': req.timestamp - self.start_time
                    })

        return bursts

    def _log_aggregation_plot(self):
        """Create and log the aggregation timeline plot."""
        if not self.aggregation_history:
            return

        fig, ax = plt.subplots(figsize=(14, 4))

        # All bursts at y=0, different colors
        y_pos = 0

        for burst in self.aggregation_history:
            start = burst['start_relative']
            duration = burst['end_relative'] - burst['start_relative']
            color = self.colors.get(burst['event_type'], '#999999')

            # Create rectangle for the burst
            rect = patches.Rectangle(
                (start, y_pos - 0.3),
                duration,
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)

        # Formatting
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, max(b['end_relative'] for b in self.aggregation_history) + 1)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Aggregated Bursts', fontsize=12)
        ax.set_title('Event Burst Timeline', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')

        # Legend
        legend_elements = [
            patches.Patch(facecolor=self.colors[et], edgecolor='black', label=et.capitalize())
            for et in ['click', 'move', 'scroll', 'key']
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()

        # Log to WandB
        wandb.log({"aggregation_timeline": wandb.Image(fig)})
        plt.close(fig)

    def _log_individual_events_plot(self):
        """Create and log the individual events timeline plot."""
        if not self.event_history:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot individual events as scatter points
        for event_type, y_pos in self.event_y_positions.items():
            events_of_type = [
                e for e in self.event_history
                if e['event_type'] == event_type
            ]

            if events_of_type:
                times = [e['relative_time'] for e in events_of_type]
                y_values = [y_pos] * len(times)
                color = self.colors.get(event_type, '#999999')

                ax.scatter(
                    times,
                    y_values,
                    c=color,
                    s=30,
                    alpha=0.6,
                    label=event_type.capitalize(),
                    marker='o'
                )

        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Event Type', fontsize=12)
        ax.set_title('Individual Events Timeline', fontsize=14, fontweight='bold')
        ax.set_yticks(list(self.event_y_positions.values()))
        ax.set_yticklabels([et.capitalize() for et in ['key', 'scroll', 'move', 'click']])
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper right', fontsize=10)

        if self.event_history:
            max_time = max(e['relative_time'] for e in self.event_history)
            ax.set_xlim(0, max_time + 1)

        plt.tight_layout()

        # Log to WandB
        wandb.log({"individual_events_timeline": wandb.Image(fig)})
        plt.close(fig)

    def log_statistics(self, stats: dict):
        """
        Log general statistics.

        Args:
            stats: Dictionary of statistics to log
        """
        wandb.log(stats)

    def finish(self):
        """Finish the WandB run."""
        # Log final visualizations
        if self.aggregation_history:
            self._log_aggregation_plot()
        if self.event_history:
            self._log_individual_events_plot()

        # Log final summary
        wandb.log({
            'summary/total_events': self.total_events,
            'summary/total_aggregations': self.total_aggregations
        })

        wandb.finish()
