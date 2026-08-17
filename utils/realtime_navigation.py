from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


EARTH_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class GnssSample:
    timestamp: float
    latitude_rad: float
    longitude_rad: float
    heading_rad: float | None = None


@dataclass(frozen=True)
class NavigationSnapshot:
    route_active: bool
    moving: bool
    speed_mps: float
    time_to_60m_s: float
    progress_m: float
    route_length_m: float


def geodetic_distance_m(first: GnssSample, second: GnssSample) -> float:
    d_lat = second.latitude_rad - first.latitude_rad
    d_lon = second.longitude_rad - first.longitude_rad
    hav = (
        math.sin(d_lat * 0.5) ** 2
        + math.cos(first.latitude_rad)
        * math.cos(second.latitude_rad)
        * math.sin(d_lon * 0.5) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, hav)))


def estimate_window_speed(samples: Sequence[GnssSample]) -> float:
    """Estimate ground speed from cumulative distance over the whole window."""
    if len(samples) < 2:
        return 0.0
    elapsed_s = float(samples[-1].timestamp - samples[0].timestamp)
    if elapsed_s <= 1e-6:
        return 0.0
    distance_m = sum(
        geodetic_distance_m(first, second)
        for first, second in zip(samples[:-1], samples[1:])
        if second.timestamp > first.timestamp
    )
    speed_mps = distance_m / elapsed_s
    return speed_mps if math.isfinite(speed_mps) else 0.0


def _mission_points(mission: Mapping) -> list[Mapping]:
    if not isinstance(mission, Mapping):
        raise ValueError("Mission must be a JSON object")
    if not all(key in mission for key in ("count", "tolerance", "points")):
        raise ValueError("Mission must contain count, tolerance and points")
    points = mission["points"]
    if not isinstance(points, list) or not points:
        raise ValueError("Mission points must be a non-empty array")
    try:
        repeat_value = mission["count"]
        repeat = int(repeat_value)
        tolerance = float(mission["tolerance"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Mission count/tolerance is invalid") from exc
    if isinstance(repeat_value, bool) or float(repeat_value) != repeat:
        raise ValueError("Mission count must be an integer")
    if repeat < 0 or not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Mission count/tolerance cannot be negative")
    expanded_count = len(points) * (2 if repeat == 0 else repeat)
    if expanded_count > 10000:
        raise ValueError("Expanded mission exceeds 10000 points")
    if repeat == 0:
        return [*points, *reversed(points)]
    return points * repeat


@dataclass(frozen=True)
class EnuMissionRoute:
    east_m: np.ndarray
    north_m: np.ndarray
    arc_m: np.ndarray
    mission_east_m: np.ndarray
    mission_north_m: np.ndarray

    @classmethod
    def from_mission(
        cls,
        mission: Mapping,
        origin_latitude_rad: float,
        origin_longitude_rad: float,
    ) -> "EnuMissionRoute":
        expanded = _mission_points(mission)
        latitude = []
        longitude = []
        for index, point in enumerate(expanded):
            if not isinstance(point, Mapping):
                raise ValueError(f"Mission point {index} must be an object")
            try:
                lat = float(point["latitude"])
                lon = float(point["longitude"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Mission point {index} needs finite latitude/longitude"
                ) from exc
            if (
                not math.isfinite(lat)
                or not math.isfinite(lon)
                or abs(lat) > math.pi * 0.5
                or abs(lon) > math.pi
            ):
                raise ValueError(f"Mission point {index} has invalid radian coordinates")
            latitude.append(lat)
            longitude.append(lon)

        latitude_np = np.asarray(latitude, dtype=np.float64)
        longitude_np = np.asarray(longitude, dtype=np.float64)
        mission_east = (
            EARTH_RADIUS_M
            * (longitude_np - float(origin_longitude_rad))
            * math.cos(float(origin_latitude_rad))
        )
        mission_north = EARTH_RADIUS_M * (
            latitude_np - float(origin_latitude_rad)
        )

        # The route starts at the vessel position at command time. If the first
        # mission point is already there, duplicate filtering removes it.
        east = np.concatenate(([0.0], mission_east))
        north = np.concatenate(([0.0], mission_north))
        segment = np.hypot(np.diff(east), np.diff(north))
        keep = np.concatenate(([True], segment > 0.002))
        east = east[keep]
        north = north[keep]
        if len(east) < 2:
            raise ValueError("Mission route has no measurable length")
        arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(east), np.diff(north)))))
        if not math.isfinite(float(arc[-1])) or arc[-1] <= 0.01:
            raise ValueError("Mission route has no measurable length")

        arrays = (east, north, arc, mission_east, mission_north)
        for array in arrays:
            array.setflags(write=False)
        return cls(*arrays)

    @property
    def length_m(self) -> float:
        return float(self.arc_m[-1])

    def _position(self, arc_m: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(arc_m, dtype=np.float64)
        clipped = np.clip(query, 0.0, self.length_m)
        east = np.interp(clipped, self.arc_m, self.east_m)
        north = np.interp(clipped, self.arc_m, self.north_m)
        beyond = np.maximum(0.0, query - self.length_m)
        if np.any(beyond > 0.0):
            final_heading = self.heading_at(self.length_m)
            east = east + beyond * math.sin(final_heading)
            north = north + beyond * math.cos(final_heading)
        return east, north

    def heading_at(self, progress_m: float) -> float:
        center = float(np.clip(progress_m, 0.0, self.length_m))
        span = min(1.0, max(0.05, self.length_m * 0.01))
        first = max(0.0, center - span)
        second = min(self.length_m, center + span)
        if second - first < 1e-6:
            first = max(0.0, self.length_m - span)
            second = self.length_m
        east, north = self._position(np.asarray([first, second]))
        return math.atan2(float(east[1] - east[0]), float(north[1] - north[0]))

    def body_window(
        self,
        progress_m: float,
        lookahead_m: float,
        sample_spacing_m: float,
        heading_offset_deg: float = 0.0,
    ) -> np.ndarray:
        progress = float(np.clip(progress_m, 0.0, self.length_m))
        count = max(2, int(round(float(lookahead_m) / sample_spacing_m)) + 1)
        local_arc = np.linspace(0.0, float(lookahead_m), count)
        east, north = self._position(progress + local_arc)
        anchor_east, anchor_north = float(east[0]), float(north[0])
        yaw = self.heading_at(progress) + math.radians(float(heading_offset_deg))
        delta_east = east - anchor_east
        delta_north = north - anchor_north
        body = np.column_stack(
            (
                delta_east * math.sin(yaw) + delta_north * math.cos(yaw),
                delta_east * math.cos(yaw) - delta_north * math.sin(yaw),
            )
        )
        body[0] = [0.0, 0.0]
        return body.astype(np.float32)


class RealtimeNavigationState:
    """Thread-safe GNSS buffer and time-driven mission navigation state."""

    def __init__(
        self,
        *,
        lookahead_m: float = 30.0,
        sample_spacing_m: float = 0.25,
        speed_window_points: int = 15,
        speed_update_hz: float = 10.0,
        moving_threshold_mps: float = 1.0,
        timestamp_tolerance_s: float = 1.0,
        heading_offset_deg: float = 0.0,
        gnss_buffer_points: int = 1024,
    ):
        if lookahead_m <= 0.0 or sample_spacing_m <= 0.0:
            raise ValueError("Navigation distances must be greater than zero")
        if speed_window_points < 2 or speed_update_hz <= 0.0:
            raise ValueError("GNSS speed window and update rate are invalid")
        if moving_threshold_mps < 0.0 or timestamp_tolerance_s <= 0.0:
            raise ValueError("Navigation thresholds are invalid")

        self.lookahead_m = float(lookahead_m)
        self.sample_spacing_m = float(sample_spacing_m)
        self.speed_window_points = int(speed_window_points)
        self.speed_update_interval_s = 1.0 / float(speed_update_hz)
        self.moving_threshold_mps = float(moving_threshold_mps)
        self.timestamp_tolerance_s = float(timestamp_tolerance_s)
        self.heading_offset_deg = float(heading_offset_deg)

        self._lock = threading.RLock()
        self._samples: deque[GnssSample] = deque(maxlen=max(gnss_buffer_points, speed_window_points))
        self._route: EnuMissionRoute | None = None
        self._trigger_timestamp = 0.0
        self._progress_m = 0.0
        self._speed_mps = 0.0
        self._moving = False
        self._last_speed_update_monotonic: float | None = None
        self._last_progress_monotonic: float | None = None
        self._stop_event = threading.Event()
        self._update_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._update_thread is not None and self._update_thread.is_alive():
                return
            self._stop_event.clear()
            self._update_thread = threading.Thread(
                target=self._update_loop,
                name="Navigation10Hz",
                daemon=True,
            )
            self._update_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._update_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._update_thread = None

    def _update_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            self.update()
            remaining = self.speed_update_interval_s - (time.monotonic() - started)
            self._stop_event.wait(max(0.0, remaining))

    def add_gnss(
        self,
        timestamp: float,
        latitude_rad: float,
        longitude_rad: float,
        heading_rad: float | None = None,
        *,
        received_at: float | None = None,
    ) -> bool:
        received = time.time() if received_at is None else float(received_at)
        try:
            sample = GnssSample(
                float(timestamp),
                float(latitude_rad),
                float(longitude_rad),
                None if heading_rad is None else float(heading_rad),
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not math.isfinite(sample.timestamp)
            or not math.isfinite(sample.latitude_rad)
            or not math.isfinite(sample.longitude_rad)
            or sample.timestamp <= 0.0
            or abs(sample.latitude_rad) > math.pi * 0.5
            or abs(sample.longitude_rad) > math.pi
            or sample.timestamp > received + self.timestamp_tolerance_s
        ):
            return False
        with self._lock:
            if self._samples and sample.timestamp <= self._samples[-1].timestamp:
                return False
            self._samples.append(sample)
        return True

    def activate_mission(
        self,
        mission: Mapping,
        trigger_timestamp: float,
        *,
        now_epoch: float | None = None,
        now_monotonic: float | None = None,
    ) -> EnuMissionRoute:
        trigger = float(trigger_timestamp)
        epoch = time.time() if now_epoch is None else float(now_epoch)
        monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not math.isfinite(trigger) or trigger <= 0.0:
            raise ValueError("Avoidance command timestamp is invalid")

        with self._lock:
            preceding = [sample for sample in self._samples if sample.timestamp <= trigger]
            if len(preceding) < self.speed_window_points:
                raise ValueError(
                    f"Need {self.speed_window_points} GNSS points before the command timestamp"
                )
            aligned = preceding[-1]
            skew = trigger - aligned.timestamp
            if skew < 0.0 or skew > self.timestamp_tolerance_s:
                raise ValueError(
                    f"GNSS/command timestamp skew {skew:.3f}s exceeds "
                    f"{self.timestamp_tolerance_s:.3f}s"
                )
            speed_window = preceding[-self.speed_window_points :]
            initial_speed = estimate_window_speed(speed_window)
            route = EnuMissionRoute.from_mission(
                mission,
                aligned.latitude_rad,
                aligned.longitude_rad,
            )

            elapsed_since_command = max(0.0, epoch - trigger)
            initial_progress = (
                initial_speed * elapsed_since_command
                if initial_speed >= self.moving_threshold_mps
                else 0.0
            )
            self._route = route
            self._trigger_timestamp = trigger
            self._progress_m = min(route.length_m, initial_progress)
            self._speed_mps = initial_speed
            self._moving = initial_speed >= self.moving_threshold_mps
            self._last_speed_update_monotonic = monotonic
            self._last_progress_monotonic = monotonic

        self._log_route(route, aligned, skew)
        return route

    @staticmethod
    def _log_route(route: EnuMissionRoute, origin: GnssSample, skew_s: float) -> None:
        segment = np.hypot(np.diff(route.east_m), np.diff(route.north_m))
        print(
            "[RealtimeNav] ENU route ready: "
            f"points={len(route.mission_east_m)}, length={route.length_m:.2f}m, "
            f"max_segment={float(np.max(segment)):.2f}m, timestamp_skew={skew_s:.3f}s"
        )
        print(
            "[RealtimeNav] ENU origin: "
            f"lat={origin.latitude_rad:.10f}rad lon={origin.longitude_rad:.10f}rad"
        )
        for index, (east, north) in enumerate(
            zip(route.mission_east_m, route.mission_north_m)
        ):
            print(f"[RealtimeNav] ENU[{index:03d}] east={east:9.3f} north={north:9.3f}")

    def _latest_speed_locked(self, now_epoch: float) -> float:
        if len(self._samples) < self.speed_window_points:
            return 0.0
        latest = self._samples[-1]
        if abs(float(now_epoch) - latest.timestamp) > self.timestamp_tolerance_s:
            return 0.0
        return estimate_window_speed(list(self._samples)[-self.speed_window_points :])

    def _advance_locked(self, now_monotonic: float) -> None:
        if self._last_progress_monotonic is None:
            self._last_progress_monotonic = now_monotonic
            return
        elapsed = max(0.0, now_monotonic - self._last_progress_monotonic)
        self._last_progress_monotonic = now_monotonic
        if self._route is None or not self._moving:
            return
        self._progress_m = min(
            self._route.length_m,
            self._progress_m + self._speed_mps * elapsed,
        )
        if self._progress_m >= self._route.length_m - 1e-6:
            print("[RealtimeNav] Avoidance route completed; returning to straight band")
            self._route = None
            self._progress_m = 0.0
            self._moving = False

    def update(
        self,
        *,
        now_epoch: float | None = None,
        now_monotonic: float | None = None,
    ) -> None:
        epoch = time.time() if now_epoch is None else float(now_epoch)
        monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        with self._lock:
            self._advance_locked(monotonic)
            update_due = (
                self._last_speed_update_monotonic is None
                or monotonic - self._last_speed_update_monotonic
                >= self.speed_update_interval_s - 1e-9
            )
            if update_due:
                self._speed_mps = self._latest_speed_locked(epoch)
                self._moving = self._speed_mps >= self.moving_threshold_mps
                self._last_speed_update_monotonic = monotonic

    def body_path(
        self,
        *,
        now_epoch: float | None = None,
        now_monotonic: float | None = None,
    ) -> np.ndarray:
        epoch = time.time() if now_epoch is None else float(now_epoch)
        monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        self.update(now_epoch=epoch, now_monotonic=monotonic)
        with self._lock:
            if self._route is not None and self._moving:
                return self._route.body_window(
                    self._progress_m,
                    self.lookahead_m,
                    self.sample_spacing_m,
                    self.heading_offset_deg,
                )

        distance = np.linspace(
            0.0,
            self.lookahead_m,
            max(2, int(round(self.lookahead_m / self.sample_spacing_m)) + 1),
            dtype=np.float32,
        )
        return np.column_stack((distance, np.zeros_like(distance))).astype(np.float32)

    def snapshot(self) -> NavigationSnapshot:
        with self._lock:
            route_length = self._route.length_m if self._route is not None else 0.0
            time_to_60 = (
                self.lookahead_m / self._speed_mps
                if self._speed_mps >= self.moving_threshold_mps
                else math.inf
            )
            return NavigationSnapshot(
                route_active=self._route is not None,
                moving=self._moving,
                speed_mps=self._speed_mps,
                time_to_60m_s=time_to_60,
                progress_m=self._progress_m,
                route_length_m=route_length,
            )
