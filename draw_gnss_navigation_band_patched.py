from __future__ import annotations

import argparse
import configparser
import datetime as dt
import math
import sys
import threading
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pandas as pd


try:
    from data_server import prepare_frame_data as _prepare_frame_data
except ImportError:
    _prepare_frame_data = None

try:
    import ar_navigation_video2 as nav_server_module
except ImportError:
    nav_server_module = None


EARTH_RADIUS_M = 6378137.0


def _find_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    by_lower = {str(column).strip().lower(): column for column in df.columns}
    return next((by_lower[name.lower()] for name in names if name.lower() in by_lower), None)


def _clock_value_to_seconds(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, dt.time):
        return (
            value.hour * 3600.0
            + value.minute * 60.0
            + value.second
            + value.microsecond / 1e6
        )
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return float(pd.Timestamp(value).timestamp())

    text = str(value).strip()
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    except ValueError:
        pass

    parsed = pd.to_datetime(text, errors="coerce")
    return float(parsed.timestamp()) if not pd.isna(parsed) else np.nan


def _elapsed_seconds(values: pd.Series) -> np.ndarray:
    """Convert elapsed, clock, or absolute timestamps to seconds from row one."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        seconds = numeric.to_numpy(dtype=np.float64)
        diffs = np.diff(seconds)
        positive = diffs[diffs > 0]
        median_step = float(np.median(positive)) if len(positive) else 0.0

        if np.nanmax(np.abs(seconds)) > 1e11 or median_step >= 5.0:
            seconds /= 1000.0
        elif np.nanmax(np.abs(seconds)) <= 2.0 and 0 < median_step < 1e-3:
            seconds *= 86400.0
    else:
        seconds = np.array([_clock_value_to_seconds(value) for value in values], dtype=np.float64)

    finite = np.isfinite(seconds)
    if not np.any(finite):
        raise ValueError("时间戳列无法解析")
    seconds -= seconds[np.flatnonzero(finite)[0]]
    return seconds


def _read_position_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError:
            if suffix == ".xls":
                raise ImportError("旧式 .xls 需要 xlrd；请另存为 .xlsx 或 .csv")
            return _read_xlsx_without_openpyxl(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"仅支持 Excel/CSV RTK 数据，收到: {path.suffix}")


def _excel_column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    index = 0
    for character in letters.upper():
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _read_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    """Read the first XLSX worksheet using only Python's standard library."""
    spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationships_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_ns = "http://schemas.openxmlformats.org/package/2006/relationships"

    with zipfile.ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{spreadsheet_ns}}}si"):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{{{spreadsheet_ns}}}t"))
                )

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find(f".//{{{spreadsheet_ns}}}sheet")
        if first_sheet is None:
            raise ValueError(f"Excel 中没有工作表: {path}")
        relationship_id = first_sheet.attrib[f"{{{relationships_ns}}}id"]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relation in relationships.findall(f"{{{package_ns}}}Relationship"):
            if relation.attrib.get("Id") == relationship_id:
                target = relation.attrib.get("Target")
                break
        if not target:
            raise ValueError(f"无法定位 Excel 第一张工作表: {path}")
        sheet_path = "xl/" + target.lstrip("/").removeprefix("xl/")
        sheet = ET.fromstring(archive.read(sheet_path))

        rows = []
        max_columns = 0
        for row in sheet.findall(f".//{{{spreadsheet_ns}}}row"):
            values = {}
            for cell in row.findall(f"{{{spreadsheet_ns}}}c"):
                column = _excel_column_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{spreadsheet_ns}}}v")
                inline_node = cell.find(f"{{{spreadsheet_ns}}}is")
                raw_value = value_node.text if value_node is not None else None
                if cell_type == "s" and raw_value is not None:
                    value = shared_strings[int(raw_value)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(
                        node.text or "" for node in inline_node.iter(f"{{{spreadsheet_ns}}}t")
                    )
                elif cell_type == "b" and raw_value is not None:
                    value = raw_value == "1"
                elif raw_value is None:
                    value = None
                else:
                    try:
                        value = float(raw_value)
                    except ValueError:
                        value = raw_value
                values[column] = value
                max_columns = max(max_columns, column + 1)
            rows.append(values)

    if not rows:
        raise ValueError(f"Excel 第一张工作表为空: {path}")
    matrix = [[row.get(column) for column in range(max_columns)] for row in rows]
    headers = [str(value).strip() if value is not None else f"column_{i}" for i, value in enumerate(matrix[0])]
    print("openpyxl 未安装，使用内置 XLSX 读取器")
    return pd.DataFrame(matrix[1:], columns=headers)


def _rdp_keep_mask(points: np.ndarray, tolerance_m: float) -> np.ndarray:
    """Keep the measured polyline shape while removing sub-pixel RTK zigzags."""
    count = len(points)
    keep = np.zeros(count, dtype=bool)
    if count <= 2:
        keep[:] = True
        return keep

    keep[0] = keep[-1] = True
    stack = [(0, count - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        segment = points[last] - points[first]
        segment_length_sq = float(np.dot(segment, segment))
        candidates = points[first + 1:last]
        if segment_length_sq < 1e-12:
            distances = np.linalg.norm(candidates - points[first], axis=1)
        else:
            relative = candidates - points[first]
            fraction = np.clip(relative @ segment / segment_length_sq, 0.0, 1.0)
            projected = points[first] + fraction[:, None] * segment
            distances = np.linalg.norm(candidates - projected, axis=1)
        relative_index = int(np.argmax(distances))
        if float(distances[relative_index]) > tolerance_m:
            index = first + 1 + relative_index
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return keep


def _simplification_tolerance(points: np.ndarray, horizontal_accuracy: pd.Series | None) -> float:
    resolution = 0.0
    for axis in range(2):
        unique = np.unique(points[:, axis])
        steps = np.diff(unique)
        positive = steps[steps > 1e-6]
        if len(positive):
            resolution = max(resolution, float(np.min(positive)))
    accuracy = 0.0
    if horizontal_accuracy is not None:
        finite = pd.to_numeric(horizontal_accuracy, errors="coerce").dropna()
        if len(finite):
            accuracy = float(finite.median())
    return float(np.clip(max(0.03, resolution * 1.05, accuracy * 3.0), 0.03, 0.15))


def load_rtk(path: str | Path) -> pd.DataFrame:
    """Load centimeter-level RTK positions without model-based smoothing."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    raw = _read_position_table(source)
    time_col = _find_column(raw, ("timestamp", "time", "received_at"))
    lat_col = _find_column(raw, ("lat", "latitude"))
    lon_col = _find_column(raw, ("lon", "lng", "longitude"))
    heading_col = _find_column(raw, ("heading", "yaw", "course"))
    if not all((time_col, lat_col, lon_col, heading_col)):
        raise KeyError(
            "RTK 数据必须包含 Timestamp、Lat、Lon、Heading；"
            f"实际列为 {list(raw.columns)}"
        )

    out = pd.DataFrame(
        {
            "t_sec": _elapsed_seconds(raw[time_col]),
            "lat": pd.to_numeric(raw[lat_col], errors="coerce"),
            "lon": pd.to_numeric(raw[lon_col], errors="coerce"),
            "heading": pd.to_numeric(raw[heading_col], errors="coerce"),
        }
    )
    accuracy_col = _find_column(raw, ("horizontal accuracy (m)", "horizontal_accuracy", "accuracy"))
    if accuracy_col:
        out["horizontal_accuracy"] = pd.to_numeric(raw[accuracy_col], errors="coerce")

    out = out.dropna(subset=["t_sec", "lat", "lon"]).sort_values("t_sec", kind="stable")
    out = out.drop_duplicates(subset="t_sec", keep="last").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("RTK 有效数据少于 2 行")
    if np.any(np.diff(out["t_sec"].to_numpy()) <= 0):
        raise ValueError("RTK 时间戳必须严格递增")

    heading = out["heading"].to_numpy(dtype=np.float64)
    valid_heading = np.isfinite(heading)
    if np.count_nonzero(valid_heading) < 2:
        raise ValueError("Heading 有效数据少于 2 行")
    unwrapped = np.unwrap(np.deg2rad(heading[valid_heading]))
    heading[~valid_heading] = np.rad2deg(
        np.interp(
            np.flatnonzero(~valid_heading),
            np.flatnonzero(valid_heading),
            unwrapped,
        )
    )
    out["heading"] = heading % 360.0

    lat0 = float(out.at[0, "lat"])
    lon0 = float(out.at[0, "lon"])
    out["east"] = (
        EARTH_RADIUS_M
        * np.deg2rad(out["lon"].to_numpy() - lon0)
        * math.cos(math.radians(lat0))
    )
    out["north"] = EARTH_RADIUS_M * np.deg2rad(out["lat"].to_numpy() - lat0)

    measured_points = out[["east", "north"]].to_numpy(dtype=np.float64)
    simplify_tolerance = _simplification_tolerance(
        measured_points,
        out["horizontal_accuracy"] if "horizontal_accuracy" in out else None,
    )
    out["nav_point"] = _rdp_keep_mask(measured_points, tolerance_m=simplify_tolerance)
    navigation_points = measured_points[out["nav_point"].to_numpy()]
    navigation_length = float(
        np.sum(np.hypot(np.diff(navigation_points[:, 0]), np.diff(navigation_points[:, 1])))
    )

    duration = float(out["t_sec"].iloc[-1])
    rate = (len(out) - 1) / duration if duration > 0 else 0.0
    print(f"RTK 加载: {source} ({len(out)} 点, {duration:.3f}s, 约 {rate:.1f}Hz)")
    print(
        f"导航折线: {len(navigation_points)} 个实测拐点, {navigation_length:.2f}m "
        f"(仅移除 <{simplify_tolerance * 100:.1f}cm 小锯齿)"
    )
    print(f"首条 RTK = 视频第 0 帧: t=0.000s")
    print("Heading 约定: 正北=0°, 顺时针增大；直接使用数据列，不从位置反推")
    return out


def _timeline_seconds(values: pd.Series) -> np.ndarray:
    """Parse timestamps while preserving their clock offset from video zero."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite_numeric = numeric.notna()
    if finite_numeric.any() and finite_numeric.sum() == values.notna().sum():
        seconds = numeric.to_numpy(dtype=np.float64)
        finite = np.isfinite(seconds)
        diffs = np.diff(seconds[finite])
        positive = diffs[diffs > 0]
        median_step = float(np.median(positive)) if len(positive) else 0.0
        if np.nanmax(np.abs(seconds)) > 1e11 or median_step >= 5.0:
            seconds /= 1000.0
        elif np.nanmax(np.abs(seconds)) <= 2.0 and 0 < median_step < 1e-3:
            seconds *= 86400.0
    else:
        seconds = np.array([_clock_value_to_seconds(value) for value in values], dtype=np.float64)
    if not np.any(np.isfinite(seconds)):
        raise ValueError("Timestamp column cannot be parsed")
    return seconds


def load_position_track(
    path: str | Path,
    label: str,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> pd.DataFrame:
    """Load timestamped Lat/Lon data without requiring a Heading column."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    raw = _read_position_table(source)
    time_col = _find_column(raw, ("timestamp", "time", "received_at"))
    lat_col = _find_column(raw, ("lat", "latitude"))
    lon_col = _find_column(raw, ("lon", "lng", "longitude"))
    if not all((time_col, lat_col, lon_col)):
        raise KeyError(
            "GNSS data must contain Timestamp, Lat and Lon; "
            f"actual columns: {list(raw.columns)}"
        )

    out = pd.DataFrame(
        {
            "t_sec": _timeline_seconds(raw[time_col]),
            "lat": pd.to_numeric(raw[lat_col], errors="coerce"),
            "lon": pd.to_numeric(raw[lon_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["t_sec", "lat", "lon"]).sort_values("t_sec", kind="stable")
    out = out.drop_duplicates(subset="t_sec", keep="last").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError(f"{label} has fewer than two valid GNSS positions")

    lat0 = float(out.at[0, "lat"]) if origin_lat is None else float(origin_lat)
    lon0 = float(out.at[0, "lon"]) if origin_lon is None else float(origin_lon)
    out["east"] = (
        EARTH_RADIUS_M
        * np.deg2rad(out["lon"].to_numpy(dtype=np.float64) - lon0)
        * math.cos(math.radians(lat0))
    )
    out["north"] = EARTH_RADIUS_M * np.deg2rad(
        out["lat"].to_numpy(dtype=np.float64) - lat0
    )

    first_t = float(out["t_sec"].iloc[0])
    last_t = float(out["t_sec"].iloc[-1])
    duration = last_t - first_t
    rate = (len(out) - 1) / duration if duration > 0 else 0.0
    print(
        f"{label}: {source} ({len(out)} points, {first_t:.3f}~{last_t:.3f}s, "
        f"about {rate:.1f}Hz)"
    )
    return out


class CalibratedGroundProjector:
    """Project boat-frame ground points through the LiDAR-to-camera calibration."""

    def __init__(self, ini_path: str | Path, width: int, height: int, ground_z: float):
        cfg = configparser.ConfigParser()
        if not cfg.read(ini_path, encoding="utf-8"):
            raise FileNotFoundError(ini_path)

        calib_width = cfg.getfloat("calibration", "width")
        calib_height = cfg.getfloat("calibration", "height")
        sx = width / calib_width
        sy = height / calib_height
        self.K = np.array(
            [
                [cfg.getfloat("calibration", "fx") * sx, 0.0, cfg.getfloat("calibration", "cx") * sx],
                [0.0, cfg.getfloat("calibration", "fy") * sy, cfg.getfloat("calibration", "cy") * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rows = [
            [float(value) for value in cfg.get("calibration", f"extrinsic_row{i}").split(",")]
            for i in range(4)
        ]
        extrinsic = np.asarray(rows, dtype=np.float64)
        self.rotation = extrinsic[:3, :3]
        self.translation = extrinsic[:3, 3]
        self.rvec, _ = cv2.Rodrigues(self.rotation)
        self.distortion = np.array(
            [cfg.getfloat("distortion", key) for key in ("k1", "k2", "p1", "p2", "k3")],
            dtype=np.float64,
        )
        self.width = width
        self.height = height
        self.ground_z = float(ground_z)

    @property
    def optical_center_x(self) -> float:
        return float(self.K[0, 2])

    def project(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(points_xy) == 0:
            return np.empty((0, 2), np.float32), np.empty(0, dtype=bool)

        points_3d = np.column_stack(
            (points_xy[:, 0], points_xy[:, 1], np.full(len(points_xy), self.ground_z))
        ).astype(np.float64)
        camera_points = points_3d @ self.rotation.T + self.translation
        pixels, _ = cv2.projectPoints(
            points_3d,
            self.rvec,
            self.translation,
            self.K,
            self.distortion,
        )
        pixels = pixels.reshape(-1, 2)
        finite = np.isfinite(pixels).all(axis=1)
        valid = (
            (camera_points[:, 2] > 0.05)
            & finite
            & (pixels[:, 0] > -8 * self.width)
            & (pixels[:, 0] < 9 * self.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < self.height)
        )
        return pixels, valid


def interpolate_pose(rtk: pd.DataFrame, query_t: float) -> tuple[float, float, float] | None:
    times = rtk["t_sec"].to_numpy()
    if query_t < times[0] or query_t > times[-1]:
        return None
    east = float(np.interp(query_t, times, rtk["east"].to_numpy()))
    north = float(np.interp(query_t, times, rtk["north"].to_numpy()))
    heading_rad = np.unwrap(np.deg2rad(rtk["heading"].to_numpy()))
    heading = math.degrees(float(np.interp(query_t, times, heading_rad))) % 360.0
    return east, north, heading


def future_path_in_body(
    rtk: pd.DataFrame,
    query_t: float,
    lookahead_m: float,
    sample_spacing_m: float,
    heading_offset_deg: float,
) -> np.ndarray | None:
    pose = interpolate_pose(rtk, query_t)
    if pose is None:
        return None
    boat_east, boat_north, heading = pose
    times = rtk["t_sec"].to_numpy()
    start = int(np.searchsorted(times, query_t, side="right"))

    navigation_mask = rtk["nav_point"].to_numpy(dtype=bool)[start:]
    east = np.concatenate(([boat_east], rtk["east"].to_numpy()[start:][navigation_mask]))
    north = np.concatenate(([boat_north], rtk["north"].to_numpy()[start:][navigation_mask]))
    if len(east) < 2:
        return None

    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(east), np.diff(north)))))
    unique = np.concatenate(([True], np.diff(arc) > 0.002))
    east, north, arc = east[unique], north[unique], arc[unique]
    if len(arc) < 2 or arc[-1] < 1e-4:
        return None

    path_length = min(float(lookahead_m), float(arc[-1]))
    sample_arc = np.arange(0.0, path_length, sample_spacing_m)
    if len(sample_arc) == 0 or sample_arc[-1] < path_length:
        sample_arc = np.append(sample_arc, path_length)
    sampled_east = np.interp(sample_arc, arc, east)
    sampled_north = np.interp(sample_arc, arc, north)

    yaw = math.radians(heading + heading_offset_deg)
    de = sampled_east - boat_east
    dn = sampled_north - boat_north
    x_forward = de * math.sin(yaw) + dn * math.cos(yaw)
    y_right = de * math.cos(yaw) - dn * math.sin(yaw)
    points = np.column_stack((x_forward, y_right))

    forward = points[:, 0] >= 0.0
    if np.count_nonzero(forward) < 2:
        return None
    return points[forward].astype(np.float32)


def estimate_heading_from_history(
    history: pd.DataFrame,
    query_t: float,
    point_count: int = 10,
) -> float:
    """Estimate north-clockwise heading from the most recent history samples."""
    times = history["t_sec"].to_numpy(dtype=np.float64)
    end = int(np.searchsorted(times, query_t, side="right"))
    if end < 2:
        end = min(len(history), max(2, point_count))
    start = max(0, end - point_count)
    points = history[["east", "north"]].to_numpy(dtype=np.float64)[start:end]

    # Linear regression across the window is less sensitive to centimetre RTK jitter
    # than the bearing between only the last two samples.
    index = np.arange(len(points), dtype=np.float64)
    centered = index - index.mean()
    denominator = float(np.dot(centered, centered))
    if denominator > 0:
        de = float(np.dot(centered, points[:, 0]) / denominator)
        dn = float(np.dot(centered, points[:, 1]) / denominator)
    else:
        de = dn = 0.0

    if math.hypot(de, dn) < 1e-5:
        de = float(points[-1, 0] - points[0, 0])
        dn = float(points[-1, 1] - points[0, 1])
    if math.hypot(de, dn) < 1e-4:
        raise ValueError("Recent GNSS history is stationary; heading cannot be estimated")
    return math.degrees(math.atan2(de, dn)) % 360.0


def straight_path_in_body(lookahead_m: float, sample_spacing_m: float) -> np.ndarray:
    distance = np.arange(0.0, lookahead_m, sample_spacing_m, dtype=np.float32)
    if len(distance) == 0 or distance[-1] < lookahead_m:
        distance = np.append(distance, np.float32(lookahead_m))
    return np.column_stack((distance, np.zeros_like(distance))).astype(np.float32)


def circular_smooth_headings(
    east: np.ndarray,
    north: np.ndarray,
    point_window: int = 20,
) -> np.ndarray:
    """Circular-average the segment tangents inside one rolling point window."""
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    count = len(east)
    if count < 2:
        return np.zeros(count, dtype=np.float64)

    window = min(count, max(2, int(point_window)))
    segment_de = np.diff(east)
    segment_dn = np.diff(north)
    segment_length = np.hypot(segment_de, segment_dn)
    valid_segment = segment_length > 1e-4
    segment_heading = np.arctan2(segment_de, segment_dn)
    tangent_weights = (
        np.arange(1, window, dtype=np.float64)
        * np.arange(window - 1, 0, -1, dtype=np.float64)
    )
    headings = np.full(count, np.nan, dtype=np.float64)

    # Each result reuses only the tangents formed by the same point window.
    # There is deliberately no second moving-average window.
    left_span = (window - 1) // 2
    for index in range(count):
        first = int(np.clip(index - left_span, 0, count - window))
        last = first + window
        valid = valid_segment[first : last - 1]
        if not np.any(valid):
            continue
        angles = segment_heading[first : last - 1][valid]
        # Triangular least-squares weights reuse every tangent in this same
        # point window while preventing either noisy endpoint from dominating.
        weights = (
            segment_length[first : last - 1][valid]
            * tangent_weights[valid]
        )
        headings[index] = math.atan2(
            float(np.average(np.sin(angles), weights=weights)),
            float(np.average(np.cos(angles), weights=weights)),
        )

    good = np.flatnonzero(np.isfinite(headings))
    if len(good) == 0:
        return np.zeros(count, dtype=np.float64)
    unwrapped = np.unwrap(headings[good])
    missing = np.flatnonzero(~np.isfinite(headings))
    if len(missing):
        headings[missing] = np.interp(missing, good, unwrapped)
    headings[good] = unwrapped
    return np.unwrap(headings)


def prediction_headings_from_track(
    prediction: pd.DataFrame,
    point_window: int = 20,
) -> np.ndarray:
    """Estimate headings exclusively from prediction positions."""
    east = prediction["east"].to_numpy(dtype=np.float64)
    north = prediction["north"].to_numpy(dtype=np.float64)
    step = np.hypot(np.diff(east), np.diff(north))
    breaks = np.flatnonzero(step > 2.0) + 1
    bounds = np.concatenate(([0], breaks, [len(prediction)]))
    headings = np.empty(len(prediction), dtype=np.float64)
    for first, last in zip(bounds[:-1], bounds[1:]):
        headings[first:last] = circular_smooth_headings(
            east[first:last],
            north[first:last],
            point_window,
        )
    return np.unwrap(headings)


def circular_interpolate(start: float, end: float, fraction: float) -> float:
    """Interpolate two radians along the shortest arc on the unit circle."""
    fraction = float(np.clip(fraction, 0.0, 1.0))
    delta = math.atan2(math.sin(end - start), math.cos(end - start))
    return float(start + fraction * delta)


def interpolate_prediction_sample(
    prediction: pd.DataFrame,
    query_t: float,
) -> tuple[float, float, float, int]:
    """Interpolate one prediction pose without crossing a packet reset."""
    times = prediction["t_sec"].to_numpy(dtype=np.float64)
    east = prediction["east"].to_numpy(dtype=np.float64)
    north = prediction["north"].to_numpy(dtype=np.float64)
    headings = prediction["heading_rad"].to_numpy(dtype=np.float64)
    query_t = float(np.clip(query_t, times[0], times[-1]))
    right = int(np.searchsorted(times, query_t, side="right"))
    if right == 0:
        return float(east[0]), float(north[0]), float(headings[0]), 1
    if right >= len(times):
        return float(east[-1]), float(north[-1]), float(headings[-1]), len(times)

    left = right - 1
    gap = math.hypot(float(east[right] - east[left]), float(north[right] - north[left]))
    if gap > 2.0:
        index = left if query_t - times[left] <= times[right] - query_t else right
        return (
            float(east[index]),
            float(north[index]),
            float(headings[index]),
            index + 1,
        )

    fraction = (query_t - times[left]) / max(times[right] - times[left], 1e-9)
    return (
        float(east[left] + fraction * (east[right] - east[left])),
        float(north[left] + fraction * (north[right] - north[left])),
        circular_interpolate(float(headings[left]), float(headings[right]), fraction),
        right,
    )


def smooth_track_coordinates(values: np.ndarray, point_window: int) -> np.ndarray:
    """Suppress centimeter GNSS zigzags before measuring path arc length."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return values.copy()
    window = min(len(values), max(3, int(point_window)))
    left = (window - 1) // 2
    right = window // 2
    padded = np.pad(values, (left, right), mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    smoothed[0] = values[0]
    return smoothed


def prediction_path_in_body(
    history: pd.DataFrame,
    prediction: pd.DataFrame,
    query_t: float,
    lookahead_m: float,
    sample_spacing_m: float,
    heading_offset_deg: float,
    heading_points: int = 20,
    yaw_heading_rad: float | None = None,
    history_heading_rad: float | None = None,
    connection_progress: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a future body-frame path, including the history/prediction join."""
    prediction_east = prediction["east"].to_numpy(dtype=np.float64)
    prediction_north = prediction["north"].to_numpy(dtype=np.float64)
    times = prediction["t_sec"].to_numpy(dtype=np.float64)
    prediction_headings = (
        prediction["heading_rad"].to_numpy(dtype=np.float64)
        if "heading_rad" in prediction
        else circular_smooth_headings(
            prediction_east,
            prediction_north,
            point_window=heading_points,
        )
    )

    predicted_anchor_east, predicted_anchor_north, prediction_heading, start = (
        interpolate_prediction_sample(prediction, query_t)
    )
    current_heading = (
        prediction_heading if yaw_heading_rad is None else float(yaw_heading_rad)
    )

    connection_progress = float(np.clip(connection_progress, 0.0, 1.0))
    if connection_progress < 1.0:
        # During the mode transition, move from the final measured position to
        # the time-matched prediction position. Keeping both anchors in the path
        # makes the history/prediction connection explicit and avoids a one-frame jump.
        history_east = float(history["east"].iloc[-1])
        history_north = float(history["north"].iloc[-1])
        anchor_east = (
            (1.0 - connection_progress) * history_east
            + connection_progress * predicted_anchor_east
        )
        anchor_north = (
            (1.0 - connection_progress) * history_north
            + connection_progress * predicted_anchor_north
        )
        joined_heading = (
            current_heading if history_heading_rad is None else float(history_heading_rad)
        )
        east = np.concatenate(
            ([anchor_east, predicted_anchor_east], prediction_east[start:])
        )
        north = np.concatenate(
            ([anchor_north, predicted_anchor_north], prediction_north[start:])
        )
        headings = np.concatenate(
            ([joined_heading, prediction_heading], prediction_headings[start:])
        )
    else:
        anchor_east = predicted_anchor_east
        anchor_north = predicted_anchor_north
        east = np.concatenate(([anchor_east], prediction_east[start:]))
        north = np.concatenate(([anchor_north], prediction_north[start:]))
        headings = np.concatenate(([prediction_heading], prediction_headings[start:]))

    # A prediction packet reset must not be connected as a 22 m backward move.
    # Work only with the continuous packet containing the current anchor.
    route_step = np.hypot(np.diff(east), np.diff(north))
    discontinuity = np.flatnonzero(route_step > 2.0)
    if len(discontinuity):
        stop = int(discontinuity[0]) + 1
        east, north, headings = east[:stop], north[:stop], headings[:stop]
    if len(east) < 2:
        east = np.append(
            east,
            east[-1] + sample_spacing_m * math.sin(prediction_heading),
        )
        north = np.append(
            north,
            north[-1] + sample_spacing_m * math.cos(prediction_heading),
        )
        headings = np.append(headings, prediction_heading)

    east = smooth_track_coordinates(east, heading_points)
    north = smooth_track_coordinates(north, heading_points)
    simplified = _rdp_keep_mask(np.column_stack((east, north)), tolerance_m=0.03)
    east, north, headings = east[simplified], north[simplified], headings[simplified]

    yaw = current_heading + math.radians(heading_offset_deg)
    forward_distance = (
        (east - anchor_east) * math.sin(yaw)
        + (north - anchor_north) * math.cos(yaw)
    )
    # Do not draw a later prediction that has already turned back toward the boat;
    # that would make the band loop down into the camera view. Keep the forward prefix.
    peak = int(np.argmax(forward_distance))
    east, north, headings = east[: peak + 1], north[: peak + 1], headings[: peak + 1]
    if len(east) < 2:
        east = np.array(
            [
                anchor_east,
                anchor_east + sample_spacing_m * math.sin(prediction_heading),
            ],
            dtype=np.float64,
        )
        north = np.array(
            [
                anchor_north,
                anchor_north + sample_spacing_m * math.cos(prediction_heading),
            ],
            dtype=np.float64,
        )
        headings = np.array(
            [prediction_heading, prediction_heading],
            dtype=np.float64,
        )

    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(east), np.diff(north)))))
    unique = np.concatenate(([True], np.diff(arc) > 0.002))
    east, north, headings, arc = east[unique], north[unique], headings[unique], arc[unique]
    if len(arc) < 2 or arc[-1] < 1e-4:
        return None

    path_length = min(float(lookahead_m), float(arc[-1]))
    sample_arc = np.arange(0.0, path_length, sample_spacing_m)
    if len(sample_arc) == 0 or sample_arc[-1] < path_length:
        sample_arc = np.append(sample_arc, path_length)
    sampled_east = np.interp(sample_arc, arc, east)
    sampled_north = np.interp(sample_arc, arc, north)
    headings = np.unwrap(headings)
    sampled_headings = np.interp(sample_arc, arc, headings)

    # The rolling prediction tangent at query_t defines the current vehicle yaw.
    de = sampled_east - anchor_east
    dn = sampled_north - anchor_north
    x_forward = de * math.sin(yaw) + dn * math.cos(yaw)
    y_right = de * math.cos(yaw) - dn * math.sin(yaw)
    local_headings = sampled_headings - yaw

    points = np.column_stack((x_forward, y_right))
    forward = points[:, 0] >= -0.05
    if np.count_nonzero(forward) < 2:
        return None
    points = points[forward]
    local_headings = local_headings[forward]
    points[0] = [0.0, 0.0]

    actual_arc = float(
        np.sum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1])))
    )
    if actual_arc < lookahead_m - 1e-3:
        remaining = lookahead_m - actual_arc
        count = max(1, int(math.ceil(remaining / sample_spacing_m)))
        distance = np.minimum(
            np.arange(1, count + 1, dtype=np.float64) * sample_spacing_m,
            remaining,
        )
        direction = math.atan2(
            math.sin(float(local_headings[-1])),
            math.cos(float(local_headings[-1])),
        )
        extension = np.column_stack(
            (
                points[-1, 0] + distance * math.cos(direction),
                points[-1, 1] + distance * math.sin(direction),
            )
        )
        points = np.vstack((points, extension))
        local_headings = np.append(
            local_headings,
            np.full(len(extension), direction, dtype=np.float64),
        )
    return points.astype(np.float32), local_headings.astype(np.float64)


class NavigationPathSmoother:
    """Smooth the displayed path while keeping RTK samples unchanged."""

    def __init__(self, temporal_alpha: float = 0.90, spatial_window: int = 11):
        self.temporal_alpha = float(temporal_alpha)
        self.spatial_window = max(3, int(spatial_window) | 1)
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def _spatial(self, path: np.ndarray) -> np.ndarray:
        if len(path) < 3:
            return path.astype(np.float32, copy=True)
        kernel = np.ones(self.spatial_window, dtype=np.float64)
        kernel = np.convolve(kernel, kernel)
        kernel /= kernel.sum()
        radius = len(kernel) // 2
        padded = np.pad(path.astype(np.float64), ((radius, radius), (0, 0)), mode="edge")
        smoothed = np.column_stack(
            [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)]
        )
        smoothed[0] = path[0]
        return smoothed.astype(np.float32)

    @staticmethod
    def _resample(path: np.ndarray, normalized_distance: np.ndarray) -> np.ndarray:
        distance = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])))))
        if distance[-1] < 1e-6:
            return np.repeat(path[:1], len(normalized_distance), axis=0)
        distance /= distance[-1]
        return np.column_stack(
            [np.interp(normalized_distance, distance, path[:, axis]) for axis in range(2)]
        )

    def update(self, path: np.ndarray | None) -> np.ndarray | None:
        if path is None or len(path) < 2:
            self.reset()
            return None

        current = self._spatial(path)
        if self.previous is not None:
            current_distance = np.concatenate(
                ([0.0], np.cumsum(np.hypot(np.diff(current[:, 0]), np.diff(current[:, 1]))))
            )
            if current_distance[-1] > 1e-6:
                normalized = current_distance / current_distance[-1]
                previous = self._resample(self.previous, normalized)
                current = self.temporal_alpha * previous + (1.0 - self.temporal_alpha) * current
        current[0] = [0.0, 0.0]
        self.previous = current.astype(np.float32)
        return self.previous.copy()


def band_boundaries(
    center: np.ndarray,
    width_m: float,
    local_headings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if local_headings is not None and len(local_headings) == len(center):
        right_normal = np.column_stack(
            (-np.sin(local_headings), np.cos(local_headings))
        )
    else:
        tangent = np.gradient(center.astype(np.float64), axis=0)
        norm = np.hypot(tangent[:, 0], tangent[:, 1])
        norm[norm < 1e-6] = 1.0
        right_normal = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
    offset = right_normal * (width_m / 2.0)
    return (center - offset).astype(np.float32), (center + offset).astype(np.float32)


def project_navigation_band(
    projector: CalibratedGroundProjector,
    center: np.ndarray,
    width_m: float,
    local_headings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    left, right = band_boundaries(center, width_m, local_headings)
    left_px, left_valid = projector.project(left)
    right_px, right_valid = projector.project(right)
    common = left_valid & right_valid
    valid_indices = np.flatnonzero(common)
    if len(valid_indices) < 2:
        return np.empty((0, 2), np.int32), np.empty((0, 2), np.int32)

    # Keep only the first continuous visible portion. Do not bridge an off-screen gap.
    end = 1
    while end < len(valid_indices) and valid_indices[end] == valid_indices[end - 1] + 1:
        end += 1
    if end < 2:
        return np.empty((0, 2), np.int32), np.empty((0, 2), np.int32)
    visible_indices = valid_indices[:end]

    left_px = np.rint(left_px[visible_indices]).astype(np.int32)
    right_px = np.rint(right_px[visible_indices]).astype(np.int32)
    center_px = np.rint((left_px.astype(np.float32) + right_px) / 2.0).astype(np.int32)
    polygon = np.vstack((left_px, right_px[::-1]))
    return polygon, center_px


def anchor_band_to_bottom(
    polygon: np.ndarray,
    center: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate the visible band so its near center is exactly bottom-center."""
    if len(center) == 0:
        return polygon, center
    target = np.array([width / 2.0, height - 1.0], dtype=np.float64)
    offset = center[0].astype(np.float64) - target
    return (
        np.rint(polygon.astype(np.float64) - offset).astype(np.int32),
        np.rint(center.astype(np.float64) - offset).astype(np.int32),
    )


def draw_navigation_band(frame: np.ndarray, polygon: np.ndarray, center: np.ndarray) -> np.ndarray:
    if len(polygon) < 4:
        return frame
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon.reshape(-1, 1, 2)], (30, 205, 85), cv2.LINE_AA)
    output = cv2.addWeighted(overlay, 0.34, frame, 0.66, 0.0)
    split = len(polygon) // 2
    cv2.polylines(output, [polygon[:split].reshape(-1, 1, 2)], False, (80, 255, 135), 2, cv2.LINE_AA)
    cv2.polylines(output, [polygon[split:][::-1].reshape(-1, 1, 2)], False, (80, 255, 135), 2, cv2.LINE_AA)
    cv2.polylines(output, [center.reshape(-1, 1, 2)], False, (220, 255, 225), 2, cv2.LINE_AA)
    return output


def _prepare_message(width: int, height: int, curve: np.ndarray, frame_id: int, timestamp: float):
    if _prepare_frame_data is not None:
        return _prepare_frame_data(
            width=width,
            height=height,
            curve=curve,
            frame_id=frame_id,
            timestamp=timestamp,
        )
    return {
        "width": width,
        "height": height,
        "curve": np.asarray(curve).tolist(),
        "frame_id": frame_id,
        "timestamp": timestamp,
    }


def _start_tcp_server(no_tcp: bool) -> bool:
    if no_tcp:
        return False
    if nav_server_module is None:
        print("未找到 ar_navigation_video2，跳过 TCP；本地视频叠加不受影响")
        return False
    threading.Thread(target=nav_server_module.start_tcp_server, daemon=True).start()
    time.sleep(1.0)
    print("TCP 服务已启动: tcp://0.0.0.0:8765")
    return True


def overlay(args) -> None:
    history = load_position_track(args.gnss_history, "GNSS history")
    origin_lat = float(history.at[0, "lat"])
    origin_lon = float(history.at[0, "lon"])
    prediction = load_position_track(
        args.gnss_prediction,
        "GNSS prediction",
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    prediction_start = (
        float(prediction["t_sec"].iloc[0])
        if args.prediction_start is None
        else float(args.prediction_start)
    )
    history_heading_deg = estimate_heading_from_history(
        history,
        prediction_start,
        args.history_heading_points,
    )
    history_heading_rad = math.radians(history_heading_deg)
    prediction_headings = prediction_headings_from_track(
        prediction,
        args.heading_points,
    )
    prediction["heading_rad"] = prediction_headings
    connection_distance = math.hypot(
        float(prediction["east"].iloc[0] - history["east"].iloc[-1]),
        float(prediction["north"].iloc[0] - history["north"].iloc[-1]),
    )
    print(
        f"Prediction headings: rolling {args.heading_points}-point direction with "
        f"circular smoothing; initial {math.degrees(prediction_headings[0]) % 360.0:.2f} deg"
    )
    print(
        f"History heading: latest {min(len(history), args.history_heading_points)} points -> "
        f"{history_heading_deg:.2f} deg"
    )
    print(
        f"Prediction becomes available at video {prediction_start:.3f}s; "
        f"history-to-prediction connection is {connection_distance:.2f}m"
    )
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise ValueError("视频 FPS 无效")

    video_duration = total / fps
    print(f"视频: {width}x{height}, {fps:.3f}fps, {total} 帧, {video_duration:.3f}s")
    print(f"时间对齐: frame 0 -> RTK 0.000s；frame n -> RTK n/{fps:.3f}s")
    print(
        f"Path modes: straight before {prediction_start:.3f}s, "
        "Lat/Lon prediction after it"
    )

    projector = CalibratedGroundProjector(
        args.calib_ini,
        width,
        height,
        ground_z=-abs(args.lidar_height),
    )
    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {args.output}")

    tcp_enabled = _start_tcp_server(args.no_tcp)
    path_smoother = NavigationPathSmoother(
        temporal_alpha=args.smooth_alpha,
        spatial_window=args.smooth_window,
    )
    frame_idx = 0
    frames_with_band = 0
    preview_frame = None
    previous_mode = None
    previous_polygon = np.empty((0, 2), np.int32)
    previous_center_px = np.empty((0, 2), np.int32)

    try:
        while True:
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            rtk_time = frame_idx / fps + args.time_offset
            mode = "prediction" if rtk_time >= prediction_start else "straight"
            if mode != previous_mode:
                path_smoother.reset()
                previous_mode = mode

            if mode == "prediction":
                transition_blend = (
                    1.0
                    if args.switch_smooth_seconds <= 0
                    else float(
                        np.clip(
                            (rtk_time - prediction_start) / args.switch_smooth_seconds,
                            0.0,
                            1.0,
                        )
                    )
                )
                _, _, raw_heading, _ = interpolate_prediction_sample(
                    prediction,
                    rtk_time,
                )
                connected_heading = circular_interpolate(
                    history_heading_rad,
                    raw_heading,
                    transition_blend,
                )
                prediction_result = prediction_path_in_body(
                    history,
                    prediction,
                    rtk_time,
                    args.lookahead,
                    args.sample_spacing,
                    args.heading_offset,
                    args.heading_points,
                    yaw_heading_rad=connected_heading,
                    history_heading_rad=history_heading_rad,
                    connection_progress=transition_blend,
                )
                if prediction_result is None:
                    body_path = None
                    local_headings = None
                else:
                    body_path, local_headings = prediction_result
                    if args.switch_smooth_seconds > 0:
                        straight = np.column_stack(
                            (
                                np.linspace(0.0, args.lookahead, len(body_path)),
                                np.zeros(len(body_path)),
                            )
                        )
                        body_path = (
                            (1.0 - transition_blend) * straight
                            + transition_blend * body_path
                        ).astype(np.float32)
            else:
                # Until prediction arrives, render a stable forward strip. Heading is
                # not needed because the centerline is already in vehicle coordinates.
                body_path = straight_path_in_body(args.lookahead, args.sample_spacing)
                local_headings = None
            polygon = np.empty((0, 2), np.int32)
            center_px = np.empty((0, 2), np.int32)
            if body_path is not None:
                body_path = path_smoother.update(body_path)
                # The centerline has just been spatially and temporally smoothed.
                # Derive its band normals from that final geometry so stale heading
                # samples cannot make the two band edges shake independently.
                local_headings = None
                polygon, center_px = project_navigation_band(
                    projector,
                    body_path,
                    args.band_width,
                    local_headings,
                )
                polygon, center_px = anchor_band_to_bottom(
                    polygon,
                    center_px,
                    width,
                    height,
                )
            else:
                path_smoother.reset()

            if len(polygon) >= 4:
                previous_polygon = polygon.copy()
                previous_center_px = center_px.copy()
            elif len(previous_polygon) >= 4:
                # A few prediction samples can momentarily leave the calibrated
                # field of view. Hold the last valid band instead of flashing off.
                polygon = previous_polygon.copy()
                center_px = previous_center_px.copy()

            display = draw_navigation_band(frame, polygon, center_px)
            if len(polygon) >= 4:
                frames_with_band += 1
            if frame_idx == 0:
                preview_frame = display.copy()

            if writer is not None:
                writer.write(display)
            if tcp_enabled:
                message = _prepare_message(width, height, center_px, frame_idx, frame_idx / fps)
                message.update(
                    {
                        "video_time": frame_idx / fps,
                        "rtk_time": rtk_time,
                        "sync_mode": "first_frame_rtk_zero",
                        "source_fps": fps,
                        "band_width_m": args.band_width,
                    }
                )
                if nav_server_module.nav_server:
                    nav_server_module.nav_server.send_prepared_nav(message, frame_idx)

            if args.display:
                cv2.imshow("RTK navigation band", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            frame_idx += 1

            if frame_idx % 250 == 0:
                print(f"  {frame_idx}/{total} 帧，导航带帧数 {frames_with_band}")
            if args.realtime or tcp_enabled:
                delay = 1.0 / fps - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if args.preview and preview_frame is not None:
        cv2.imwrite(str(args.preview), preview_frame)
        print(f"首帧预览: {args.preview}")
    print(f"完成: {frame_idx} 帧，其中 {frames_with_band} 帧绘制导航带")
    if args.output:
        print(f"输出视频: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="厘米级 RTK 轨迹 -> 视频导航带")
    parser.add_argument(
        "--gnss-history",
        default="gnss_history.xlsx",
        help="ship history track with Timestamp/Lat/Lon",
    )
    parser.add_argument(
        "--gnss-prediction",
        "--gnss",
        "--csv",
        dest="gnss_prediction",
        default="gnss_csv.xlsx",
        help="avoidance prediction track with Timestamp/Lat/Lon",
    )
    parser.add_argument(
        "--prediction-start",
        type=float,
        default=None,
        help="video second when prediction arrives; default is its first Timestamp",
    )
    parser.add_argument("--switch-smooth-seconds",type=float,default=1.0,help="seconds used to blend the straight strip into the prediction path",)
    parser.add_argument("--history-heading-points",type=int,default=10,help="recent history points used for the initial heading; use 20 for faster response",)
    parser.add_argument("--heading-points",type=int,default=20,help="prediction points reused for one local-tangent circular average",)
    parser.add_argument("--video", default="video.mp4", help="输入视频")
    parser.add_argument("--output", default="video_rtk_navigation1.mp4", help="输出 MP4；传空字符串则不输出")
    parser.add_argument("--preview", default="rtk_navigation_first_frame.jpg", help="保存首帧预览；传空字符串则不保存")
    parser.add_argument("--calib-ini", default="calibration.ini", help="相机-LiDAR 标定文件")
    parser.add_argument("--lidar-height", type=float, default=1.6, help="LiDAR 距地面/水面高度，米")
    parser.add_argument("--lookahead", type=float, default=60.0, help="沿真实轨迹向前绘制距离，米")
    parser.add_argument("--sample-spacing", type=float, default=0.25, help="导航带中心线采样间隔，米")
    parser.add_argument("--band-width", type=float, default=2.2, help="导航带实际宽度，米")
    parser.add_argument("--smooth-alpha",type=float,default=0.90,help="时间平滑中上一帧轨迹权重，越大越稳，默认 0.90",)
    parser.add_argument("--smooth-window",type=int,default=11,help="空间平滑窗口基数，默认 11（实际使用奇数窗口）",)
    parser.add_argument("--heading-offset", "--yaw-offset", dest="heading_offset", type=float, default=0.0, help="Heading 安装修正角，度")
    parser.add_argument("--time-offset", type=float, default=0.0, help="RTK 相对视频的额外时间偏移，秒")
    parser.add_argument("--display", action="store_true", help="显示实时预览窗口")
    parser.add_argument("--realtime", action="store_true", help="按视频帧率运行，否则尽快离线处理")
    parser.add_argument("--no-tcp", action="store_true", help="不启动可选 TCP 服务")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.lookahead <= 0
        or args.sample_spacing <= 0
        or args.band_width <= 0
        or not 0.0 <= args.smooth_alpha < 1.0
        or args.smooth_window < 3
        or args.history_heading_points < 2
        or args.heading_points < 2
        or args.switch_smooth_seconds < 0
    ):
        print("lookahead、sample-spacing、band-width 必须大于 0；smooth-alpha 必须在 [0,1)", file=sys.stderr)
        raise SystemExit(2)
    overlay(args)


if __name__ == "__main__":
    main()
