from __future__ import annotations

import configparser
import datetime as dt
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6378137.0


def _find_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    columns = {str(column).strip().lower(): column for column in df.columns}
    return next((columns[name.lower()] for name in names if name.lower() in columns), None)


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
    parts = str(value).strip().split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    except ValueError:
        pass
    parsed = pd.to_datetime(str(value), errors="coerce")
    return float(parsed.timestamp()) if not pd.isna(parsed) else np.nan


def _timeline_seconds(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == values.notna().sum():
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
        seconds = np.array([_clock_value_to_seconds(value) for value in values])
    if not np.any(np.isfinite(seconds)):
        raise ValueError("GNSS Timestamp 列无法解析")
    return seconds


def _excel_column_index(reference: str) -> int:
    index = 0
    for character in "".join(c for c in reference if c.isalpha()).upper():
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _read_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationships_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{{{spreadsheet_ns}}}t"))
                for item in root.findall(f"{{{spreadsheet_ns}}}si")
            ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find(f".//{{{spreadsheet_ns}}}sheet")
        if first_sheet is None:
            raise ValueError(f"Excel 中没有工作表: {path}")
        relationship_id = first_sheet.attrib[f"{{{relationships_ns}}}id"]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(
            (
                relation.attrib.get("Target")
                for relation in relationships.findall(f"{{{package_ns}}}Relationship")
                if relation.attrib.get("Id") == relationship_id
            ),
            None,
        )
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
                raw = value_node.text if value_node is not None else None
                if cell_type == "s" and raw is not None:
                    value = shared_strings[int(raw)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(
                        node.text or "" for node in inline_node.iter(f"{{{spreadsheet_ns}}}t")
                    )
                elif raw is None:
                    value = None
                else:
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw
                values[column] = value
                max_columns = max(max_columns, column + 1)
            rows.append(values)
    matrix = [[row.get(column) for column in range(max_columns)] for row in rows]
    if not matrix:
        raise ValueError(f"Excel 第一张工作表为空: {path}")
    headers = [str(value).strip() for value in matrix[0]]
    print("openpyxl 未安装，使用内置 XLSX 读取器")
    return pd.DataFrame(matrix[1:], columns=headers)


def _read_position_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError:
            if path.suffix.lower() == ".xls":
                raise ImportError("旧式 .xls 需要 xlrd；请另存为 .xlsx 或 .csv")
            return _read_xlsx_without_openpyxl(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"仅支持 Excel/CSV GNSS 数据: {path}")


def load_position_track(
    path: str | Path,
    label: str,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    raw = _read_position_table(source)
    time_col = _find_column(raw, ("timestamp", "time", "received_at"))
    lat_col = _find_column(raw, ("lat", "latitude"))
    lon_col = _find_column(raw, ("lon", "lng", "longitude"))
    if not all((time_col, lat_col, lon_col)):
        raise KeyError(f"GNSS 数据必须包含 Timestamp/Lat/Lon，实际列为 {list(raw.columns)}")
    out = pd.DataFrame(
        {
            "t_sec": _timeline_seconds(raw[time_col]),
            "lat": pd.to_numeric(raw[lat_col], errors="coerce"),
            "lon": pd.to_numeric(raw[lon_col], errors="coerce"),
        }
    )
    out = out.dropna().sort_values("t_sec", kind="stable")
    out = out.drop_duplicates("t_sec", keep="last").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError(f"{label} 有效位置少于两个")
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
    print(
        f"{label}: {source} ({len(out)} points, "
        f"{float(out['t_sec'].iloc[0]):.3f}~{float(out['t_sec'].iloc[-1]):.3f}s)"
    )
    return out


class CalibratedGroundProjector:
    def __init__(self, ini_path: str | Path, width: int, height: int, ground_z: float):
        cfg = configparser.ConfigParser()
        if not cfg.read(ini_path, encoding="utf-8"):
            raise FileNotFoundError(ini_path)
        sx = width / cfg.getfloat("calibration", "width")
        sy = height / cfg.getfloat("calibration", "height")
        self.camera_matrix = np.array(
            [
                [cfg.getfloat("calibration", "fx") * sx, 0.0, cfg.getfloat("calibration", "cx") * sx],
                [0.0, cfg.getfloat("calibration", "fy") * sy, cfg.getfloat("calibration", "cy") * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        extrinsic = np.asarray(
            [
                [float(value) for value in cfg.get("calibration", f"extrinsic_row{i}").split(",")]
                for i in range(4)
            ],
            dtype=np.float64,
        )
        self.rotation = extrinsic[:3, :3]
        self.translation = extrinsic[:3, 3]
        self.rotation_vector, _ = cv2.Rodrigues(self.rotation)
        self.distortion = np.array(
            [cfg.getfloat("distortion", key) for key in ("k1", "k2", "p1", "p2", "k3")]
        )
        self.width = int(width)
        self.height = int(height)
        self.ground_z = float(ground_z)

    def project(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(points_xy) == 0:
            return np.empty((0, 2), np.float32), np.empty(0, dtype=bool)
        points_3d = np.column_stack(
            (points_xy[:, 0], points_xy[:, 1], np.full(len(points_xy), self.ground_z))
        ).astype(np.float64)
        camera_points = points_3d @ self.rotation.T + self.translation
        pixels, _ = cv2.projectPoints(
            points_3d,
            self.rotation_vector,
            self.translation,
            self.camera_matrix,
            self.distortion,
        )
        pixels = pixels.reshape(-1, 2)
        valid = (
            (camera_points[:, 2] > 0.05)
            & np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] > -8 * self.width)
            & (pixels[:, 0] < 9 * self.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < self.height)
        )
        return pixels, valid


def _project_centerline(
    projector: CalibratedGroundProjector,
    body_path: np.ndarray,
) -> np.ndarray:
    pixels, valid = projector.project(body_path)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return np.empty((0, 2), dtype=np.int32)
    end = 1
    while end < len(indices) and indices[end] == indices[end - 1] + 1:
        end += 1
    if end < 2:
        return np.empty((0, 2), dtype=np.int32)
    curve = np.rint(pixels[indices[:end]]).astype(np.int32)
    offset = curve[0].astype(np.float64) - np.array(
        [projector.width / 2.0, projector.height - 1.0]
    )
    return np.rint(curve.astype(np.float64) - offset).astype(np.int32)


def estimate_heading_from_history(
    history: pd.DataFrame,
    query_t: float,
    point_count: int = 10,
) -> float:
    times = history["t_sec"].to_numpy(dtype=np.float64)
    end = int(np.searchsorted(times, query_t, side="right"))
    if end < 2:
        end = min(len(history), max(2, point_count))
    points = history[["east", "north"]].to_numpy(dtype=np.float64)[
        max(0, end - point_count):end
    ]
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
        raise ValueError("最近的 GNSS 历史位置静止，无法估计航向")
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
    left_span = (window - 1) // 2
    for index in range(count):
        first = int(np.clip(index - left_span, 0, count - window))
        last = first + window
        valid = valid_segment[first:last - 1]
        if not np.any(valid):
            continue
        angles = segment_heading[first:last - 1][valid]
        weights = segment_length[first:last - 1][valid] * tangent_weights[valid]
        headings[index] = math.atan2(
            float(np.average(np.sin(angles), weights=weights)),
            float(np.average(np.cos(angles), weights=weights)),
        )
    good = np.flatnonzero(np.isfinite(headings))
    if len(good) == 0:
        return np.zeros(count, dtype=np.float64)
    missing = np.flatnonzero(~np.isfinite(headings))
    if len(missing):
        headings[missing] = np.interp(missing, good, np.unwrap(headings[good]))
    headings[good] = np.unwrap(headings[good])
    return np.unwrap(headings)


def prediction_headings_from_track(
    prediction: pd.DataFrame,
    point_window: int = 20,
) -> np.ndarray:
    east = prediction["east"].to_numpy(dtype=np.float64)
    north = prediction["north"].to_numpy(dtype=np.float64)
    breaks = np.flatnonzero(np.hypot(np.diff(east), np.diff(north)) > 2.0) + 1
    bounds = np.concatenate(([0], breaks, [len(prediction)]))
    headings = np.empty(len(prediction), dtype=np.float64)
    for first, last in zip(bounds[:-1], bounds[1:]):
        headings[first:last] = circular_smooth_headings(
            east[first:last], north[first:last], point_window
        )
    return np.unwrap(headings)


def circular_interpolate(start: float, end: float, fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    delta = math.atan2(math.sin(end - start), math.cos(end - start))
    return float(start + fraction * delta)


@dataclass(frozen=True)
class PredictionSample:
    east: float
    north: float
    heading_rad: float
    route_arc_m: float
    next_index: int


@dataclass(frozen=True)
class PredictionRoute:
    """Read-only prediction arrays prepared once for the real-time loop."""

    times: np.ndarray
    east: np.ndarray
    north: np.ndarray
    headings: np.ndarray
    route_arc_m: np.ndarray

    @classmethod
    def from_frame(cls, prediction: pd.DataFrame, heading_points: int) -> "PredictionRoute":
        arrays = (
            prediction["t_sec"].to_numpy(dtype=np.float64, copy=True),
            prediction["east"].to_numpy(dtype=np.float64, copy=True),
            prediction["north"].to_numpy(dtype=np.float64, copy=True),
            prediction_headings_from_track(prediction, heading_points),
        )
        times, east, north, headings = arrays
        route_arc_m = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(east), np.diff(north))))
        )
        for array in (*arrays, route_arc_m):
            array.setflags(write=False)
        return cls(times, east, north, headings, route_arc_m)

    def sample(self, query_t: float) -> PredictionSample:
        query_t = float(np.clip(query_t, self.times[0], self.times[-1]))
        right = int(np.searchsorted(self.times, query_t, side="right"))
        if right == 0:
            return PredictionSample(
                float(self.east[0]), float(self.north[0]),
                float(self.headings[0]), float(self.route_arc_m[0]), 1,
            )
        if right >= len(self.times):
            last = len(self.times) - 1
            return PredictionSample(
                float(self.east[last]), float(self.north[last]),
                float(self.headings[last]), float(self.route_arc_m[last]),
                len(self.times),
            )

        left = right - 1
        fraction = (query_t - self.times[left]) / max(
            self.times[right] - self.times[left], 1e-9
        )
        gap_m = float(self.route_arc_m[right] - self.route_arc_m[left])
        if gap_m > 2.0:
            index = left if fraction <= 0.5 else right
            return PredictionSample(
                float(self.east[index]), float(self.north[index]),
                float(self.headings[index]), float(self.route_arc_m[index]),
                index + 1,
            )

        return PredictionSample(
            float(self.east[left] + fraction * (self.east[right] - self.east[left])),
            float(self.north[left] + fraction * (self.north[right] - self.north[left])),
            circular_interpolate(
                float(self.headings[left]), float(self.headings[right]), fraction
            ),
            float(
                self.route_arc_m[left]
                + fraction * (self.route_arc_m[right] - self.route_arc_m[left])
            ),
            right,
        )

    def forward_window(
        self,
        sample: PredictionSample,
        distance_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return views covering only the route immediately ahead."""
        start = min(sample.next_index, len(self.times))
        stop = int(
            np.searchsorted(
                self.route_arc_m,
                sample.route_arc_m + float(distance_m),
                side="right",
            )
        )
        stop = min(len(self.times), max(start + 1, stop))
        return self.east[start:stop], self.north[start:stop], self.headings[start:stop]


def _smooth_coordinates(values: np.ndarray, point_window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return values.copy()
    window = min(len(values), max(3, int(point_window)))
    padded = np.pad(values, ((window - 1) // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    smoothed[0] = values[0]
    return smoothed


def prediction_path_in_body(
    route: PredictionRoute,
    history_end: tuple[float, float],
    sample: PredictionSample,
    lookahead_m: float,
    sample_spacing_m: float,
    heading_offset_deg: float,
    heading_points: int = 20,
    yaw_heading_rad: float | None = None,
    history_heading_rad: float | None = None,
    connection_progress: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    predicted_east = sample.east
    predicted_north = sample.north
    prediction_heading = sample.heading_rad
    smoothing_margin_m = max(5.0, 0.25 * float(lookahead_m))
    future_east, future_north, future_headings = route.forward_window(
        sample,
        float(lookahead_m) + smoothing_margin_m,
    )
    current_heading = prediction_heading if yaw_heading_rad is None else float(yaw_heading_rad)
    connection_progress = float(np.clip(connection_progress, 0.0, 1.0))
    if connection_progress < 1.0:
        history_east, history_north = history_end
        anchor_east = (1.0 - connection_progress) * history_east + connection_progress * predicted_east
        anchor_north = (1.0 - connection_progress) * history_north + connection_progress * predicted_north
        joined_heading = current_heading if history_heading_rad is None else float(history_heading_rad)
        east = np.concatenate(([anchor_east, predicted_east], future_east))
        north = np.concatenate(([anchor_north, predicted_north], future_north))
        headings = np.concatenate(
            ([joined_heading, prediction_heading], future_headings)
        )
    else:
        anchor_east, anchor_north = predicted_east, predicted_north
        east = np.concatenate(([anchor_east], future_east))
        north = np.concatenate(([anchor_north], future_north))
        headings = np.concatenate(([prediction_heading], future_headings))
    discontinuity = np.flatnonzero(np.hypot(np.diff(east), np.diff(north)) > 2.0)
    if len(discontinuity):
        stop = int(discontinuity[0]) + 1
        east, north, headings = east[:stop], north[:stop], headings[:stop]
    if len(east) < 2:
        east = np.append(east, east[-1] + sample_spacing_m * math.sin(prediction_heading))
        north = np.append(north, north[-1] + sample_spacing_m * math.cos(prediction_heading))
        headings = np.append(headings, prediction_heading)
    east = _smooth_coordinates(east, heading_points)
    north = _smooth_coordinates(north, heading_points)
    yaw = current_heading + math.radians(heading_offset_deg)
    forward_distance = (east - anchor_east) * math.sin(yaw) + (north - anchor_north) * math.cos(yaw)
    peak = int(np.argmax(forward_distance))
    east, north, headings = east[:peak + 1], north[:peak + 1], headings[:peak + 1]
    if len(east) < 2:
        east = np.array([anchor_east, anchor_east + sample_spacing_m * math.sin(prediction_heading)])
        north = np.array([anchor_north, anchor_north + sample_spacing_m * math.cos(prediction_heading)])
        headings = np.array([prediction_heading, prediction_heading])
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
    sampled_headings = np.interp(sample_arc, arc, np.unwrap(headings))
    de = sampled_east - anchor_east
    dn = sampled_north - anchor_north
    points = np.column_stack(
        (de * math.sin(yaw) + dn * math.cos(yaw), de * math.cos(yaw) - dn * math.sin(yaw))
    )
    local_headings = sampled_headings - yaw
    forward = points[:, 0] >= -0.05
    if np.count_nonzero(forward) < 2:
        return None
    points, local_headings = points[forward], local_headings[forward]
    points[0] = [0.0, 0.0]
    actual_arc = float(np.sum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))))
    if actual_arc < lookahead_m - 1e-3:
        remaining = lookahead_m - actual_arc
        count = max(1, int(math.ceil(remaining / sample_spacing_m)))
        distance = np.minimum(np.arange(1, count + 1) * sample_spacing_m, remaining)
        direction = math.atan2(math.sin(float(local_headings[-1])), math.cos(float(local_headings[-1])))
        points = np.vstack(
            (
                points,
                np.column_stack(
                    (points[-1, 0] + distance * math.cos(direction), points[-1, 1] + distance * math.sin(direction))
                ),
            )
        )
        local_headings = np.append(
            local_headings,
            np.full(len(distance), local_headings[-1], dtype=np.float64),
        )

    # Normalize the final output so downstream rendering always receives the
    # same number of points for a fixed look-ahead and sample spacing.
    final_arc = np.concatenate(
        ([0.0], np.cumsum(np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))))
    )
    target_count = max(2, int(round(float(lookahead_m) / sample_spacing_m)) + 1)
    target_arc = np.linspace(0.0, float(lookahead_m), target_count)
    points = np.column_stack(
        [np.interp(target_arc, final_arc, points[:, axis]) for axis in range(2)]
    )
    local_headings = np.interp(target_arc, final_arc, np.unwrap(local_headings))
    points[0] = [0.0, 0.0]
    return points.astype(np.float32), local_headings.astype(np.float64)


class NavigationPathSmoother:
    def __init__(self, temporal_alpha: float = 0.90, spatial_window: int = 11):
        self.temporal_alpha = float(temporal_alpha)
        self.spatial_window = max(3, int(spatial_window) | 1)
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def _spatial(self, path: np.ndarray) -> np.ndarray:
        if len(path) < 3:
            return path.astype(np.float32, copy=True)
        kernel = np.convolve(
            np.ones(self.spatial_window), np.ones(self.spatial_window)
        )
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
            distance = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(current[:, 0]), np.diff(current[:, 1])))))
            if distance[-1] > 1e-6:
                previous = self._resample(self.previous, distance / distance[-1])
                current = self.temporal_alpha * previous + (1.0 - self.temporal_alpha) * current
        current[0] = [0.0, 0.0]
        self.previous = current.astype(np.float32)
        return self.previous.copy()


class GnssProjectionEngine:
    """Build a GNSS navigation centerline in video pixel coordinates."""

    def __init__(
        self,
        *,
        history_path: str,
        prediction_path: str,
        calibration_path: str,
        width: int,
        height: int,
        lidar_height_m: float,
        lookahead_m: float,
        sample_spacing_m: float,
        projection_scale: float,
        update_hz: float = 5.0,
        prediction_start_s: float | None = None,
        switch_smooth_s: float = 1.0,
        history_heading_points: int = 10,
        heading_points: int = 20,
        smooth_alpha: float = 0.90,
        smooth_window: int = 11,
        heading_offset_deg: float = 0.0,
        time_offset_s: float = 0.0,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("投映分辨率必须大于 0")
        if lookahead_m <= 0 or sample_spacing_m <= 0:
            raise ValueError("导航距离和采样间隔必须大于 0")
        if not 0.0 < projection_scale <= 1.0:
            raise ValueError("nav-scale 必须在 (0, 1] 范围内")

        if not math.isfinite(update_hz) or update_hz <= 0:
            raise ValueError("GNSS projection update rate must be greater than 0")

        self.width = int(width)
        self.height = int(height)
        self.lookahead_m = float(lookahead_m)
        self.sample_spacing_m = float(sample_spacing_m)
        self.projection_scale = float(projection_scale)
        self.update_interval_s = 1.0 / float(update_hz)
        self.switch_smooth_s = float(switch_smooth_s)
        self.heading_points = int(heading_points)
        self.heading_offset_deg = float(heading_offset_deg)
        self.time_offset_s = float(time_offset_s)

        history = load_position_track(history_path, "GNSS history")
        origin_lat = float(history.at[0, "lat"])
        origin_lon = float(history.at[0, "lon"])
        prediction = load_position_track(
            prediction_path,
            "GNSS prediction",
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )
        self.prediction_start_s = (
            float(prediction["t_sec"].iloc[0])
            if prediction_start_s is None
            else float(prediction_start_s)
        )
        self.history_end = (
            float(history["east"].iloc[-1]),
            float(history["north"].iloc[-1]),
        )

        history_heading_deg = estimate_heading_from_history(
            history,
            self.prediction_start_s,
            history_heading_points,
        )
        self.history_heading_rad = math.radians(history_heading_deg)
        self.prediction_route = PredictionRoute.from_frame(
            prediction,
            self.heading_points,
        )

        self.projector = CalibratedGroundProjector(
            calibration_path,
            self.width,
            self.height,
            ground_z=-abs(float(lidar_height_m)),
        )
        # Preserve roughly the same smoothing time constant that the old
        # per-frame (30 FPS) projection had after lowering the update rate.
        effective_smooth_alpha = float(smooth_alpha) ** (30.0 / float(update_hz))
        self.path_smoother = NavigationPathSmoother(
            temporal_alpha=effective_smooth_alpha,
            spatial_window=smooth_window,
        )
        self._previous_mode: str | None = None
        self._previous_curve = np.empty((0, 2), dtype=np.int32)
        self._last_projection_query_time: float | None = None

        print(
            "GNSS 投映: "
            f"前视 {self.lookahead_m:.1f}m, "
            f"画面等比缩放 {self.projection_scale:.3f}, "
            f"预测切换 {self.prediction_start_s:.3f}s"
        )

    def _build_body_path(self, query_time_s: float) -> np.ndarray | None:
        mode = "prediction" if query_time_s >= self.prediction_start_s else "straight"
        if mode != self._previous_mode:
            self.path_smoother.reset()
            self._previous_mode = mode

        if mode == "straight":
            return straight_path_in_body(self.lookahead_m, self.sample_spacing_m)

        transition = (
            1.0
            if self.switch_smooth_s <= 0
            else float(
                np.clip(
                    (query_time_s - self.prediction_start_s) / self.switch_smooth_s,
                    0.0,
                    1.0,
                )
            )
        )
        sample = self.prediction_route.sample(query_time_s)
        connected_heading = circular_interpolate(
            self.history_heading_rad,
            sample.heading_rad,
            transition,
        )
        result = prediction_path_in_body(
            self.prediction_route,
            self.history_end,
            sample,
            self.lookahead_m,
            self.sample_spacing_m,
            self.heading_offset_deg,
            self.heading_points,
            yaw_heading_rad=connected_heading,
            history_heading_rad=self.history_heading_rad,
            connection_progress=transition,
        )
        if result is None:
            return None

        body_path, _ = result
        if self.switch_smooth_s > 0:
            straight = np.column_stack(
                (
                    np.linspace(0.0, self.lookahead_m, len(body_path)),
                    np.zeros(len(body_path)),
                )
            )
            body_path = (
                (1.0 - transition) * straight + transition * body_path
            ).astype(np.float32)
        return body_path

    def _scale_about_bottom_center(self, curve: np.ndarray) -> np.ndarray:
        if len(curve) == 0 or self.projection_scale == 1.0:
            return curve
        anchor = np.array(
            [self.width / 2.0, self.height - 1.0],
            dtype=np.float64,
        )
        scaled = anchor + (curve.astype(np.float64) - anchor) * self.projection_scale
        return np.rint(scaled).astype(np.int32)

    def project(self, video_time_s: float) -> np.ndarray:
        """Return the current navigation centerline in full-frame pixels."""
        query_time_s = float(video_time_s) + self.time_offset_s
        if self._last_projection_query_time is not None:
            elapsed_s = query_time_s - self._last_projection_query_time
            still_on_static_approach = (
                query_time_s < self.prediction_start_s
                and self._last_projection_query_time < self.prediction_start_s
            )
            crossed_prediction_start = (
                self._last_projection_query_time < self.prediction_start_s
                <= query_time_s
            )
            if still_on_static_approach or (
                not crossed_prediction_start
                and 0.0 <= elapsed_s
                and elapsed_s + 1e-9 < self.update_interval_s
            ):
                return self._previous_curve.copy()
        self._last_projection_query_time = query_time_s

        body_path = self._build_body_path(query_time_s)
        if body_path is None:
            self.path_smoother.reset()
            return self._previous_curve.copy()

        body_path = self.path_smoother.update(body_path)
        if body_path is None:
            return self._previous_curve.copy()

        center_px = _project_centerline(self.projector, body_path)
        if len(center_px) < 2:
            return self._previous_curve.copy()

        center_px = self._scale_about_bottom_center(center_px)
        self._previous_curve = center_px.copy()
        return center_px
