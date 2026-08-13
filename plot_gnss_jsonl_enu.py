from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from openpyxl import Workbook

from plot_gnss_enu import is_number, make_enu_converter, plot_enu


DATA_FIELDS = (
    "ts",
    "lon",
    "lat",
    "alt",
    "quality",
    "satellites",
    "hdop",
    "speed",
    "course",
    "heading",
    "pitch",
    "roll",
)


def load_jsonl(input_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                data = message["payload"]["data"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid GNSS JSON at line {line_number}: {exc}") from exc

            lon = data.get("lon")
            lat = data.get("lat")
            alt = data.get("alt", 0.0)
            if not all(is_number(value) for value in (lon, lat, alt)):
                raise ValueError(f"Invalid lon/lat/alt at line {line_number}")
            records.append(
                {
                    "line_number": line_number,
                    "recv_ts": message.get("recv_ts"),
                    "topic": message.get("topic"),
                    "payload_ts": message.get("payload", {}).get("ts"),
                    **{field: data.get(field) for field in DATA_FIELDS},
                }
            )
    if not records:
        raise ValueError("No valid GNSS records found")
    return records


def convert_jsonl(input_path: Path, output_xlsx: Path):
    records = load_jsonl(input_path)
    first = records[0]
    origin = (float(first["lon"]), float(first["lat"]), float(first["alt"]))
    to_enu = make_enu_converter(*origin)
    valid_enu: list[tuple[float, float, float]] = []
    east: list[float] = []
    north: list[float] = []

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "ENU相对坐标"
    source_fields = ("line_number", "recv_ts", "topic", "payload_ts", *DATA_FIELDS)
    sheet.append([*source_fields, "East (m)", "North (m)", "Up (m)"])

    previous_ts: float | None = None
    for record in records:
        enu = to_enu(float(record["lon"]), float(record["lat"]), float(record["alt"]))
        valid_enu.append(enu)
        current_ts = record.get("ts")
        if (
            previous_ts is not None
            and is_number(current_ts)
            and (float(current_ts) <= previous_ts or float(current_ts) - previous_ts > 5000.0)
        ):
            east.append(math.nan)
            north.append(math.nan)
        east.append(enu[0])
        north.append(enu[1])
        if is_number(current_ts):
            previous_ts = float(current_ts)
        sheet.append([*(record[field] for field in source_fields), *enu])

    sheet.freeze_panes = "A2"
    for column in sheet.iter_cols(min_col=len(source_fields) + 1, max_col=len(source_fields) + 3):
        for cell in column[1:]:
            cell.number_format = "0.0000"
    workbook.save(output_xlsx)
    return origin, valid_enu, east, north


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert nested GNSS JSONL data to local WGS84 ENU coordinates")
    parser.add_argument("input", type=Path, nargs="?", default=Path("gnss_gnss_01.jsonl"))
    parser.add_argument("--png", type=Path, default=Path("gnss_gnss_01_enu_2d.png"))
    parser.add_argument("--xlsx", type=Path, default=Path("gnss_gnss_01_enu.xlsx"))
    args = parser.parse_args()

    origin, valid_enu, east, north = convert_jsonl(args.input, args.xlsx)
    plot_enu(args.png, origin, valid_enu, east, north)
    print(f"Origin (lon, lat, h): {origin}")
    print(f"Valid points: {len(valid_enu)}")
    print(f"East range: {min(p[0] for p in valid_enu):.4f} .. {max(p[0] for p in valid_enu):.4f} m")
    print(f"North range: {min(p[1] for p in valid_enu):.4f} .. {max(p[1] for p in valid_enu):.4f} m")
    print(f"Up range: {min(p[2] for p in valid_enu):.4f} .. {max(p[2] for p in valid_enu):.4f} m")
    print(f"PNG: {args.png.resolve()}")
    print(f"XLSX: {args.xlsx.resolve()}")


if __name__ == "__main__":
    main()
