import cv2
import numpy as np


def scale_curve_xy(curve: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if curve is None or len(curve) == 0:
        return np.empty((0, 2), dtype=np.int32)

    out = curve.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy

    return out.astype(np.int32)


def scale_ships_for_display(ships: list, sx: float, sy: float) -> list:
    if not ships or (sx == 1.0 and sy == 1.0):
        return ships

    result = []

    for s in ships:
        ss = dict(s)

        x1, y1, x2, y2 = s["bbox"]
        ss["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        ss["center"] = [s["center"][0] * sx, s["center"][1] * sy]

        result.append(ss)

    return result


def scale_ships_for_send(
    ships: list,
    sx: float,
    sy: float,
    dst_w: int = None,
    dst_h: int = None,
) -> list:
    if not ships:
        return []

    result = []

    for s in ships:
        if "bbox" not in s or s["bbox"] is None or len(s["bbox"]) < 4:
            continue

        ss = dict(s)

        x1, y1, x2, y2 = [float(v) for v in s["bbox"]]

        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

        if dst_w is not None:
            x1 = float(np.clip(x1, 0, dst_w - 1))
            x2 = float(np.clip(x2, 0, dst_w - 1))

        if dst_h is not None:
            y1 = float(np.clip(y1, 0, dst_h - 1))
            y2 = float(np.clip(y2, 0, dst_h - 1))

        ss["bbox"] = [x1, y1, x2, y2]

        if "center" in s and s["center"] is not None and len(s["center"]) >= 2:
            cx = float(s["center"][0]) * sx
            cy = float(s["center"][1]) * sy

            if dst_w is not None:
                cx = float(np.clip(cx, 0, dst_w - 1))

            if dst_h is not None:
                cy = float(np.clip(cy, 0, dst_h - 1))

            ss["center"] = [cx, cy]
        else:
            ss["center"] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        result.append(ss)

    return result


def extract_polygons_json(
    water_mask,
    bridge_mask,
    epsilon_ratio=0.005,
    min_area_px=200,
    max_polys=4,
):
    h, w = water_mask.shape
    epsilon = max(1.0, epsilon_ratio * max(w, h))

    def _extract(mask):
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        polys = []

        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:max_polys]:
            if cv2.contourArea(cnt) < min_area_px:
                continue

            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

            pts = [[int(p[0]), int(p[1])] for p in approx.reshape(-1, 2)]

            if len(pts) >= 3:
                polys.append(pts)

        return polys

    return _extract(water_mask), _extract(bridge_mask)


def should_draw_nav_band(ships, external_ships, max_distance):
    distances = []

    for s in ships:
        d = float(s.get("distance", -1))
        if d > 0:
            distances.append(d)

    for s in external_ships:
        d = float(s.get("bridge_pier_distance", -1))
        if d > 0:
            distances.append(d)

    if not distances:
        return True

    return min(distances) <= max_distance


def merge_ship_data(yolo_ships, external_ships):
    if not yolo_ships:
        return []

    if not external_ships:
        return yolo_ships

    for ys in yolo_ships:
        cx, cy = ys["center"]
        best_ext = None
        best_d = 1e9

        for ext in external_ships:
            if "center" in ext:
                ex, ey = ext["center"]
            elif "bbox" in ext:
                bx1, by1, bx2, by2 = ext["bbox"]
                ex, ey = (bx1 + bx2) / 2, (by1 + by2) / 2
            else:
                continue

            d = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5

            if d < best_d:
                best_d = d
                best_ext = ext

        if best_ext is not None and best_d < 80:
            ys["speed"] = float(best_ext.get("speed", 0.0))
            ys["bearing"] = float(best_ext.get("bearing", 0.0))
            ys["distance"] = float(best_ext.get("distance", 0.0))
            ys["bridge_pier_distance"] = float(
                best_ext.get("bridge_pier_distance", -1.0)
            )
            ys["hasSpeedBearing"] = True
        else:
            ys["hasSpeedBearing"] = False

    return yolo_ships
