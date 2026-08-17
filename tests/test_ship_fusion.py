import ast
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = MagicMock()

from data_server import NavigationDataServer, prepare_frame_data
from utils.geometry import merge_ship_data


WINDOWS_SCRIPT = Path(__file__).resolve().parents[1] / "ar_navigation_video1.py"


def load_windows_fusion_functions():
    tree = ast.parse(WINDOWS_SCRIPT.read_text(encoding="utf-8"))
    function_names = {
        "_ship_center",
        "_fusion_values",
        "_optional_float",
        "merge_ship_data",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    namespace = {"math": math}
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(WINDOWS_SCRIPT), "exec"), namespace)

    receiver = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FusionReceiver"
    )
    convert = next(
        node
        for node in receiver.body
        if isinstance(node, ast.FunctionDef) and node.name == "_convert"
    )
    convert_module = ast.fix_missing_locations(
        ast.Module(body=[convert], type_ignores=[])
    )
    exec(compile(convert_module, str(WINDOWS_SCRIPT), "exec"), namespace)
    return namespace


def yolo_ship(center_x):
    return {
        "ship_id": 0,
        "label": "ship",
        "bbox": [center_x - 10, 40, center_x + 10, 60],
        "center": [center_x, 50],
        "conf": 0.9,
    }


def fusion_ship(center_x, north_vel):
    return {
        "bbox": [center_x - 8, 42, center_x + 8, 58],
        "center": [center_x, 50],
        "north_vel": north_vel,
        "east_vel": north_vel + 1.0,
        "distance": north_vel + 10.0,
        "yaw": north_vel + 20.0,
    }


class ShipFusionMatchingTests(unittest.TestCase):
    def test_matches_fusion_targets_one_to_one_by_closest_center(self):
        yolo_ships = [yolo_ship(100), yolo_ship(130)]
        external_ships = [fusion_ship(112, 1.0), fusion_ship(132, 2.0)]

        merged = merge_ship_data(yolo_ships, external_ships)

        self.assertEqual(merged[0]["north_vel"], 1.0)
        self.assertEqual(merged[1]["north_vel"], 2.0)
        self.assertTrue(all(ship["has_fusion_data"] for ship in merged))

    def test_rejects_distant_target_and_clears_stale_fusion_fields(self):
        ship = yolo_ship(100)
        ship.update({"north_vel": 9.0, "has_fusion_data": True})

        merged = merge_ship_data([ship], [fusion_ship(181, 1.0)])

        self.assertFalse(merged[0]["has_fusion_data"])
        self.assertEqual(merged[0]["north_vel"], 0.0)

    def test_one_fusion_target_is_not_reused_for_two_yolo_boxes(self):
        merged = merge_ship_data(
            [yolo_ship(100), yolo_ship(130)],
            [fusion_ship(118, 3.0)],
        )

        self.assertEqual(sum(ship["has_fusion_data"] for ship in merged), 1)
        self.assertFalse(merged[0]["has_fusion_data"])
        self.assertTrue(merged[1]["has_fusion_data"])

    def test_invalid_fusion_records_are_ignored_without_raising(self):
        invalid_records = [
            None,
            {},
            {"center": [100, 50], "north_vel": 1.0},
            {
                "center": [100, 50],
                "north_vel": float("nan"),
                "east_vel": 1.0,
                "distance": 10.0,
                "yaw": 20.0,
            },
        ]

        merged = merge_ship_data([yolo_ship(100)], invalid_records)

        self.assertFalse(merged[0]["has_fusion_data"])
        self.assertEqual(merged[0]["north_vel"], 0.0)

    def test_invalid_fusion_container_is_treated_as_no_data(self):
        merged = merge_ship_data([yolo_ship(100)], 123)

        self.assertFalse(merged[0]["has_fusion_data"])


class ShipFusionSerializationTests(unittest.TestCase):
    def setUp(self):
        self.ship = yolo_ship(100)
        self.ship.update(
            {
                "north_vel": 1.25,
                "east_vel": -2.5,
                "distance": 31.75,
                "yaw": 87.5,
                "has_fusion_data": True,
            }
        )

    def assert_fusion_fields(self, ship_message):
        self.assertEqual(ship_message["north_vel"], 1.25)
        self.assertEqual(ship_message["east_vel"], -2.5)
        self.assertEqual(ship_message["distance"], 31.75)
        self.assertEqual(ship_message["yaw"], 87.5)
        self.assertTrue(ship_message["has_fusion_data"])

    def test_prepare_frame_data_preserves_fusion_fields(self):
        message = prepare_frame_data(1920, 1080, [], 1, 10.0, ships_data=[self.ship])
        self.assert_fusion_fields(message["ships"][0])

    def test_send_nav_data_preserves_fusion_fields(self):
        server = NavigationDataServer.__new__(NavigationDataServer)
        captured = {}

        def capture(message, frame_id):
            captured["message"] = message
            captured["frame_id"] = frame_id

        server.send_prepared_nav = capture
        server.get_ships = lambda: []
        server.send_nav_data(1, 1920, 1080, ships=[self.ship])

        self.assertEqual(captured["frame_id"], 1)
        self.assert_fusion_fields(captured["message"]["ships"][0])


class WindowsStandaloneFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_windows_fusion_functions()

    def test_windows_matcher_has_same_one_to_one_and_empty_data_behavior(self):
        matcher = self.functions["merge_ship_data"]
        merged = matcher(
            [yolo_ship(100), yolo_ship(130)],
            [fusion_ship(118, 3.0)],
        )

        self.assertEqual(sum(ship["has_fusion_data"] for ship in merged), 1)
        self.assertTrue(merged[1]["has_fusion_data"])
        self.assertFalse(matcher([yolo_ship(100)], 123)[0]["has_fusion_data"])

    def test_windows_receiver_requires_and_preserves_yaw(self):
        convert = self.functions["_convert"]
        obstacle = {
            "pixel_x1": 90,
            "pixel_y1": 40,
            "pixel_x2": 110,
            "pixel_y2": 60,
            "north_vel": 1.25,
            "east_vel": -2.5,
            "distance": 31.75,
            "yaw": 87.5,
        }

        converted = convert(None, [obstacle])
        self.assertEqual(converted[0]["yaw"], 87.5)

        del obstacle["yaw"]
        self.assertEqual(convert(None, [obstacle]), [])


if __name__ == "__main__":
    unittest.main()
