from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

from utils.realtime_navigation import RealtimeNavigationState

try:
    import paho.mqtt.client as mqtt
except ImportError:  # Kept optional so the pure navigation logic remains testable.
    mqtt = None


DEFAULT_PREDICTION_TOPIC = "v1/11/prediction/result"
DEFAULT_GNSS_TOPIC = "v1/11/sensor/gnss/gnss_01"


@dataclass(frozen=True)
class MqttNavigationConfig:
    host: str = "127.0.0.1"
    prediction_topic: str = DEFAULT_PREDICTION_TOPIC
    gnss_topic: str = DEFAULT_GNSS_TOPIC
    client_prefix: str = "nav_visualizer"
    username: str | None = None
    password: str | None = None
    keepalive_s: int = 30


def _mqtt_client(client_id: str):
    if mqtt is None:
        raise RuntimeError(
            "paho-mqtt is required for realtime navigation; "
            "install paho-mqtt>=2.0"
        )
    callback_api = getattr(mqtt, "CallbackAPIVersion", None)
    if callback_api is not None:
        return mqtt.Client(
            callback_api.VERSION1,
            client_id=client_id,
            protocol=mqtt.MQTTv5,
        )
    return mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)


def _reason_code_value(reason_code) -> int:
    return int(getattr(reason_code, "value", reason_code))


def _normalise_prediction(data: Mapping) -> dict:
    """Convert common avoidance-result shapes into the internal mission shape."""
    candidate = data
    points = None
    for key in ("result", "prediction"):
        nested = candidate.get(key)
        if isinstance(nested, Mapping):
            candidate = nested
            break
        if isinstance(nested, list):
            points = nested
            break

    if points is None:
        points = candidate.get("points")
    if not isinstance(points, list):
        for key in ("path", "trajectory", "route", "waypoints"):
            value = candidate.get(key)
            if isinstance(value, list):
                points = value
                break
    if not isinstance(points, list) or not points:
        raise ValueError("Prediction result needs a non-empty path array")

    normalised_points = []
    for index, point in enumerate(points):
        if isinstance(point, Mapping):
            latitude = point.get("latitude", point.get("lat"))
            longitude = point.get("longitude", point.get("lon", point.get("lng")))
            tolerance = point.get("tolerance")
        elif (
            isinstance(point, Sequence)
            and not isinstance(point, (str, bytes, bytearray))
            and len(point) >= 2
        ):
            latitude, longitude = point[0], point[1]
            tolerance = None
        else:
            raise ValueError(f"Prediction point {index} has an unsupported shape")
        item = {"latitude": latitude, "longitude": longitude}
        if tolerance is not None:
            item["tolerance"] = tolerance
        normalised_points.append(item)

    return {
        "count": candidate.get("count", 1),
        "tolerance": candidate.get("tolerance", 5.0),
        "points": normalised_points,
    }


class MqttNavigationClient:
    def __init__(
        self,
        state: RealtimeNavigationState,
        config: MqttNavigationConfig,
        *,
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mqtt is None:
            raise RuntimeError(
                "paho-mqtt is required for realtime navigation; "
                "install paho-mqtt>=2.0"
            )
        if not config.host.strip():
            raise ValueError("MQTT host cannot be empty")
        if not config.prediction_topic.strip() or not config.gnss_topic.strip():
            raise ValueError("MQTT topics cannot be empty")
        self.state = state
        self.config = config
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._client = None
        self._gnss_received_count = 0

    def start(self) -> None:
        suffix = uuid.uuid4().hex[:8]
        client = _mqtt_client(f"{self.config.client_prefix}_{suffix}")
        if self.config.username:
            client.username_pw_set(self.config.username, self.config.password)
        client.reconnect_delay_set(min_delay=1, max_delay=10)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        self._client = client
        client.connect_async(self.config.host, keepalive=self.config.keepalive_s)
        client.loop_start()
        print(
            f"[MQTT Nav] starting broker={self.config.host}; "
            f"prediction={self.config.prediction_topic}; GNSS={self.config.gnss_topic}"
        )

    def stop(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.disconnect()
        except Exception as exc:
            print(f"[MQTT Nav] disconnect warning: {exc}")
        finally:
            client.loop_stop()
        self._client = None

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties=None) -> None:
        if _reason_code_value(reason_code) != 0:
            print(f"[MQTT Nav] connection rejected: {reason_code}")
            return
        client.subscribe(self.config.prediction_topic, qos=1)
        client.subscribe(self.config.gnss_topic, qos=0)
        print(
            f"[MQTT Nav] subscribed {self.config.prediction_topic} and "
            f"{self.config.gnss_topic}"
        )

    @staticmethod
    def _on_disconnect(_client, _userdata, reason_code, _properties=None) -> None:
        if _reason_code_value(reason_code) != 0:
            print(f"[MQTT Nav] disconnected unexpectedly: {reason_code}")

    @staticmethod
    def _decode_message(message) -> dict:
        try:
            envelope = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid MQTT JSON on {message.topic}: {exc}") from exc
        if not isinstance(envelope, dict) or not isinstance(envelope.get("data"), dict):
            raise ValueError(f"MQTT message on {message.topic} has no data object")
        return envelope

    def _on_message(self, _client, _userdata, message) -> None:
        try:
            envelope = self._decode_message(message)
            if message.topic == self.config.prediction_topic:
                self._handle_prediction(envelope)
            elif message.topic == self.config.gnss_topic:
                self._handle_gnss(envelope)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            print(f"[MQTT Nav] ignored {message.topic}: {exc}")

    def _handle_prediction(self, envelope: Mapping) -> None:
        data = envelope["data"]
        timestamp = float(data.get("sample_ts", envelope["ts"]))
        if not math.isfinite(timestamp) or timestamp <= 0.0:
            raise ValueError("Prediction timestamp is invalid")
        mission = _normalise_prediction(data)
        route = self.state.activate_mission(
            mission,
            timestamp,
            now_epoch=self._clock(),
            now_monotonic=self._monotonic_clock(),
        )
        print(
            f"[MQTT Nav] activated avoidance path: "
            f"{len(route.mission_east_m)} points, {route.length_m:.2f}m"
        )

    def _handle_gnss(self, envelope: Mapping) -> None:
        data = envelope["data"]
        timestamp = float(data.get("sample_ts", envelope["ts"]))
        latitude = float(data["lat"])
        longitude = float(data["lon"])
        heading_value = data.get("heading")
        heading = None if heading_value is None else float(heading_value)
        accepted = self.state.add_gnss(
            timestamp,
            latitude,
            longitude,
            heading,
            received_at=self._clock(),
        )
        if not accepted:
            print(f"[MQTT Nav] rejected GNSS sample ts={timestamp:.3f}")
            return
        self._gnss_received_count += 1
        if self._gnss_received_count % 50 == 0:
            print(
                f"[MQTT Nav] received GNSS={self._gnss_received_count} "
                f"latest_ts={timestamp:.3f}",
                flush=True,
            )
