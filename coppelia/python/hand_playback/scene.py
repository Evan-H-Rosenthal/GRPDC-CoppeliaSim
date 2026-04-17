from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recording import HandFrame
from .transforms import (
    align_quaternion_hemisphere,
    ThumbCalibration,
    lerp_vector,
    normalize_quaternion,
    quat_to_single_axis_joint,
    remap_quaternion,
    remap_vector,
    thumb_angles_from_metacarpal,
)


@dataclass(frozen=True)
class JointHandles:
    pointer_prox: int | None = None
    pointer_mid: int | None = None
    pointer_dist: int | None = None
    middle_prox: int | None = None
    middle_mid: int | None = None
    middle_dist: int | None = None
    ring_prox: int | None = None
    ring_mid: int | None = None
    ring_dist: int | None = None
    thumb_meta: int | None = None
    thumb_base: int | None = None
    thumb_prox: int | None = None
    thumb_dist: int | None = None


@dataclass(frozen=True)
class SceneHandles:
    root_object: int
    moving_dummy: int | None
    kuka_target: int | None
    joints: JointHandles


class SceneController:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._client = None
        self.sim = None
        self.handles: SceneHandles | None = None
        self.filtered_position: list[float] | None = None
        self.filtered_quaternion: list[float] | None = None
        self.thumb_calibration = ThumbCalibration()

    def connect(self) -> None:
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'coppeliasim-zmqremoteapi-client'. "
                "Install it with 'python -m pip install -r coppelia/requirements.txt'."
            ) from exc

        remote_api = self.config["remoteApi"]
        print(
            f"Connecting to CoppeliaSim ZeroMQ Remote API at "
            f"{remote_api.get('host', 'localhost')}:{int(remote_api.get('port', 23001))}..."
        )
        self._client = RemoteAPIClient(
            host=remote_api.get("host", "localhost"),
            port=int(remote_api.get("port", 23001)),
        )
        self.sim = self._client.require("sim")
        self.handles = self._resolve_handles()
        print("Connected and resolved scene handles.")

    def _resolve_handles(self) -> SceneHandles:
        if self.sim is None:
            raise RuntimeError("Remote API is not connected.")

        aliases = self.config["sceneObjects"]
        return SceneHandles(
            root_object=self._get_object(aliases["rootObject"]),
            moving_dummy=self._get_optional_object(aliases.get("movingDummy")),
            kuka_target=self._get_optional_object(aliases.get("kukaTarget")),
            joints=JointHandles(
                pointer_prox=self._get_optional_object(aliases.get("pointerProx")),
                pointer_mid=self._get_optional_object(aliases.get("pointerMid")),
                pointer_dist=self._get_optional_object(aliases.get("pointerDist")),
                middle_prox=self._get_optional_object(aliases.get("middleProx")),
                middle_mid=self._get_optional_object(aliases.get("middleMid")),
                middle_dist=self._get_optional_object(aliases.get("middleDist")),
                ring_prox=self._get_optional_object(aliases.get("ringProx")),
                ring_mid=self._get_optional_object(aliases.get("ringMid")),
                ring_dist=self._get_optional_object(aliases.get("ringDist")),
                thumb_meta=self._get_optional_object(aliases.get("thumbMeta")),
                thumb_base=self._get_optional_object(aliases.get("thumbBase")),
                thumb_prox=self._get_optional_object(aliases.get("thumbProx")),
                thumb_dist=self._get_optional_object(aliases.get("thumbDist")),
            ),
        )

    def _get_object(self, alias: str) -> int:
        if self.sim is None:
            raise RuntimeError("Remote API is not connected.")
        return self.sim.getObject(alias)

    def _get_optional_object(self, alias: str | None) -> int | None:
        if not alias:
            return None
        return self._get_object(alias)

    def apply_frame(self, frame: HandFrame) -> None:
        if self.sim is None or self.handles is None:
            raise RuntimeError("Remote API is not connected.")

        alpha = float(self.config["playback"].get("smoothingAlpha", 0.2))
        transform = self.config.get("transform", {})
        position = remap_vector(
            frame.root_pose.position,
            axes=transform.get("positionAxes", ["z", "x", "y"]),
            signs=transform.get("positionSigns", [1.0, 1.0, 1.0]),
            offset=transform.get("positionOffset", [0.0, 0.0, 0.0]),
        )
        rotation = remap_quaternion(
            frame.root_pose.rotation,
            axes=transform.get("rotationAxes", ["z", "x", "y"]),
            signs=transform.get("rotationSigns", [1.0, 1.0, 1.0]),
        )
        rotation = align_quaternion_hemisphere(self.filtered_quaternion, rotation)

        self.filtered_position = lerp_vector(self.filtered_position, position, alpha)
        self.filtered_quaternion = normalize_quaternion(
            lerp_vector(self.filtered_quaternion, rotation, alpha)
        )

        if self.handles.kuka_target is not None:
            pose_handle = self.handles.kuka_target
        else:
            pose_handle = self.handles.root_object

        self.sim.setObjectPosition(pose_handle, -1, self.filtered_position)
        self.sim.setObjectQuaternion(pose_handle, -1, self.filtered_quaternion)

        joints = self.handles.joints
        rotations = frame.joint_rotations
        self._set_joint_if_present(joints.pointer_prox, rotations, "XRHand_IndexProximal")
        self._set_joint_if_present(joints.pointer_mid, rotations, "XRHand_IndexIntermediate")
        self._set_joint_if_present(joints.pointer_dist, rotations, "XRHand_IndexDistal")
        self._set_joint_if_present(joints.middle_prox, rotations, "XRHand_MiddleProximal")
        self._set_joint_if_present(joints.middle_mid, rotations, "XRHand_MiddleIntermediate")
        self._set_joint_if_present(joints.middle_dist, rotations, "XRHand_MiddleDistal")
        self._set_joint_if_present(joints.ring_prox, rotations, "XRHand_RingProximal")
        self._set_joint_if_present(joints.ring_mid, rotations, "XRHand_RingIntermediate")
        self._set_joint_if_present(joints.ring_dist, rotations, "XRHand_RingDistal")
        self._apply_thumb(joints, rotations)

    def _set_joint_if_present(
        self,
        joint_handle: int | None,
        rotations: dict[str, list[float]],
        key: str,
    ) -> None:
        rotation = rotations.get(key)
        if joint_handle is None or rotation is None or self.sim is None:
            return
        self.sim.setJointPosition(joint_handle, quat_to_single_axis_joint(rotation))

    def _apply_thumb(self, joints: JointHandles, rotations: dict[str, list[float]]) -> None:
        if self.sim is None:
            return

        metacarpal = rotations.get("XRHand_ThumbMetacarpal")
        if metacarpal is not None and joints.thumb_meta is not None and joints.thumb_base is not None:
            meta_angle, base_angle = thumb_angles_from_metacarpal(
                metacarpal,
                self.thumb_calibration,
            )
            self.sim.setJointPosition(joints.thumb_meta, meta_angle)
            self.sim.setJointPosition(joints.thumb_base, base_angle)

        proximal = rotations.get("XRHand_ThumbProximal")
        if proximal is not None and joints.thumb_prox is not None:
            self.sim.setJointPosition(joints.thumb_prox, quat_to_single_axis_joint(proximal))

        distal = rotations.get("XRHand_ThumbDistal")
        if distal is not None and joints.thumb_dist is not None:
            self.sim.setJointPosition(joints.thumb_dist, quat_to_single_axis_joint(distal))

    def playback(
        self,
        frames: list[HandFrame],
        *,
        use_recorded_timing: bool,
        speed: float,
        start_simulation: bool,
        stepping: bool,
    ) -> None:
        if self.sim is None or self._client is None:
            raise RuntimeError("Remote API is not connected.")

        if stepping:
            self._client.setStepping(True)
            print("Stepping mode enabled.")

        started_here = False
        if start_simulation:
            state = self.sim.getSimulationState()
            if state == self.sim.simulation_stopped:
                self.sim.startSimulation()
                started_here = True
                print("Simulation started.")

        previous_timestamp = None
        try:
            for index, frame in enumerate(frames, start=1):
                self.apply_frame(frame)

                if index == 1:
                    print(f"Applied first frame at t={frame.timestamp:.3f}s.")
                elif index % 300 == 0 or index == len(frames):
                    print(f"Applied frame {index}/{len(frames)}.")

                if stepping:
                    self._client.step()
                    continue

                if use_recorded_timing and previous_timestamp is not None:
                    delay = max((frame.timestamp - previous_timestamp) / max(speed, 1e-6), 0.0)
                    if delay > 0:
                        time.sleep(delay)
                previous_timestamp = frame.timestamp
        finally:
            if started_here:
                self.sim.stopSimulation()
                print("Simulation stopped.")


def load_scene_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
