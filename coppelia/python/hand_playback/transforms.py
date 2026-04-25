from __future__ import annotations

import math
from dataclasses import dataclass


AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def remap_vector(
    vector: list[float],
    axes: list[str],
    signs: list[float] | None = None,
    offset: list[float] | None = None,
) -> list[float]:
    if signs is None:
        signs = [1.0, 1.0, 1.0]
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    return [
        signs[i] * vector[AXIS_INDEX[axes[i]]] + offset[i]
        for i in range(3)
    ]


def remap_quaternion(
    rotation: list[float],
    axes: list[str],
    signs: list[float] | None = None,
) -> list[float]:
    if signs is None:
        signs = [1.0, 1.0, 1.0]
    basis = _build_basis_matrix(axes, signs)
    source_rotation = _quaternion_to_matrix(rotation)
    target_rotation = _matmul(_matmul(basis, source_rotation), _transpose(basis))
    return normalize_quaternion(_matrix_to_quaternion(target_rotation))


def _build_basis_matrix(axes: list[str], signs: list[float]) -> list[list[float]]:
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    for row, axis_name in enumerate(axes):
        matrix[row][AXIS_INDEX[axis_name]] = signs[row]
    return matrix


def _quaternion_to_matrix(rotation: list[float]) -> list[list[float]]:
    x, y, z, w = normalize_quaternion(rotation)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _matrix_to_quaternion(matrix: list[list[float]]) -> list[float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2][1] - matrix[1][2]) / s
        y = (matrix[0][2] - matrix[2][0]) / s
        z = (matrix[1][0] - matrix[0][1]) / s
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        s = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        w = (matrix[2][1] - matrix[1][2]) / s
        x = 0.25 * s
        y = (matrix[0][1] + matrix[1][0]) / s
        z = (matrix[0][2] + matrix[2][0]) / s
    elif matrix[1][1] > matrix[2][2]:
        s = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        w = (matrix[0][2] - matrix[2][0]) / s
        x = (matrix[0][1] + matrix[1][0]) / s
        y = 0.25 * s
        z = (matrix[1][2] + matrix[2][1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        w = (matrix[1][0] - matrix[0][1]) / s
        x = (matrix[0][2] + matrix[2][0]) / s
        y = (matrix[1][2] + matrix[2][1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(left[row][k] * right[k][col] for k in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [
        [matrix[col][row] for col in range(3)]
        for row in range(3)
    ]


def normalize_quaternion(quaternion: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in quaternion))
    if magnitude == 0:
        return [0.0, 0.0, 0.0, 1.0]
    return [component / magnitude for component in quaternion]


def align_quaternion_hemisphere(
    previous: list[float] | None,
    current: list[float],
) -> list[float]:
    if previous is None:
        return current
    dot = sum(previous[i] * current[i] for i in range(4))
    if dot < 0.0:
        return [-component for component in current]
    return current


def quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def quaternion_inverse(rotation: list[float]) -> list[float]:
    x, y, z, w = normalize_quaternion(rotation)
    return [-x, -y, -z, w]


def nearest_cube_equivalent_rotation(
    previous: list[float] | None,
    current: list[float],
) -> list[float]:
    current = normalize_quaternion(current)
    if previous is None:
        return current

    best = current
    best_dot = -1.0
    for symmetry in _cube_symmetry_quaternions():
        candidate = normalize_quaternion(quaternion_multiply(current, symmetry))
        candidate = align_quaternion_hemisphere(previous, candidate)
        dot = abs(sum(previous[i] * candidate[i] for i in range(4)))
        if dot > best_dot:
            best_dot = dot
            best = candidate
    return best


_CUBE_SYMMETRY_CACHE: list[list[float]] | None = None


def _cube_symmetry_quaternions() -> list[list[float]]:
    global _CUBE_SYMMETRY_CACHE
    if _CUBE_SYMMETRY_CACHE is not None:
        return _CUBE_SYMMETRY_CACHE

    symmetries: list[list[float]] = []
    permutations = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    for perm in permutations:
        parity = _permutation_parity(perm)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    determinant = parity * sx * sy * sz
                    if determinant < 0.0:
                        continue
                    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
                    matrix[0][perm[0]] = sx
                    matrix[1][perm[1]] = sy
                    matrix[2][perm[2]] = sz
                    quat = normalize_quaternion(_matrix_to_quaternion(matrix))
                    if not any(abs(sum(q[i] * quat[i] for i in range(4))) > 0.9999 for q in symmetries):
                        symmetries.append(quat)

    _CUBE_SYMMETRY_CACHE = symmetries
    return symmetries


def _permutation_parity(perm: tuple[int, int, int]) -> float:
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
    return -1.0 if inversions % 2 else 1.0


def quaternion_to_yaw_only(rotation: list[float]) -> list[float]:
    x, y, z, w = normalize_quaternion(rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    half = yaw * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def lerp_vector(previous: list[float] | None, current: list[float], alpha: float) -> list[float]:
    if previous is None:
        return current
    return [
        previous[index] + alpha * (current[index] - previous[index])
        for index in range(len(current))
    ]


def quat_to_single_axis_joint(rotation: list[float]) -> float:
    qx, _, _, qw = rotation
    return 2.0 * math.atan2(qx, qw)


@dataclass
class ThumbCalibration:
    metacarpal_reference: list[float] | None = None
    meta_baseline: float | None = None
    base_baseline: float | None = None
    chain_reference: list[float] | None = None
    chain_reference_direction: list[float] | None = None
    previous_meta_angle: float | None = None
    previous_base_angle: float | None = None


def thumb_angles_from_metacarpal(
    rotation: list[float],
    calibration: ThumbCalibration,
) -> tuple[float, float]:
    qx, qy, qz, qw = rotation
    sqx = qx * qx
    sqy = qy * qy

    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (sqx + sqy))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))

    if calibration.meta_baseline is None:
        calibration.meta_baseline = pitch
    if calibration.base_baseline is None:
        calibration.base_baseline = roll

    swing = -(pitch - calibration.meta_baseline) * 5.0
    twist = (roll - calibration.base_baseline) * 4.0

    meta_angle = math.radians(15.069) + swing
    base_angle = twist

    meta_angle = min(max(meta_angle, math.radians(15.069)), math.radians(79.985))
    base_angle = min(max(base_angle, math.radians(-45.0)), math.radians(45.0))
    return meta_angle, base_angle


def thumb_angles_from_metacarpal_xz(
    rotation: list[float],
    calibration: ThumbCalibration,
) -> tuple[float, float]:
    if calibration.metacarpal_reference is None:
        calibration.metacarpal_reference = normalize_quaternion(rotation)

    relative_rotation = normalize_quaternion(
        quaternion_multiply(
            quaternion_inverse(calibration.metacarpal_reference),
            rotation,
        )
    )
    matrix = _quaternion_to_matrix(relative_rotation)

    # Decompose the recorded thumb-metacarpal motion in the same serial order as
    # the Allegro thumb root chain: metacarpal first, then base swivel.
    meta_delta = math.atan2(-matrix[1][2], matrix[2][2])
    swivel_delta = math.atan2(matrix[1][0], matrix[0][0])
    rotvec = quaternion_to_rotation_vector(relative_rotation)
    opposition_delta = max(rotvec[1], 0.0)

    # The Allegro thumb needs additional positive base swivel to emulate human
    # thumb opposition. We use the metacarpal's secondary local rotation as an
    # "opposition assist" and blend in a small amount of metacarpal flexion so
    # the thumb can fold inward instead of just hinging across the palm.
    base_delta = (
        swivel_delta
        + 0.85 * opposition_delta
        + 0.25 * max(meta_delta, 0.0)
    )

    meta_angle = math.radians(15.069) + meta_delta
    base_angle = base_delta

    meta_angle = min(max(meta_angle, math.radians(15.069)), math.radians(79.985))
    base_angle = min(max(base_angle, math.radians(-6.0)), math.radians(66.0))
    return meta_angle, base_angle


def quaternion_to_rotation_vector(rotation: list[float]) -> list[float]:
    x, y, z, w = normalize_quaternion(rotation)
    angle = 2.0 * math.acos(max(-1.0, min(1.0, w)))
    sin_half = math.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-6:
        return [2.0 * x, 2.0 * y, 2.0 * z]
    axis = [x / sin_half, y / sin_half, z / sin_half]
    return [axis[i] * angle for i in range(3)]


_THUMB_ROOT_RPY = [0.0, -1.65806278845, -1.5707963259]
_THUMB_META_REST = math.radians(15.069)
_THUMB_META_MIN = math.radians(15.069)
_THUMB_META_MAX = math.radians(79.985)
_THUMB_BASE_MIN = math.radians(-6.0)
_THUMB_BASE_MAX = math.radians(66.0)


def solve_thumb_root_angles(
    metacarpal_rotation: list[float],
    proximal_rotation: list[float] | None,
    proximal_angle: float,
    calibration: ThumbCalibration,
) -> tuple[float, float]:
    combined_rotation = normalize_quaternion(metacarpal_rotation)
    if proximal_rotation is not None:
        combined_rotation = normalize_quaternion(
            quaternion_multiply(combined_rotation, proximal_rotation)
        )

    if calibration.chain_reference is None:
        calibration.chain_reference = combined_rotation
        calibration.chain_reference_direction = _thumb_model_direction(
            _THUMB_META_REST,
            0.0,
            proximal_angle,
        )
        calibration.previous_meta_angle = _THUMB_META_REST
        calibration.previous_base_angle = 0.0

    relative_rotation = normalize_quaternion(
        quaternion_multiply(
            combined_rotation,
            quaternion_inverse(calibration.chain_reference),
        )
    )
    target_direction = _matvec(
        _quaternion_to_matrix(relative_rotation),
        calibration.chain_reference_direction or [0.0, 0.0, 1.0],
    )
    target_direction = _normalize_vector(target_direction)

    initial_meta = (
        calibration.previous_meta_angle
        if calibration.previous_meta_angle is not None
        else _THUMB_META_REST
    )
    initial_base = (
        calibration.previous_base_angle
        if calibration.previous_base_angle is not None
        else 0.0
    )

    # Coarse-to-fine search around the previous solution.
    best_meta = initial_meta
    best_base = initial_base
    best_cost = _thumb_direction_cost(best_meta, best_base, proximal_angle, target_direction)
    for meta_step_deg, base_step_deg in ((18.0, 20.0), (8.0, 10.0), (3.0, 4.0), (1.0, 1.5)):
        meta_step = math.radians(meta_step_deg)
        base_step = math.radians(base_step_deg)
        improved = True
        while improved:
            improved = False
            for meta_dir in (-1.0, 0.0, 1.0):
                for base_dir in (-1.0, 0.0, 1.0):
                    candidate_meta = min(
                        max(best_meta + meta_dir * meta_step, _THUMB_META_MIN),
                        _THUMB_META_MAX,
                    )
                    candidate_base = min(
                        max(best_base + base_dir * base_step, _THUMB_BASE_MIN),
                        _THUMB_BASE_MAX,
                    )
                    cost = _thumb_direction_cost(
                        candidate_meta,
                        candidate_base,
                        proximal_angle,
                        target_direction,
                    )
                    if cost + 1e-8 < best_cost:
                        best_cost = cost
                        best_meta = candidate_meta
                        best_base = candidate_base
                        improved = True

    calibration.previous_meta_angle = best_meta
    calibration.previous_base_angle = best_base
    return best_meta, best_base


def _thumb_direction_cost(
    meta_angle: float,
    base_angle: float,
    proximal_angle: float,
    target_direction: list[float],
) -> float:
    direction = _thumb_model_direction(meta_angle, base_angle, proximal_angle)
    dot = max(-1.0, min(1.0, sum(direction[i] * target_direction[i] for i in range(3))))
    return math.acos(dot)


def _thumb_model_direction(
    meta_angle: float,
    base_angle: float,
    proximal_angle: float,
) -> list[float]:
    rotation = _matmul(
        _rpy_to_matrix(_THUMB_ROOT_RPY),
        _matmul(
            _rotation_x(-meta_angle),
            _matmul(
                _rotation_z(base_angle),
                _rotation_y(proximal_angle),
            ),
        ),
    )
    return _normalize_vector(_matvec(rotation, [0.0, 0.0, 1.0]))


def _rotation_x(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ]


def _rotation_y(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ]


def _rotation_z(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _rpy_to_matrix(rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    return _matmul(
        _rotation_z(yaw),
        _matmul(_rotation_y(pitch), _rotation_x(roll)),
    )


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    ]


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude <= 1e-9:
        return [0.0, 0.0, 1.0]
    return [component / magnitude for component in vector]
