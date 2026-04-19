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
    meta_baseline: float | None = None
    base_baseline: float | None = None


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
