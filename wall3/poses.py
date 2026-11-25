"""벽돌 자세 ↔ TCP(패드 중심) 자세. 대본·도달 지도가 같이 쓴다.

③ 집는 높이: 손가락 최하단이 벽돌 아랫면보다 grasp_clear(2 mm) 위. 패드 중심은 벽돌 중심보다 dz_pad 위.
TCP 방향: 접근축(z) = 세계 −z, 닫힘축(y) = 벽돌 너비 방향, x = 벽돌 길이 방향. 벽돌은 앞뒤가 같아 yaw와 yaw+π 중 고른다.
"""
import numpy as np

from wall3.config import H
from wall3.ik import down_R, wrap


def dz_pad(cal, clear):
    """Height of the pad centre above the brick centre when grasping (m)."""
    return (cal["deepest_finger_depth"] - cal["pad_center_depth"] + clear) - H / 2


def tcp_for_brick(p_brick, yaw, cal, clear, lift=0.0):
    p = np.asarray(p_brick, float) + [0.0, 0.0, dz_pad(cal, clear) + lift]
    return p, down_R(yaw)


def pick_yaw(yaw_brick, wrist3_now, wrist3_of_yaw):
    """Of yaw and yaw+π (same brick), the one whose wrist_3 angle is closest to the current one."""
    cands = [yaw_brick, yaw_brick + np.pi]
    return min(cands, key=lambda y: abs(wrap(wrist3_of_yaw(y) - wrist3_now)))


def yaw_of_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def grip_joint_qpos(theta):
    """Finger joint angles for a gripper closure theta with the A1+ parallel coupling (left +, right mirrored)."""
    return {"finger_joint": theta, "left_inner_finger_joint": theta, "left_inner_knuckle_joint": -theta,
            "right_outer_knuckle_joint": -theta, "right_inner_finger_joint": theta, "right_inner_knuckle_joint": -theta}
