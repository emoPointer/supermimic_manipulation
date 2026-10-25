"""Differentiable serial-chain pose algebra; never controls the simulation.

The FK accepts the existing SerialLinkCfg sequence without importing Warp or
mjlab. Quaternions are wxyz, rotations act on column vectors, and six-dimensional
rotations store the first two matrix columns.
"""

import torch
import torch.nn.functional as F


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat(
        (aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.linalg.cross(av, bv)),
        -1,
    )


def quat_inverse(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_apply(q, v):
    t = 2 * torch.linalg.cross(q[..., 1:], v)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t)


def quat_to_matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_to_six(q):
    return quat_to_matrix(q)[..., :, :2].transpose(-1, -2).flatten(-2)


def six_to_matrix(six):
    first = F.normalize(six[..., :3], dim=-1, eps=1e-8)
    second = six[..., 3:] - (first * six[..., 3:]).sum(-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-8)
    return torch.stack((first, second, torch.linalg.cross(first, second)), -1)


def matrix_to_quat(matrix):
    """Largest-component quaternion conversion with bounded inactive gradients."""
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    squared = torch.stack(
        (1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1
    )
    roots = squared.clamp_min(1e-12).sqrt()
    candidates = torch.stack(
        (
            torch.stack((squared[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, squared[..., 1], m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, squared[..., 2], m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, squared[..., 3]), -1),
        ),
        -2,
    ) / (2 * roots.clamp_min(0.1)[..., :, None])
    index = roots.argmax(-1)[..., None, None].expand(*roots.shape[:-1], 1, 4)
    q = candidates.gather(-2, index).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def six_to_quat(six):
    return matrix_to_quat(six_to_matrix(six))


def axis_angle(q):
    q = torch.where(q[..., :1] < 0, -q, q)
    norm = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, q[..., :1])
    # sinc's origin limit prevents 0/0 and preserves the identity derivative.
    return q[..., 1:] / (0.5 * torch.sinc(angle / (2 * torch.pi)))


def quat_from_rotation_vector(vector):
    """SO(3) exponential with its analytic, finite identity derivative."""
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return torch.cat(((angle / 2).cos(), 0.5 * torch.sinc(angle / (2 * torch.pi)) * vector), -1)


def relative_pose(target_pos, target_quat, reference_pos, reference_quat):
    inverse = quat_inverse(reference_quat)
    return torch.cat(
        (
            quat_apply(inverse, target_pos - reference_pos),
            axis_angle(quat_mul(inverse, target_quat)),
        ),
        -1,
    )


def serial_chain_fk(joints, links):
    """Same origin-then-axis rotation order as BatchedSerialChain.evaluate."""
    if joints.shape[-1] != len(links):
        raise ValueError("Joint width must match the declared serial links")
    position = torch.zeros_like(joints[..., :3])
    orientation = torch.zeros((*joints.shape[:-1], 4), device=joints.device, dtype=joints.dtype)
    orientation[..., 0] = 1
    for index, link in enumerate(links):
        origin = joints.new_tensor(link.position).expand_as(position)
        rotation = joints.new_tensor(link.quaternion).expand_as(orientation)
        position = position + quat_apply(orientation, origin)
        orientation = quat_mul(orientation, rotation)
        half = joints[..., index : index + 1] / 2
        axis = joints.new_tensor(link.axis)
        relative = torch.cat((half.cos(), half.sin() * axis), -1)
        orientation = quat_mul(orientation, relative)
    return position, F.normalize(orientation, dim=-1)
