import torch
import torch.nn as nn


def kd_loss(pred_embeds, target_embeds):
    dist_loss = torch.square(target_embeds - pred_embeds).sum(dim=-1)
    cosine_similarity = 1 - torch.cosine_similarity(target_embeds, pred_embeds, dim=-1)
    return (dist_loss + cosine_similarity).mean()


def rkd_loss(student_embeds, teacher_embeds):
    distance_loss = nn.functional.huber_loss(
        rkd_dist(student_embeds), rkd_dist(teacher_embeds)
    )
    angle_loss = nn.functional.huber_loss(
        rkd_angle(student_embeds), rkd_angle(teacher_embeds)
    )
    return (distance_loss + angle_loss).mean()


def rkd_dist(embeds):
    N = embeds.shape[0]
    epsilon = 1e-6
    diff = embeds.unsqueeze(0) - embeds.unsqueeze(1)
    diff = torch.linalg.vector_norm(diff, dim=-1)
    mu = diff.sum() / (N * N) + epsilon
    return diff / mu


def rkd_angle(embeds):
    epsilon = 1e-6
    N, dim = embeds.shape
    v_i = embeds.reshape(1, 1, N, dim)
    v_j = embeds.reshape(1, N, 1, dim)
    v_k = embeds.reshape(N, 1, 1, dim)

    e_ij = (v_i - v_j) / (
        torch.linalg.vector_norm(v_i - v_j) + epsilon
    )  # [1, N, N, dim]
    e_kj = (v_k - v_j) / (
        torch.linalg.vector_norm(v_k - v_j) + epsilon
    )  # [N, N, 1, dim]

    cos_angle = (e_ij * e_kj).sum(-1)
    return cos_angle
