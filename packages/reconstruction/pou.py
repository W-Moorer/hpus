"""
PoU (Partition of Unity) 权重模块

实现 C² 紧支撑 bump 函数和权重计算。
"""

import numpy as np
from typing import List, Tuple, Optional


class C2Bump:
    """
    C² 紧支撑 bump 函数
    
    ρ(t) = (1-t)^4 * (4t+1),  0 ≤ t < 1
         = 0,                  t ≥ 1
    
    满足 ρ(1) = ρ'(1) = ρ''(1) = 0
    """
    
    @staticmethod
    def evaluate(t: np.ndarray) -> np.ndarray:
        """
        计算 bump 函数值
        
        参数:
            t: 归一化距离 (可以是标量或数组)
        
        返回:
            ρ(t): bump 函数值
        """
        t = np.asarray(t)
        result = np.zeros_like(t, dtype=float)
        mask = t < 1.0
        t_masked = t[mask]
        result[mask] = np.power(1 - t_masked, 4) * (4 * t_masked + 1)
        return result
    
    @staticmethod
    def gradient(t: np.ndarray) -> np.ndarray:
        """
        计算 bump 函数一阶导数
        
        ρ'(t) = -4(1-t)^3 * (4t+1) + 4(1-t)^4
              = -4(1-t)^3 * (4t+1 - (1-t))
              = -4(1-t)^3 * (5t)
              = -20t(1-t)^3
        
        参数:
            t: 归一化距离
        
        返回:
            ρ'(t): bump 函数一阶导数
        """
        t = np.asarray(t)
        result = np.zeros_like(t, dtype=float)
        mask = t < 1.0
        t_masked = t[mask]
        result[mask] = -20 * t_masked * np.power(1 - t_masked, 3)
        return result
    
    @staticmethod
    def hessian(t: np.ndarray) -> np.ndarray:
        """
        计算 bump 函数二阶导数
        
        ρ''(t) = -20(1-t)^3 + 60t(1-t)^2
               = -20(1-t)^2 * ((1-t) - 3t)
               = -20(1-t)^2 * (1 - 4t)
        
        参数:
            t: 归一化距离
        
        返回:
            ρ''(t): bump 函数二阶导数
        """
        t = np.asarray(t)
        result = np.zeros_like(t, dtype=float)
        mask = t < 1.0
        t_masked = t[mask]
        result[mask] = -20 * np.power(1 - t_masked, 2) * (1 - 4 * t_masked)
        return result


def compute_quadratic_distance(x: np.ndarray, center: np.ndarray, 
                                Q: Optional[np.ndarray] = None) -> float:
    """
    计算二次型距离 q(x) = (x - ξ)ᵀ Q (x - ξ)
    
    参数:
        x: 查询点 (3,)
        center: patch 中心 ξ (3,)
        Q: 正定二次型矩阵 (3x3)，默认为单位矩阵（球形 patch）
    
    返回:
        q: 二次型距离
    """
    diff = x - center
    if Q is None:
        return np.dot(diff, diff)
    else:
        return np.dot(diff, Q @ diff)


def compute_weights(x: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                    Q_matrices: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, float]:
    """
    计算归一化 PoU 权重
    
    w_m(x) = w̃_m(x) / Σ_j w̃_j(x)
    
    其中 w̃_m(x) = ρ(q_m(x) / ρ_m²)
    
    参数:
        x: 查询点 (3,)
        centers: patch 中心 (M x 3)
        radii: patch 半径 (M,)
        Q_matrices: 二次型矩阵列表，可选
    
    返回:
        weights: 归一化权重 (M,)
        weight_sum: 未归一化权重之和（用于检查正下界）
    """
    M = len(centers)
    raw_weights = np.zeros(M)
    
    for m in range(M):
        Q = Q_matrices[m] if Q_matrices is not None else None
        q = compute_quadratic_distance(x, centers[m], Q)
        t = q / (radii[m] ** 2)
        raw_weights[m] = C2Bump.evaluate(t)
    
    weight_sum = np.sum(raw_weights)
    
    if weight_sum > 1e-10:
        weights = raw_weights / weight_sum
    else:
        # 分母过小，返回零权重
        weights = np.zeros(M)
    
    return weights, weight_sum


def compute_weights_batch(query_points: np.ndarray, centers: np.ndarray, 
                          radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量计算权重（向量化版本，仅支持球形 patch）
    
    参数:
        query_points: 查询点 (N x 3)
        centers: patch 中心 (M x 3)
        radii: patch 半径 (M,)
    
    返回:
        weights: 归一化权重 (N x M)
        weight_sums: 未归一化权重之和 (N,)
    """
    N = len(query_points)
    M = len(centers)
    
    # 计算距离平方 (N x M)
    diff = query_points[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, M, 3)
    dist_sq = np.sum(diff ** 2, axis=2)  # (N, M)
    
    # 归一化距离
    t = dist_sq / (radii ** 2)  # (N, M)
    
    # 计算 bump 权重
    raw_weights = C2Bump.evaluate(t)  # (N, M)
    
    # 归一化
    weight_sums = np.sum(raw_weights, axis=1)  # (N,)
    
    # 避免除零
    safe_sums = np.maximum(weight_sums, 1e-10)
    weights = raw_weights / safe_sums[:, np.newaxis]
    
    return weights, weight_sums


def find_active_patches(x: np.ndarray, centers: np.ndarray, 
                        radii: np.ndarray) -> np.ndarray:
    """
    找到对查询点有贡献的 patch（权重非零）
    
    参数:
        x: 查询点 (3,)
        centers: patch 中心 (M x 3)
        radii: patch 半径 (M,)
    
    返回:
        active_indices: 有贡献的 patch 索引
    """
    dist_sq = np.sum((centers - x) ** 2, axis=1)
    active = dist_sq < radii ** 2
    return np.where(active)[0]
