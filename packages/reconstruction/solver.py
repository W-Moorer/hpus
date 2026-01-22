"""
HRBF GPU 求解器模块 (PyTorch)

使用 PyTorch 实现 GPU 加速的批量评估。
"""

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class NormalDirectionalSolverGPU:
    """
    GPU 加速的 HRBF 求解器
    
    使用 PyTorch 实现向量化批量评估。
    """
    
    def __init__(self, kernel_type: str = 'phs', k: int = 5, 
                 device: str = 'cuda'):
        """
        初始化 GPU 求解器
        
        参数:
            kernel_type: 核函数类型
            k: PHS 多项式次数
            device: 计算设备 ('cuda' 或 'cpu')
        """
        if not HAS_TORCH:
            raise ImportError("需要安装 PyTorch: pip install torch")
        
        self.kernel_type = kernel_type
        self.k = k
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 缓存求解结果 (保持在 GPU 上)
        self.alpha = None
        self.eta = None
        self.c = None
        self.constraint_points = None
        self.constraint_normals = None
        self.shift = 0.0  # 常数移位 μ_m
    
    def phi(self, r: torch.Tensor) -> torch.Tensor:
        """PHS 核函数值"""
        return torch.pow(r, self.k)
    
    def d1(self, r: torch.Tensor) -> torch.Tensor:
        """PHS 核函数一阶导数"""
        return self.k * torch.pow(r, self.k - 1)
    
    def solve(self, points: np.ndarray, normals: np.ndarray,
              f: np.ndarray = None, g: np.ndarray = None,
              ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        求解 HRBF 系数 (CPU 上求解，结果传到 GPU)
        
        参数:
            points: 约束点坐标 (N x 3)
            normals: 约束点法向 (N x 3)
            f: 值约束目标
            g: 法向导数约束目标
            ridge: 正则化参数
        
        返回:
            alpha, eta, c
        """
        N = len(points)
        size = 2 * N + 4
        
        if f is None:
            f = np.zeros(N)
        if g is None:
            g = np.ones(N)
        
        # CPU 上组装矩阵
        A = np.zeros((size, size))
        b = np.zeros(size)
        
        # 计算距离
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        r = np.linalg.norm(diff, axis=2)
        r_safe = np.maximum(r, 1e-10)
        
        # 核函数
        Phi = np.power(r, self.k)
        d1 = self.k * np.power(r, self.k - 1)
        d2 = self.k * (self.k - 1) * np.power(r, self.k - 2)
        
        c1 = np.zeros_like(r)
        mask = r > 1e-10
        c1[mask] = d1[mask] / r_safe[mask]
        
        # G 和 D 块
        G = np.sum(c1[:, :, np.newaxis] * diff * normals[np.newaxis, :, :], axis=2)
        D = np.sum(c1[:, :, np.newaxis] * diff * normals[:, np.newaxis, :], axis=2)
        
        # H 块（向量化）
        u = np.zeros_like(diff)
        u[mask] = diff[mask] / r[mask, np.newaxis]
        
        # H_ij = n_i^T (d2 * u u^T + c1 * (I - u u^T)) n_j
        # = d2 * (n_i · u)(u · n_j) + c1 * (n_i · n_j - (n_i · u)(u · n_j))
        ni_dot_u = np.sum(normals[:, np.newaxis, :] * u, axis=2)  # (N, N)
        u_dot_nj = np.sum(u * normals[np.newaxis, :, :], axis=2)  # (N, N)
        ni_dot_nj = np.sum(normals[:, np.newaxis, :] * normals[np.newaxis, :, :], axis=2)
        
        H = d2 * ni_dot_u * u_dot_nj + c1 * (ni_dot_nj - ni_dot_u * u_dot_nj)
        
        # 多项式块
        P = np.hstack([np.ones((N, 1)), points])
        R = np.hstack([np.zeros((N, 1)), normals])
        
        # 组装
        A[:N, :N] = Phi
        A[:N, N:2*N] = -G
        A[:N, 2*N:] = P
        A[N:2*N, :N] = D
        A[N:2*N, N:2*N] = -H
        A[N:2*N, 2*N:] = R
        A[2*N:, :N] = P.T
        A[2*N:, N:2*N] = R.T
        
        b[:N] = f
        b[N:2*N] = g
        
        # 正则化
        A += ridge * np.eye(size)
        
        # 求解
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # 存储到 GPU
        self.alpha = torch.tensor(x[:N], dtype=torch.float32, device=self.device)
        self.eta = torch.tensor(x[N:2*N], dtype=torch.float32, device=self.device)
        self.c = torch.tensor(x[2*N:], dtype=torch.float32, device=self.device)
        self.constraint_points = torch.tensor(points, dtype=torch.float32, device=self.device)
        self.constraint_normals = torch.tensor(normals, dtype=torch.float32, device=self.device)
        
        # 计算常数移位 μ_m
        self.shift = 0.0
        values_at_constraints = self.evaluate_batch(points)
        self.shift = float(np.mean(values_at_constraints))
        
        return x[:N], x[N:2*N], x[2*N:]
    
    def evaluate_batch(self, query_points: np.ndarray, return_tensor: bool = False) -> np.ndarray:
        """
        GPU 批量评估 HRBF 值
        
        参数:
            query_points: 查询点 (M x 3)
            return_tensor: 是否返回 torch.Tensor (保持在 GPU 上)
        
        返回:
            values: HRBF 值 (M,)
        """
        if self.alpha is None:
            raise RuntimeError("请先调用 solve() 方法")
        
        M = len(query_points)
        N = len(self.constraint_points)
        
        # 转移到 GPU
        x = torch.tensor(query_points, dtype=torch.float32, device=self.device)  # (M, 3)
        
        # 计算距离 (M, N, 3)
        diff = x[:, None, :] - self.constraint_points[None, :, :]  # (M, N, 3)
        r = torch.norm(diff, dim=2)  # (M, N)
        
        # 核函数值
        phi_vals = self.phi(r)  # (M, N)
        
        # φ(r) 项：Σ_j α_j φ(r_j)
        term1 = torch.sum(phi_vals * self.alpha[None, :], dim=1)  # (M,)
        
        # 梯度项：-Σ_j η_j (n_j · ∇φ)
        r_safe = torch.clamp(r, min=1e-10)
        c1 = self.d1(r) / r_safe  # (M, N)
        c1 = torch.where(r > 1e-10, c1, torch.zeros_like(c1))
        
        # grad_phi · n_j = c1 * (diff · n_j)
        grad_dot_n = torch.sum(diff * self.constraint_normals[None, :, :], dim=2)  # (M, N)
        grad_dot_n = c1 * grad_dot_n
        
        term2 = torch.sum(grad_dot_n * self.eta[None, :], dim=1)  # (M,)
        
        # 多项式项
        ones = torch.ones(M, 1, device=self.device)
        p = torch.cat([ones, x], dim=1)  # (M, 4)
        term3 = torch.sum(p * self.c[None, :], dim=1)  # (M,)
        
        values = term1 - term2 + term3
        
        if return_tensor:
            return values
        return values.cpu().numpy()


def batch_evaluate_all_solvers(query_points: np.ndarray, 
                               solvers: list,
                               patch_centers: np.ndarray,
                               patch_radius: float,
                               device: str = 'cuda',
                               use_linear_blend: bool = True,
                               tau: float = 0.01,
                               gamma: float = 0.5) -> np.ndarray:
    """
    批量评估所有 solver 的值并融合
    
    参数:
        query_points: 查询点 (M x 3)
        solvers: NormalDirectionalSolverGPU 列表
        patch_centers: patch 中心 (P x 3)
        patch_radius: patch 半径
        device: 计算设备
        use_linear_blend: True=线性融合, False=NG-RBlend非线性融合
        tau: NG-RBlend 温度参数（越小越接近 min/max）
        gamma: NG-RBlend 门控参数（0=soft-min, 1=soft-max）
    
    返回:
        values: 融合后的场值 (M,)
    """
    if not HAS_TORCH:
        raise ImportError("需要安装 PyTorch")
    
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    M = len(query_points)
    P = len(patch_centers)
    
    # 转移到 GPU
    x = torch.tensor(query_points, dtype=torch.float32, device=dev)  # (M, 3)
    centers = torch.tensor(patch_centers, dtype=torch.float32, device=dev)  # (P, 3)
    
    # 计算距离平方
    dist_sq = torch.sum((x[:, None, :] - centers[None, :, :]) ** 2, dim=2)  # (M, P)
    
    # C2 Bump 权重
    t = dist_sq / (patch_radius ** 2)
    weights = torch.where(
        t < 1.0,
        torch.pow(1 - t, 4) * (4 * t + 1),
        torch.zeros_like(t)
    )  # (M, P)
    
    # 权重和
    weight_sum = torch.sum(weights, dim=1)  # (M,)
    
    # 收集所有 patch 的局部场值 (M, P)
    local_values = torch.full((M, P), float('nan'), device=dev)
    
    # 对每个 solver 评估
    for p_idx, solver in enumerate(solvers):
        if solver is None:
            continue
        
        # 找到该 patch 有贡献的查询点
        active = weights[:, p_idx] > 1e-10
        if not torch.any(active):
            continue
        
        active_indices = torch.where(active)[0]
        active_points = x[active]
        
        # 评估局部 HRBF
        diff = active_points[:, None, :] - solver.constraint_points[None, :, :]
        r = torch.norm(diff, dim=2)
        
        phi_vals = solver.phi(r)
        term1 = torch.sum(phi_vals * solver.alpha[None, :], dim=1)
        
        r_safe = torch.clamp(r, min=1e-10)
        c1 = solver.d1(r) / r_safe
        c1 = torch.where(r > 1e-10, c1, torch.zeros_like(c1))
        
        grad_dot_n = torch.sum(diff * solver.constraint_normals[None, :, :], dim=2)
        grad_dot_n = c1 * grad_dot_n
        term2 = torch.sum(grad_dot_n * solver.eta[None, :], dim=1)
        
        ones = torch.ones(len(active_points), 1, device=dev)
        p_poly = torch.cat([ones, active_points], dim=1)
        term3 = torch.sum(p_poly * solver.c[None, :], dim=1)
        
        vals = term1 - term2 + term3
        
        # 应用常数移位 μ_m
        vals = vals - solver.shift
        
        local_values[active_indices, p_idx] = vals
    
    # 融合
    if use_linear_blend:
        # 线性 PoU 融合
        local_values_safe = torch.where(torch.isnan(local_values), torch.zeros_like(local_values), local_values)
        weighted_sum = torch.sum(weights * local_values_safe, dim=1)
        safe_weight_sum = torch.clamp(weight_sum, min=1e-10)
        field_values = weighted_sum / safe_weight_sum
    else:
        # NG-RBlend 非线性融合 (完全向量化 soft-min/soft-max)
        
        # 创建有效值掩码 (M, P)
        valid_mask = (~torch.isnan(local_values)) & (weights > 1e-10)
        
        # 将无效位置设为特殊值（不影响 log-sum-exp 结果）
        # 对于 soft-min: 无效位置的 -v/tau 设为 -inf（exp后为0）
        # 对于 soft-max: 无效位置的 v/tau 设为 -inf（exp后为0）
        local_values_safe = torch.where(valid_mask, local_values, torch.zeros_like(local_values))
        weights_safe = torch.where(valid_mask, weights, torch.zeros_like(weights))
        
        # 权重和 B = Σ w (M,)
        B = torch.sum(weights_safe, dim=1)
        B_safe = torch.clamp(B, min=1e-10)
        
        if tau < 1e-10:
            # τ → 0 退化为 min/max
            # 使用 masked 操作找 min/max
            masked_for_min = torch.where(valid_mask, local_values, torch.full_like(local_values, float('inf')))
            masked_for_max = torch.where(valid_mask, local_values, torch.full_like(local_values, float('-inf')))
            s_min = torch.min(masked_for_min, dim=1).values
            s_max = torch.max(masked_for_max, dim=1).values
        else:
            # Soft-min: s_min = -τ log(Σ w exp(-s/τ)) + τ log(B)
            # 使用 log-sum-exp 技巧保证数值稳定
            scaled_min = -local_values_safe / tau  # (M, P)
            scaled_min = torch.where(valid_mask, scaled_min, torch.full_like(scaled_min, float('-inf')))
            max_scaled_min = torch.max(scaled_min, dim=1, keepdim=True).values  # (M, 1)
            exp_min = weights_safe * torch.exp(scaled_min - max_scaled_min)  # (M, P)
            log_sum_min = max_scaled_min.squeeze(1) + torch.log(torch.sum(exp_min, dim=1))  # (M,)
            s_min = -tau * log_sum_min + tau * torch.log(B_safe)
            
            # Soft-max: s_max = τ log(Σ w exp(s/τ)) - τ log(B)
            scaled_max = local_values_safe / tau  # (M, P)
            scaled_max = torch.where(valid_mask, scaled_max, torch.full_like(scaled_max, float('-inf')))
            max_scaled_max = torch.max(scaled_max, dim=1, keepdim=True).values  # (M, 1)
            exp_max = weights_safe * torch.exp(scaled_max - max_scaled_max)  # (M, P)
            log_sum_max = max_scaled_max.squeeze(1) + torch.log(torch.sum(exp_max, dim=1))  # (M,)
            s_max = tau * log_sum_max - tau * torch.log(B_safe)
        
        # 混合 soft-min 和 soft-max
        field_values = gamma * s_max + (1 - gamma) * s_min
    
    # 处理无覆盖区域
    no_coverage = weight_sum < 1e-10
    field_values[no_coverage] = 1.0  # 外部默认值
    
    return field_values.cpu().numpy()

