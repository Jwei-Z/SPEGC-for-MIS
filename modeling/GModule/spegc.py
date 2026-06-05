import torch
import torch.nn as nn
import torch.nn.functional as F

class SPEGC(nn.Module):
    """
    SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering 
    for Medical Image Segmentation (CVPR 2026)
    """
    def __init__(self, dim=256, Z=48, M=8, tau=0.1, theta=0.05, lambda_c=0.2):
        super(SPEGC, self).__init__()
        self.dim = dim
        self.Z = Z              # Number of target clusters
        self.M = M              # Number of prompts
        self.tau = tau          # Temperature for density-based edge similarity
        self.theta = theta      # Entropic regularization parameter for Sinkhorn
        self.lambda_c = lambda_c  # Balancing hyperparameter for clustering loss L_C

        # Learnable prompt pools
        self.P_CO = nn.Parameter(torch.randn(M, dim) * 0.01)
        self.P_HE = nn.Parameter(torch.randn(M, dim) * 0.01)

        # Learnable pooling context vector for dynamic query pooling
        self.c_p = nn.Parameter(torch.randn(dim) * 0.01)

        # Projections for similarity matrix
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.W_q.weight, std=0.01)
        nn.init.normal_(self.W_k.weight, std=0.01)

    def run_sinkhorn(self, D_cost, k):
        """
        Parallel Sinkhorn algorithm to solve optimal transport with marginals:
        r = 1_E (row sums = 1)
        c = [E - k, k]^T (column sums)
        """
        E = D_cost.shape[0]
        device = D_cost.device
        
        # Gamma0 = exp(-D_cost / theta)
        Gamma = torch.exp(-D_cost / self.theta)
        
        # Clamp Gamma to prevent underflow to exactly 0.0 which causes NaN gradients in division
        Gamma = torch.clamp(Gamma, min=1e-12)
        
        r = torch.ones(E, 1, dtype=D_cost.dtype, device=device)
        c = torch.tensor([[E - k], [k]], dtype=D_cost.dtype, device=device)
        
        # Run iterations to converge
        for _ in range(20):
            # Row normalization
            row_sums = Gamma.sum(dim=1, keepdim=True)
            Gamma = Gamma / torch.clamp(row_sums, min=1e-12)
            
            # Column normalization
            col_sums = Gamma.sum(dim=0, keepdim=True).T # (2, 1)
            scale = c / torch.clamp(col_sums, min=1e-12)
            Gamma = Gamma * scale.T
            
        return Gamma

    def forward(self, nodes_batch, centroids):
        """
        Args:
            nodes_batch: List of B tensors, each representing node features V_i of shape (n_i, dim)
            centroids: Cluster centers of shape (Z, dim)
        Returns:
            loss: Joint loss L = L_G + lambda * L_C
        """
        device = centroids.device
        B = len(nodes_batch)
        if B <= 1:
            return torch.tensor(0.0, device=device)

        # 1. Semantic Prompt Feature Enhancement (SPFE)
        enhanced_nodes = []
        p_CO_list = []
        
        for V_i in nodes_batch:
            # Dynamic query pooling
            scores = torch.matmul(V_i, self.c_p) # (n_i,)
            attn_weights = F.softmax(scores, dim=0) # (n_i,)
            q_i = torch.matmul(attn_weights, V_i) # (dim,)
            
            # Normalize for cosine similarity calculation
            q_i_norm = F.normalize(q_i, p=2, dim=-1)
            P_CO_norm = F.normalize(self.P_CO, p=2, dim=-1)
            P_HE_norm = F.normalize(self.P_HE, p=2, dim=-1)
            
            # Cosine similarity
            cos_sim_CO = torch.matmul(P_CO_norm, q_i_norm) # (M,)
            cos_sim_HE = torch.matmul(P_HE_norm, q_i_norm) # (M,)
            
            # Retrieval weights
            alpha_CO = F.relu(-cos_sim_CO) # (M,)
            alpha_HE = F.softmax(cos_sim_HE, dim=0) # (M,)
            
            # Decoupled prompts retrieval
            p_CO = torch.matmul(alpha_CO, self.P_CO) # (dim,)
            p_HE = torch.matmul(alpha_HE, self.P_HE) # (dim,)
            
            p_CO_list.append(p_CO)
            
            # Modulate feature maps
            V_i_enhanced = V_i + p_CO.unsqueeze(0) + p_HE.unsqueeze(0)
            enhanced_nodes.append(V_i_enhanced)

        # Concatenate nodes to form global pseudo-batch
        V_star = torch.cat(enhanced_nodes, dim=0) # (V, dim)
        V_total = V_star.shape[0]

        # 2. Differentiable Graph Clustering Solver (DGCS)
        q = self.W_q(V_star) # (V, dim)
        k_proj = self.W_k(V_star) # (V, dim)
        S = torch.matmul(q, k_proj.T) / (self.dim ** 0.5) # (V, V)

        # Node density calculation
        S_plus = F.relu(S)
        D = S_plus.sum(dim=1) # (V,)

        # Density-based edge similarity matrix S'
        density_diff = D.unsqueeze(0) - D.unsqueeze(1) # (V, V) where diff[i, j] = D[j] - D[i]
        S_prime = F.relu(S) * torch.sigmoid(density_diff / self.tau) # (V, V)

        # Flatten S_prime to directed affinity vector
        d = S_prime.view(-1) # (E,) where E = V^2
        E = d.shape[0]
        
        d_max = d.max()
        d_min = d.min()

        # Binary cost matrix (Way 3: D_i1 = cost of rejecting, D_i2 = cost of selecting)
        D_cost = torch.zeros(E, 2, dtype=d.dtype, device=device)
        D_cost[:, 0] = d - d_min     # Cost to reject (smaller for small d_i)
        D_cost[:, 1] = d_max - d     # Cost to select (smaller for large d_i)

        # Normalize D_cost to [0, 1] to prevent exp() underflow/overflow and stabilize gradients
        d_range = d_max - d_min
        if d_range > 1e-8:
            D_cost = D_cost / d_range

        # Spanning forest edge budget: k = V - Z
        edge_budget = max(1, V_total - self.Z)

        # Solve optimal transport via Sinkhorn
        Gamma = self.run_sinkhorn(D_cost, edge_budget)
        
        # refined edge similarity matrix S^*
        S_star = Gamma[:, 1].view(V_total, V_total)

        # Soft cluster assignment predictions P_i
        P = F.softmax(torch.matmul(V_star, centroids.T), dim=1) # (V, Z)
        P_detached = P.detach()
        
        # Clamp assignment distributions to prevent log(0) NaN gradients
        log_P = torch.log(torch.clamp(P, min=1e-12))
        log_P_detached = torch.log(torch.clamp(P_detached, min=1e-12))
        
        # Vectorized KL Divergence: KL(P_j || sg(P_i))
        neg_entropy_j = (P * log_P).sum(dim=1) # (V,)
        cross_term = torch.matmul(P, log_P_detached.T) # (V, V)
        kl = neg_entropy_j.unsqueeze(0) - cross_term.T # (V, V)
        
        # Graph Consistency Loss L_G
        L_G = (S_star * kl).sum() / V_total

        # Clustering Loss L_C (forces commonality prompts p_CO to align across batch)
        P_CO_batch = torch.stack(p_CO_list, dim=0) # (B, dim)
        P_CO_batch_norm = F.normalize(P_CO_batch, p=2, dim=1)
        sim = torch.matmul(P_CO_batch_norm, P_CO_batch_norm.T) # (B, B)
        L_C = (1.0 - sim).mean()

        # Joint loss
        loss = L_G + self.lambda_c * L_C
        return loss
