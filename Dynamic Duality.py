import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


# ====================
# 1. 量子树系统（带时间戳 + 动态纠缠）
# ====================

class QuantumTreeSystem:
    """
    量子树系统：节点之间通过量子纠缠连接
    自指性来自量子测量 + 视角间可学习的翻译器
    """

    def __init__(self):
        # 节点定义
        self.nodes = ['root', 'N1', 'N2', 'N3', 'L1', 'L2', 'L3', 'L4', 'L5']
        self.n_nodes = len(self.nodes)

        # 节点索引映射
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}

        # 寿命（根最长，叶最短）
        self.lifetime = torch.tensor([
            100,  # root
            60, 50, 40,  # N1, N2, N3
            10, 8, 6, 7, 5  # L1-L5
        ], dtype=torch.float32)

        # ===== 量子态 =====
        self.n_qubits = self.n_nodes
        self.dim = 2 ** self.n_qubits

        # 初始化量子态 |0...0>
        self.quantum_state = np.zeros(self.dim, dtype=complex)
        self.quantum_state[0] = 1.0

        # 纠缠结构（初始由树拓扑决定，之后动态演化）
        self.entanglement_matrix = self._build_entanglement_matrix()

        # ===== 可学习的视角间翻译器 =====
        # 加大容量，加深网络
        self.translator_q_to_rl = nn.Sequential(
            nn.Linear(self.n_nodes, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_nodes),
            nn.LogSoftmax(dim=-1)  # 用 LogSoftmax 更稳定
        )

        self.translator_rl_to_q = nn.Sequential(
            nn.Linear(self.n_nodes, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_nodes),
            nn.LogSoftmax(dim=-1)
        )

        self.translator_optimizer = optim.Adam(
            list(self.translator_q_to_rl.parameters()) +
            list(self.translator_rl_to_q.parameters()),
            lr=0.005  # 稍小的学习率
        )

        # ===== 带时间戳的测量历史 =====
        self.q_measurements = deque(maxlen=500)  # (step, result, features)
        self.rl_measurements = deque(maxlen=500)

        # ===== 真正的自我认知指标 =====
        self.self_awareness = 0.0
        self.prediction_accuracy = 0.0
        self.mutual_info = 0.0

    def _build_entanglement_matrix(self):
        """根据树拓扑构建初始纠缠结构"""
        # 树结构
        tree_structure = {
            'root': ['N1', 'N2', 'N3', 'L1', 'L4'],
            'N1': ['root', 'L1', 'L2'],
            'N2': ['root', 'L3'],
            'N3': ['root', 'L4', 'L5'],
            'L1': ['root', 'N1'],
            'L2': ['N1'],
            'L3': ['N2'],
            'L4': ['root', 'N3'],
            'L5': ['N3']
        }

        # 纠缠强度矩阵（与寿命相关）
        E = np.zeros((self.n_nodes, self.n_nodes))
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if node2 in tree_structure.get(node1, []):
                    # 纠缠强度由两个节点的寿命决定
                    E[i, j] = np.sqrt(self.lifetime[i].item() * self.lifetime[j].item()) / 100
        return torch.tensor(E, dtype=torch.float32)

    def _get_quantum_state_features(self, base_node):
        """从当前量子态提取特征"""
        # 基于纠缠矩阵生成特征
        features = self.entanglement_matrix[base_node].clone()
        # 加入少量量子噪声（模拟不确定性）
        features += torch.randn(self.n_nodes) * 0.05
        return torch.softmax(features, dim=-1)

    def measure_from_perspective(self, perspective, step):
        """
        从特定视角测量系统
        返回：(测量节点, 测量结果, 测量时的量子特征)
        带时间戳记录
        """
        if perspective == 'quantum':
            # 量子视角：倾向于测量根附近
            probs = torch.softmax(self.entanglement_matrix[0], dim=-1).numpy()
            base_node = 0
        else:  # 'rl'
            # RL视角：倾向于测量叶子
            leaf_indices = [4, 5, 6, 7, 8]
            # 但也会受纠缠影响
            leaf_probs = []
            for leaf in leaf_indices:
                leaf_probs.append(self.entanglement_matrix[leaf].sum().item())
            leaf_probs = np.array(leaf_probs)
            leaf_probs = leaf_probs / leaf_probs.sum()
            base_node = np.random.choice(leaf_indices, p=leaf_probs)
            probs = torch.softmax(self.entanglement_matrix[base_node], dim=-1).numpy()

        # 测量导致坍缩
        measurement_result = np.random.choice(self.n_nodes, p=probs)

        # 获取测量时的量子特征
        features = self._get_quantum_state_features(base_node)

        # 带时间戳记录
        if perspective == 'quantum':
            self.q_measurements.append((step, measurement_result, features))
        else:
            self.rl_measurements.append((step, measurement_result, features))

        return base_node, measurement_result, features

    def update_entanglement(self, step):
        """
        根据两个视角的测量更新纠缠结构
        这才是真正的量子演化！
        """
        if len(self.q_measurements) < 10 or len(self.rl_measurements) < 10:
            return

        # 找出最近的时间对齐的测量对
        q_dict = {s: (r, f) for s, r, f in self.q_measurements}
        rl_dict = {s: (r, f) for s, r, f in self.rl_measurements}

        common_steps = sorted(set(q_dict.keys()) & set(rl_dict.keys()))
        if len(common_steps) < 5:
            return

        # 用最近的对齐测量更新纠缠
        recent_steps = common_steps[-5:]

        for s in recent_steps:
            q_res, q_feat = q_dict[s]
            rl_res, rl_feat = rl_dict[s]

            # 如果两个视角测量到相同或相邻节点，增强纠缠
            if q_res == rl_res:
                self.entanglement_matrix[q_res, :] *= 1.02
                self.entanglement_matrix[:, q_res] *= 1.02
            elif abs(q_res - rl_res) <= 2:  # 相邻
                self.entanglement_matrix[q_res, rl_res] *= 1.01
                self.entanglement_matrix[rl_res, q_res] *= 1.01

            # 如果测量时间很近，也增强纠缠
            if len(common_steps) > 1 and s - common_steps[-2] < 3:
                self.entanglement_matrix[q_res, rl_res] *= 1.005

        # 重新归一化，防止爆炸
        self.entanglement_matrix = self.entanglement_matrix / self.entanglement_matrix.max()

    def train_translators(self, current_step):
        """
        按时间对齐训练翻译器
        这才是真正的视角间学习！
        """
        if len(self.q_measurements) < 50 or len(self.rl_measurements) < 50:
            return

        # 构建时间索引
        q_dict = {step: (res, feat) for step, res, feat in self.q_measurements}
        rl_dict = {step: (res, feat) for step, res, feat in self.rl_measurements}

        # 找出共同的时间点
        common_steps = sorted(set(q_dict.keys()) & set(rl_dict.keys()))
        if len(common_steps) < 30:
            return

        # 用最近的30个共同时间点
        recent_steps = common_steps[-30:]

        # ===== 量子→RL 训练 =====
        q_features = torch.stack([q_dict[step][1] for step in recent_steps])
        rl_targets = torch.tensor([rl_dict[step][0] for step in recent_steps])

        rl_pred_log_probs = self.translator_q_to_rl(q_features)
        loss_q2r = nn.NLLLoss()(rl_pred_log_probs, rl_targets)

        # ===== RL→量子 训练 =====
        rl_features = torch.stack([rl_dict[step][1] for step in recent_steps])
        q_targets = torch.tensor([q_dict[step][0] for step in recent_steps])

        q_pred_log_probs = self.translator_rl_to_q(rl_features)
        loss_r2q = nn.NLLLoss()(q_pred_log_probs, q_targets)

        # 总损失
        total_loss = loss_q2r + loss_r2q

        # 反向传播
        self.translator_optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪，防止不稳定
        torch.nn.utils.clip_grad_norm_(
            list(self.translator_q_to_rl.parameters()) +
            list(self.translator_rl_to_q.parameters()),
            max_norm=1.0
        )
        self.translator_optimizer.step()

        # ===== 计算预测准确率 =====
        with torch.no_grad():
            # 量子→RL 准确率
            rl_pred = rl_pred_log_probs.exp()
            rl_pred_classes = rl_pred.argmax(dim=-1)
            acc_q2r = (rl_pred_classes == rl_targets).float().mean().item()

            # RL→量子 准确率
            q_pred = q_pred_log_probs.exp()
            q_pred_classes = q_pred.argmax(dim=-1)
            acc_r2q = (q_pred_classes == q_targets).float().mean().item()

            self.prediction_accuracy = (acc_q2r + acc_r2q) / 2

        # ===== 计算互信息 =====
        self._compute_mutual_info(recent_steps, q_dict, rl_dict)

    def _compute_mutual_info(self, steps, q_dict, rl_dict):
        """基于对齐步骤的互信息"""
        if len(steps) < 20:
            return

        q_vals = [q_dict[step][0] for step in steps]
        rl_vals = [rl_dict[step][0] for step in steps]

        # 构建联合分布
        joint = np.zeros((self.n_nodes, self.n_nodes))
        for q, r in zip(q_vals, rl_vals):
            joint[q, r] += 1
        joint = joint / (joint.sum() + 1e-10)

        # 边缘分布
        p_q = joint.sum(axis=1)
        p_r = joint.sum(axis=0)

        # 计算互信息
        mi = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if joint[i, j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (p_q[i] * p_r[j] + 1e-10))

        self.mutual_info = mi / np.log2(self.n_nodes)

    def update_self_awareness(self):
        """
        真正的自我认知更新：
        基于视角间预测准确率和互信息
        """
        # 预测准确率权重
        acc_weight = 0.7  # 增加准确率权重

        # 互信息权重
        mi_weight = 0.3

        # 计算新的自我认知
        new_awareness = acc_weight * self.prediction_accuracy + mi_weight * self.mutual_info

        # 滑动平均（让学习更平滑）
        self.self_awareness = 0.95 * self.self_awareness + 0.05 * new_awareness

        return self.self_awareness

    def compute_entanglement_entropy(self):
        """计算系统的整体纠缠熵（用于对比）"""
        s = torch.linalg.svdvals(self.entanglement_matrix)
        s = s / (s.sum() + 1e-10)
        entropy = -torch.sum(s * torch.log2(s + 1e-10)).item()
        return entropy / np.log2(self.n_nodes)


# ====================
# 2. 两个视角
# ====================

class QuantumPerspective:
    def __init__(self, system):
        self.system = system
        self.position = 'root'
        self.history = deque(maxlen=100)

    def step(self, step):
        base, result, features = self.system.measure_from_perspective('quantum', step)
        self.position = self.system.idx_to_node[result]
        self.history.append(result)
        return result, features

    def get_pc1(self):
        idx = self.system.node_to_idx[self.position]
        return -self.system.lifetime[idx].item() / 50 + 1


class RLPerspective:
    def __init__(self, system):
        self.system = system
        self.position = np.random.choice(['L1', 'L2', 'L3', 'L4', 'L5'])
        self.history = deque(maxlen=100)

    def step(self, step):
        base, result, features = self.system.measure_from_perspective('rl', step)
        self.position = self.system.idx_to_node[result]
        self.history.append(result)
        return result, features

    def get_pc1(self):
        idx = self.system.node_to_idx[self.position]
        return -self.system.lifetime[idx].item() / 50 + 1


# ====================
# 3. 运行实验
# ====================

def run_experiment(steps=1000):
    system = QuantumTreeSystem()
    q_view = QuantumPerspective(system)
    rl_view = RLPerspective(system)

    history = {
        'pc1': [],
        'self_awareness': [],
        'prediction_accuracy': [],
        'mutual_info': [],
        'entanglement_entropy': [],
        'q_pos': [],
        'rl_pos': []
    }

    for step in range(steps):
        # 两个视角测量（带时间戳）
        q_result, q_features = q_view.step(step)
        rl_result, rl_features = rl_view.step(step)

        # 记录位置
        history['q_pos'].append(q_result)
        history['rl_pos'].append(rl_result)

        # 计算PC1
        pc1 = q_view.get_pc1()
        history['pc1'].append(pc1)

        # ===== 关键：每5步更新一次 =====
        if step > 100 and step % 5 == 0:
            # 更新纠缠（量子演化）
            system.update_entanglement(step)

            # 训练翻译器
            system.train_translators(step)

        # 更新自我认知
        self_awareness = system.update_self_awareness()
        history['self_awareness'].append(self_awareness)

        # 记录其他指标
        history['prediction_accuracy'].append(system.prediction_accuracy)
        history['mutual_info'].append(system.mutual_info)
        history['entanglement_entropy'].append(system.compute_entanglement_entropy())

        if step % 100 == 0:
            print(f"Step {step}: PC1={pc1:.2f}, "
                  f"SelfAware={self_awareness:.3f}, "
                  f"Acc={system.prediction_accuracy:.3f}, "
                  f"MI={system.mutual_info:.3f}, "
                  f"Entropy={history['entanglement_entropy'][-1]:.3f}")

    return history


# ====================
# 4. 可视化
# ====================

def plot_results(history):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    steps = range(len(history['pc1']))

    # PC1 和 SelfAwareness
    ax1 = axes[0, 0]
    ax1.plot(steps, history['pc1'], 'b-', alpha=0.5, label='PC1')
    ax1.set_ylabel('PC1', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(steps, history['self_awareness'], 'r-', linewidth=2, label='SelfAware')
    ax2.set_ylabel('SelfAwareness', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim([0, 1])
    ax1.set_title('PC1 vs authentic self-awareness')

    # 预测准确率
    axes[0, 1].plot(steps, history['prediction_accuracy'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Prediction Accuracy')
    axes[0, 1].set_title('view prediction accuracy')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].axhline(y=0.111, color='gray', linestyle='--', label='随机基准')
    axes[0, 1].legend()

    # 互信息
    axes[0, 2].plot(steps, history['mutual_info'], 'purple', linewidth=2)
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Mutual Information')
    axes[0, 2].set_title('inter-view mutual information')
    axes[0, 2].set_ylim([0, 1])

    # 纠缠熵
    axes[1, 0].plot(steps, history['entanglement_entropy'], 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Entanglement Entropy')
    axes[1, 0].set_title('total entanglement entropy of the system')

    # 自我认知 vs 预测准确率
    axes[1, 1].scatter(history['prediction_accuracy'], history['self_awareness'],
                       c=steps, cmap='viridis', alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Prediction Accuracy')
    axes[1, 1].set_ylabel('Self Awareness')
    axes[1, 1].set_title('self-cognition vs prediction accuracy')

    # 位置分布
    axes[1, 2].hist(history['q_pos'], bins=9, alpha=0.5, label='Quantum')
    axes[1, 2].hist(history['rl_pos'], bins=9, alpha=0.5, label='RL')
    axes[1, 2].set_xlabel('Node')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('viewpoint position distribution')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()


# ====================
# 运行！
# ====================

if __name__ == "__main__":
    print("=" * 70)
    print("量子树系统：时间对齐 + 动态纠缠 + 密集训练")
    print("SelfAware 现在基于真正的时间对齐预测")
    print("=" * 70)

    history = run_experiment(steps=1000)

    print("\n实验结束！")
    print(f"最终自我认知水平: {history['self_awareness'][-1]:.3f}")
    print(f"最后100步平均预测准确率: {np.mean(history['prediction_accuracy'][-100:]):.3f}")
    print(f"最后100步平均互信息: {np.mean(history['mutual_info'][-100:]):.3f}")
    print(f"最后100步平均纠缠熵: {np.mean(history['entanglement_entropy'][-100:]):.3f}")

    # 检查自我认知是否真的依赖于视角间交流
    final_corr = np.corrcoef(
        history['self_awareness'][-200:],
        history['prediction_accuracy'][-200:]
    )[0, 1]
    print(f"自我认知 vs 预测准确率 相关性: {final_corr:.3f}")

    plot_results(history)