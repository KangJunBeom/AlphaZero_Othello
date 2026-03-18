"""
============================================================
AlphaZero-style Othello AI
============================================================

"""

# Colab 설치 (필요시 주석 해제)

import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import matplotlib.pyplot as plt
import time

# 전역 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"학습에 사용되는 디바이스: {DEVICE}")

if torch.cuda.is_available():
    # 입력 크기가 고정되어 있으므로 cuDNN이 최적 커널을 캐싱하도록 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled   = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# Part 1. 오델로 환경 (numpy 기반)

class OthelloEnv:
    DIRECTIONS = [(-1,-1),(-1,0),(-1,1),
                  ( 0,-1),        ( 0,1),
                  ( 1,-1),( 1,0),( 1,1)]

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.board[3][3] = 1; self.board[3][4] = 2
        self.board[4][3] = 2; self.board[4][4] = 1
        self.current_player = 1
        self.done   = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        s = np.zeros((3, 8, 8), dtype=np.float32)
        s[0] = (self.board == self.current_player).astype(np.float32)
        s[1] = (self.board == (3 - self.current_player)).astype(np.float32)
        s[2] = float(self.current_player == 1)
        return s

    def get_valid_moves(self, board=None, player=None):
        if board  is None: board  = self.board
        if player is None: player = self.current_player
        opponent = 3 - player
        valid = []
        for r in range(8):
            for c in range(8):
                if board[r][c] != 0:
                    continue
                for dr, dc in self.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    found = False
                    while 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opponent:
                        nr += dr; nc += dc; found = True
                    if found and 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == player:
                        valid.append((r, c)); break
        return valid

    def apply_move(self, move, board=None, player=None):
        inplace = (board is None)
        if board  is None: board  = self.board
        if player is None: player = self.current_player
        opponent = 3 - player
        if not inplace: board = board.copy()

        r, c = move
        board[r][c] = player
        for dr, dc in self.DIRECTIONS:
            to_flip = []
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opponent:
                to_flip.append((nr, nc)); nr += dr; nc += dc
            if to_flip and 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == player:
                for fr, fc in to_flip:
                    board[fr][fc] = player

        if inplace:
            next_p = opponent
            if not self.get_valid_moves(board, next_p):
                next_p = player
                if not self.get_valid_moves(board, next_p):
                    self._finish(board); return
            self.current_player = next_p
        else:
            return board

    def _finish(self, board):
        black = int(np.sum(board == 1))
        white = int(np.sum(board == 2))
        self.done = True
        self.winner = 1 if black > white else (2 if white > black else 0)

    def step(self, move):
        if self.done: raise RuntimeError("Game is already over.")
        valid = self.get_valid_moves()
        if not valid:
            self.current_player = 3 - self.current_player
            if not self.get_valid_moves():
                self._finish(self.board)
        if not self.done and move in valid:
            self.apply_move(move)
        if not self.done and int(np.sum(self.board != 0)) == 64:
            self._finish(self.board)
        reward = 0
        if self.done:
            reward = 1 if self.winner == 1 else (-1 if self.winner == 2 else 0)
        return self.get_state(), reward, self.done

    def get_valid_mask(self):
        mask = np.zeros(64, dtype=np.float32)
        for r, c in self.get_valid_moves():
            mask[r * 8 + c] = 1.0
        return mask

    def clone(self):
        env = OthelloEnv()
        env.board          = self.board.copy()
        env.current_player = self.current_player
        env.done           = self.done
        env.winner         = self.winner
        return env


# Part 2. CNN 정책망 + 가치망

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return F.relu(self.net(x) + x, inplace=True)


class OthelloNet(nn.Module):
    def __init__(self, num_res_blocks=4, channels=64):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks  = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(32 * 8 * 8, 64),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        return self.policy_head(x), self.value_head(x)

    @torch.no_grad()
    def predict(self, state: np.ndarray):
        self.eval()
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE, non_blocking=True)
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits, value = self(s)
        policy = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]
        return policy, float(value.float().cpu().numpy()[0][0])

    @torch.no_grad()
    def predict_batch(self, states: np.ndarray):
        """
        [최적화 1] 배치 추론
        states: (N, 3, 8, 8) numpy → policy (N, 64), value (N,) numpy
        MCTS 리프 평가를 개별 호출 대신 배치로 묶어 GPU 활용률 향상
        """
        self.eval()
        s = torch.tensor(states, dtype=torch.float32).to(DEVICE, non_blocking=True)
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits, value = self(s)
        policy = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        value  = value.float().squeeze(-1).cpu().numpy()
        return policy, value

# Part 3. MCTS(Monte-Carlo Tree Search)

class MCTSNode:
    __slots__ = ["parent", "children", "visit", "value_sum", "prior"]

    def __init__(self, prior=0.0, parent=None):
        self.parent    = parent
        self.children  = {}
        self.visit     = 0
        self.value_sum = 0.0
        self.prior     = prior

    def is_leaf(self):   return len(self.children) == 0
    def Q(self):         return self.value_sum / self.visit if self.visit else 0.0
    def U(self, c=1.5):  return c * self.prior * (self.parent.visit ** 0.5) / (1 + self.visit)

    def select_child(self, c_puct=1.5):
        return max(self.children.items(), key=lambda kv: kv[1].Q() + kv[1].U(c_puct))

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(prior=prob, parent=self)

    def update_recur(self, value):
        self.visit     += 1
        self.value_sum += value
        if self.parent:
            self.parent.update_recur(-value)


class MCTS:
    """
    [최적화 2] 배치 리프 평가 MCTS
    n_simul번의 시뮬레이션을 leaf_batch_size 단위로 묶어
    신경망을 배치로 호출 → GPU 유휴 시간 감소
    """
    def __init__(self, net, n_simul=200, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25,
                 leaf_batch_size=8):
        self.net              = net
        self.n_simul          = n_simul
        self.c_puct           = c_puct
        self.dirichlet_alpha  = dirichlet_alpha
        self.dirichlet_eps    = dirichlet_eps
        self.leaf_batch_size  = leaf_batch_size

    def run(self, env):
        root = MCTSNode()
        policy, _ = self.net.predict(env.get_state())
        self._expand(root, env, policy, add_dirichlet=True)

        sims_done = 0
        while sims_done < self.n_simul:
            batch_size = min(self.leaf_batch_size, self.n_simul - sims_done)
            leaf_nodes = []
            leaf_envs  = []

            for _ in range(batch_size):
                node    = root
                sim_env = env.clone()

                # Selection
                while not node.is_leaf():
                    action, node = node.select_child(self.c_puct)
                    sim_env.step(action)
                    if sim_env.done: break

                leaf_nodes.append(node)
                leaf_envs.append(sim_env)

            states      = np.array([e.get_state() for e in leaf_envs])
            policies, values = self.net.predict_batch(states)

            for i, (node, sim_env) in enumerate(zip(leaf_nodes, leaf_envs)):
                if sim_env.done:
                    if sim_env.winner == 0:
                        leaf_value = 0.0
                    else:
                        leaf_value = 1.0 if sim_env.winner == sim_env.current_player else -1.0
                else:
                    self._expand(node, sim_env, policies[i], add_dirichlet=False)
                    leaf_value = float(values[i])

                node.update_recur(leaf_value)

            sims_done += batch_size

        visits = np.zeros(64, dtype=np.float32)
        for action, child in root.children.items():
            visits[action[0] * 8 + action[1]] = child.visit

        valid_mask = env.get_valid_mask()
        if visits.sum() == 0:
            visits = valid_mask / (valid_mask.sum() + 1e-8)
        else:
            visits /= visits.sum()
        return visits

    def _expand(self, node, env, policy, add_dirichlet):
        valid_mask    = env.get_valid_mask()
        masked_policy = policy * valid_mask
        total         = masked_policy.sum()
        if total > 0:
            masked_policy /= total
        else:
            masked_policy = valid_mask / (valid_mask.sum() + 1e-8)

        if add_dirichlet:
            valid_idx = np.where(valid_mask > 0)[0]
            if len(valid_idx) > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_idx))
                for i, idx in enumerate(valid_idx):
                    masked_policy[idx] = ((1 - self.dirichlet_eps) * masked_policy[idx]
                                          + self.dirichlet_eps * noise[i])

        action_priors = [((idx // 8, idx % 8), float(masked_policy[idx]))
                         for idx in range(64) if valid_mask[idx] > 0]
        node.expand(action_priors)


# Part 4. Self-Play (자가 대국)

def self_play_game(net, n_simul=200, temp_threshold=10, leaf_batch_size=8):
    mcts = MCTS(net, n_simul=n_simul, leaf_batch_size=leaf_batch_size)
    env  = OthelloEnv()
    data = []
    step = 0

    while not env.done:
        valid = env.get_valid_moves()
        if not valid:
            env.current_player = 3 - env.current_player
            if not env.get_valid_moves():
                env._finish(env.board)
            continue

        pi     = mcts.run(env)
        state  = env.get_state()

        if step < temp_threshold:
            valid_pi = pi.copy()
            s = valid_pi.sum()
            flat_idx = np.random.choice(64, p=valid_pi / s if s > 0 else valid_pi)
        else:
            flat_idx = int(np.argmax(pi))

        action = (flat_idx // 8, flat_idx % 8)
        data.append((state, pi, env.current_player))
        env.step(action)
        step += 1

    z_map = ({env.winner: 1.0, 3 - env.winner: -1.0} if env.winner != 0 else {})
    return [(s, p, z_map.get(pl, 0.0)) for s, p, pl in data]


# Part 5. 최적화 학습 루프

def build_dataloader(memory, batch_size):
    """
    [최적화 3] DataLoader + pin_memory
    replay buffer → TensorDataset → DataLoader
    pin_memory=True: CPU 메모리를 페이지 고정 → GPU 전송 속도 향상
    """
    sample = random.sample(memory, min(len(memory), batch_size * 8))
    states, pis, zs = zip(*sample)

    s_tensor  = torch.tensor(np.array(states), dtype=torch.float32)
    pi_tensor = torch.tensor(np.array(pis),    dtype=torch.float32)
    z_tensor  = torch.tensor(np.array(zs),     dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(s_tensor, pi_tensor, z_tensor)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        pin_memory  = torch.cuda.is_available(),   # [최적화 3]
        num_workers = 0,
    )
    return loader


def train_epoch(net, optimizer, scaler, loader):
    """
    [최적화 4] AMP (Automatic Mixed Precision)
    forward: float16으로 연산 → 처리량 약 2배
    backward: GradScaler로 underflow 방지
    """
    net.train()
    total_loss = p_loss_sum = v_loss_sum = 0.0
    steps = 0

    for s, pi, z in loader:
        # [최적화 5] non_blocking: CPU→GPU 전송을 비동기로 수행
        s   = s.to(DEVICE,  non_blocking=True)
        pi  = pi.to(DEVICE, non_blocking=True)
        z   = z.to(DEVICE,  non_blocking=True)

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits, value  = net(s)
            policy_loss    = -(pi * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            value_loss     = F.mse_loss(value, z)
            loss           = policy_loss + value_loss

        optimizer.zero_grad(set_to_none=True)   # [최적화 6] set_to_none: 메모리 효율 향상
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss.item()
        p_loss_sum  += policy_loss.item()
        v_loss_sum  += value_loss.item()
        steps       += 1

    return total_loss / steps, p_loss_sum / steps, v_loss_sum / steps


def train(
    n_iterations   = 30,
    n_self_play    = 20,
    n_simul        = 150,
    batch_size     = 256,
    leaf_batch_size = 8,
    lr             = 1e-3,
    replay_buffer  = 15000,
    num_res_blocks = 4,
    channels       = 64,
    save_path      = "othello_net_optimized.pth",
    resume_path    = None,
    plot_path      = "loss_curve.png"
):
    net = OthelloNet(num_res_blocks=num_res_blocks, channels=channels).to(DEVICE)
    if resume_path and os.path.exists(resume_path):
        state_dict = torch.load(resume_path, map_location=DEVICE)

        # torch.compile로 저장된 경우 _orig_mod. 접두사 제거
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }

        net.load_state_dict(state_dict)
        print(f"✓ 기존 모델 로드: {resume_path}")

    # [최적화 7] torch.compile (PyTorch 2.0+)
    # 신경망 연산 그래프를 컴파일해 반복 실행 시 커널 실행 오버헤드 감소
    if hasattr(torch, "compile"):
        try:
            net = torch.compile(net)
            print("✓ torch.compile 적용됨")
        except Exception as e:
            print(f"torch.compile 불가 ({e}), 스킵")

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.5)
    scaler    = GradScaler(enabled=torch.cuda.is_available())   # AMP scaler
    memory    = deque(maxlen=replay_buffer)
    history = {"loss": [], "policy_loss": [], "value_loss": []}  # ← 추가

    print("\n" + "=" * 60)
    print("  AlphaZero-style Othello AI")
    print(f"  Device       : {DEVICE}")
    print(f"  Iterations   : {n_iterations} | Self-play/iter: {n_self_play}")
    print(f"  MCTS simul   : {n_simul} | Leaf batch: {leaf_batch_size}")
    print(f"  Train batch  : {batch_size} | AMP: {torch.cuda.is_available()}")
    print("=" * 60)

    for iteration in range(1, n_iterations + 1):
        t0 = time.time()

        print(f"\n[Iter {iteration:02d}/{n_iterations}] Self-Play", end=" ", flush=True)
        for g in range(n_self_play):
            game_data = self_play_game(net, n_simul=n_simul,
                                       leaf_batch_size=leaf_batch_size)
            memory.extend(game_data)
            print(f".", end="", flush=True)
        print(f" {n_self_play}게임 완료 | 버퍼: {len(memory)}")

        if len(memory) < batch_size:
            print("  버퍼 부족, 학습 스킵")
            continue

        loader = build_dataloader(memory, batch_size)
        loss, p_loss, v_loss = train_epoch(net, optimizer, scaler, loader)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        print(f"  Loss: {loss:.4f} (P: {p_loss:.4f} | V: {v_loss:.4f}) | "
              f"LR: {lr_now:.5f} | {elapsed:.1f}s")

        history["loss"].append(loss)
        history["policy_loss"].append(p_loss)
        history["value_loss"].append(v_loss)

        if iteration % 10 == 0:
            torch.save(net.state_dict(), save_path)
            print(f"  ✓ 체크포인트 저장: {save_path}")

    torch.save(net.state_dict(), save_path)
    print(f"\n학습 완료. 최종 모델: {save_path}")

    if history["loss"]:
        iters = list(range(1, len(history["loss"]) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Training History", fontsize=14)
        for ax, key, color, title in zip(
            axes,
            ["loss", "policy_loss", "value_loss"],
            ["steelblue", "tomato", "seagreen"],
            ["Total Loss", "Policy Loss", "Value Loss"]
        ):
            ax.plot(iters, history[key], color=color, linewidth=2, marker="o", markersize=3)
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"  ✓ 학습 곡선 저장: {plot_path}")
    
    return net

# Part 6. 평가

def evaluate(net, n_games=20, n_simul=100):
    mcts = MCTS(net, n_simul=n_simul)
    wins = draws = losses = 0

    for _ in range(n_games):
        env = OthelloEnv()
        while not env.done:
            valid = env.get_valid_moves()
            if not valid:
                env.current_player = 3 - env.current_player
                if not env.get_valid_moves(): env._finish(env.board)
                continue

            if env.current_player == 1:
                pi     = mcts.run(env)
                action = (int(np.argmax(pi)) // 8, int(np.argmax(pi)) % 8)
            else:
                action = random.choice(valid)

            env.step(action)

        if   env.winner == 1: wins   += 1
        elif env.winner == 0: draws  += 1
        else:                 losses += 1

    print(f"\n[평가] {n_games}게임 | 학습AI(흑) vs 랜덤(백)")
    print(f"  승: {wins} | 무: {draws} | 패: {losses} | 승률: {wins/n_games*100:.1f}%")
    return wins, draws, losses

if __name__ == "__main__":
    trained_net = train(
        n_iterations    = 20,
        n_self_play     = 20,
        n_simul         = 400,
        batch_size      = 256,
        leaf_batch_size = 8,
        lr              = 1e-3,
        replay_buffer   = 5000,
        num_res_blocks  = 4,
        channels        = 64,
        save_path       = "othello_net2.pth",
        plot_path       = "loss_curve.png"
    )

    evaluate(trained_net, n_games=20, n_simul=100)


