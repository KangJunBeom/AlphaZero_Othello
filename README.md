# AlphaZero-style Othello AI <br> (PyTorch & CUDA Optimized)

> CUDA-accelerated self-play reinforcement learning for Othello,  
> inspired by DeepMind's AlphaZero architecture.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=flat-square&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

본 프로젝트는 **DeepMind의 AlphaZero 알고리즘을** 기반으로 구현된 **오델로 인공지능**입니다.  

2022년 건국대학교 컴퓨터공학부 오픈소스SW프로젝트 과목의 팀프로젝트 결과물을 발전시킨 코드로, 단순히 강화학습 모델을 학습시키는 것을 넘어, **PyTorch와 CUDA의 최신 기능들을 적극적으로 활용하여 학습 및 추론 파이프라인의 병목을 최소화하고 GPU 활용률을 극대화**하는 데 초점을 맞추었습니다.


## 1. 이론적 배경

이 AI는 인간의 기보 데이터 없이, 오직 강화학습의 **자가 대국(Self-Play)** 만을 통해 학습합니다. 

* **MCTS (Monte Carlo Tree Search)**  
무작위 롤아웃 대신 신경망의 예측값(Policy, Value)을 활용하여 탐색 트리를 효율적으로 확장합니다. 

* **Self-Play (자가 대국)**  
현재 버전의 신경망을 사용해 스스로 게임을 플레이하며, 이 과정에서 생성된 상태(State), 탐색 확률(MCTS Policy), 승패 결과(Value)를 수집하여 새로운 학습 데이터를 생성합니다.

* **Actor-Critic 아키텍처**  
하나의 이미지(보드 상태)를 보고, '어디에 두는 것이 좋은지(Policy)'와 '현재 보드 상황이 얼마나 유리한지(Value)'를 동시에 예측합니다.

---

## 2. 주요 AI 개념

* **Dual-Head ResNet 아키텍처**  
알파고 제로(AlphaGo Zero)에서 증명된 구조를 차용하여, 공통된 특징 추출 레이어(ResBlocks)를 통과한 후 정책망(Policy Head)과 가치망(Value Head)으로 나뉘어 예측을 수행합니다.

* **PUCT (Predictor + UCB applied to Trees)**  
MCTS의 Selection 단계에서 신경망이 예측한 사전 확률(Prior)과 방문 횟수를 결합하여, 유망한 노드를 탐색(Exploitation)하는 동시에 방문 횟수가 적은 노드를 탐험(Exploration)하도록 유도합니다.

* **노드 선택 기준 (PUCT)**

$$\text{score}(s, a) = Q(s,a) + U(s,a), \quad U(s,a) = c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

* **Dirichlet Noise (디리클레 노이즈)**  
루트 노드의 정책 확률에 노이즈를 주입하여, AI가 항상 같은 수만 두는 부분 최적화(Local Minima)에 빠지지 않고 다양한 오프닝을 탐험하도록 합니다.

* **루트 디리클레 노이즈 수식** — 탐색 다양성 확보:

$$P'(s,a) = (1-\varepsilon) \cdot P(s,a) + \varepsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha=0.3)$$

* **Temperature Scaling**  
초반 턴에는 방문 확률에 비례하여 무작위로 착수(Softmax-like)하여 다양성을 확보하고, 후반 턴에는 가장 방문 횟수가 많은 최적의 수를 선택(Argmax)하여 승리를 굳히는 전략을 사용합니다.

---

## 3. CUDA 및 파이프라인 최적화

이 프로젝트의 핵심은 MCTS와 딥러닝 추론/학습 과정에서 발생하는 병목을 해소한 것입니다.

* **AMP (Automatic Mixed Precision)**  
`torch.cuda.amp.autocast`와 `GradScaler`를 활용해 float16과 float32를 혼합 연산합니다. 모델의 파라미터 업데이트 안정성을 유지하면서도 GPU 메모리 사용량을 절반으로 줄이고, Tensor Core 연산 처리량을 비약적으로 향상시켰습니다.

* **배치 리프 평가 (Batched Leaf Evaluation in MCTS)**  
MCTS는 본질적으로 순차적 탐색 구조라 GPU를 비효율적으로 사용합니다. 이를 극복하기 위해 `leaf_batch_size` 단위로 시뮬레이션을 모은 후, 신경망 추론을 한 번의 배치(Batch)로 처리하여 GPU 유휴 시간을 크게 줄였습니다.

* **비동기 메모리 전송 (pin_memory & non_blocking)**  
`DataLoader`에 `pin_memory=True`를 설정하여 CPU의 페이지 고정 메모리를 할당받고, `to(non_blocking=True)`를 통해 CPU에서 GPU로의 텐서 복사를 비동기적으로 수행하여 연산과 데이터 전송을 오버랩시켰습니다.

* **torch.compile (PyTorch 2.0+)**  
신경망의 연산 그래프를 미리 JIT 컴파일하여, 파이썬 오버헤드를 줄이고 커널 실행을 퓨전(Fusion)하여 학습 속도를 가속화했습니다.

* **cuDNN Benchmark 자동 최적화**  
오델로 환경의 입력 크기(3x8x8)가 고정되어 있다는 점을 활용, `torch.backends.cudnn.benchmark = True`를 설정하여 하드웨어에 가장 최적화된 합성곱(Convolution) 커널 알고리즘을 자동 탐색하도록 했습니다.

---

## 4. 코드 구조 및 함수 설명

코드는 역할별로 명확히 분리되어 있습니다.

### Part 1. OthelloEnv (환경)
* `get_state()`  
보드의 현재 상태를 신경망이 이해할 수 있는 3채널(내 돌, 상대 돌, 차례) 텐서 형태로 변환합니다.

* `get_valid_moves()` / `apply_move()`  
오델로의 룰(돌 뒤집기 등)을 numpy 연산으로 구현하여 시뮬레이션을 수행합니다.

### Part 2. OthelloNet (신경망)
* `ResBlock`  
Residual Connection을 통해 기울기 소실 없이 깊은 네트워크 학습을 돕습니다.

* `predict_batch()`  
MCTS에서 전달된 다수의 상태(State) 텐서를 배치로 묶어 빠르게 정책과 가치를 병렬 추론합니다.

### Part 3. MCTS (탐색 알고리즘)
* `MCTSNode`  
트리의 각 노드 상태, 방문 횟수, 가치 합, 자식 노드 정보를 저장합니다 (`__slots__` 활용으로 메모리 최적화).

* `run()`  
루트 노드에서 시작해 트리를 확장하며, 배치 단위 평가를 통해 방문 확률 분포(Policy)를 반환합니다.

### Part 4. Self-Play (데이터 수집)
* `self_play_game()`  
환경과 MCTS를 연동하여 하나의 게임을 끝까지 플레이합니다. 게임 종료 후 승패 결과를 역추적하여 각 상태의 최종 리워드(Value)를 매핑합니다.

### Part 5. Train (학습 파이프라인)
* `build_dataloader()`  
수집된 Replay Buffer 데이터 중 미니배치를 샘플링하여 텐서로 변환합니다.

* `train_epoch()`  
손실 함수(Cross-Entropy for Policy, MSE for Value)를 계산하고 역전파를 수행합니다.

### Part 6. Evaluate (평가)
* `evaluate()`  
학습된 모델과 랜덤(또는 휴리스틱) 에이전트를 맞붙여 모델의 승률을 검증합니다.



## 🔬 핵심 하이퍼파라미터 (최적화 예정)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `num_res_blocks` | 4 | ResBlock 개수 |
| `channels` | 64 | CNN 채널 수 |
| `n_simul` | 150 | MCTS 시뮬레이션 횟수 |
| `leaf_batch_size` | 8 | 배치 리프 평가 크기 |
| `c_puct` | 1.5 | 탐색-활용 균형 계수 |
| `dirichlet_alpha` | 0.3 | 디리클레 노이즈 강도 |
| `dirichlet_eps` | 0.25 | 노이즈 혼합 비율 |
| `temp_threshold` | 10 | temperature 전환 시점 (수) |
| `replay_buffer` | 15,000 | 리플레이 버퍼 크기 |

---

## 5. 실행 환경 및 실행 방법

```
.
├── othello_alphazero_optimized.py   # 메인 구현 파일 (All-in-one)
│   ├── OthelloEnv                   # 오델로 게임 환경
│   ├── OthelloNet                   # CNN 정책망 + 가치망
│   ├── MCTS / MCTSNode              # 신경망 연동 MCTS
│   ├── self_play_game()             # Self-Play 데이터 생성
│   ├── train()                      # 학습 루프
│   └── evaluate()                   # 평가 (vs 랜덤 AI)
├── othello_net.pth                  # 학습된 모델 가중치
└── README.md
```

### 환경 요구 사항
* Python 3.9+
* PyTorch 2.0+ (CUDA 11.x 또는 12.x 권장)
* library set

### 실행 방법

```bash
# Google Colab 기준
1. 구글 코랩(Google Colab)을 열고 새 수첩을 생성합니다.
2. 상단 메뉴에서 가속기를 GPU (T4, V100, A100 등)로 설정합니다.
3. 소스 코드를 셀에 붙여넣고 실행합니다.
4. 학습이 완료되면 `othello_net_optimized.pth` 파일로 모델 가중치가 자동 저장됩니다.
```
```bash
# 로컬에서 실행 시
python main.py (파일명을 변경하고 실행해야 합니다)
```

## 📚 참고 문헌

- Silver, D. et al. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*. DeepMind.
- Silver, D. et al. (2016). *Mastering the game of Go with deep neural networks and tree search*. Nature.

