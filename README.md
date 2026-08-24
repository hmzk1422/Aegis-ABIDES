# ABIDES: Agent-Based Interactive Discrete Event Simulation environment

> ABIDES is an Agent-Based Interactive Discrete Event Simulation environment. ABIDES is designed from the ground up to support AI agent research in market applications. While simulations are certainly available within trading firms for their own internal use, there are no broadly available high-fidelity market simulation environments. We hope that the availability of such a platform will facilitate AI research in this important area. ABIDES currently enables the simulation of tens of thousands of trading agents interacting with an exchange agent to facilitate transactions. It supports configurable pairwise network latencies between each individual agent as well as the exchange. Our simulator's message-based design is modeled after NASDAQ's published equity trading protocols ITCH and OUCH. 

Please see our arXiv paper for preliminary documentation:

https://arxiv.org/abs/1904.12066

Please see the wiki for tutorials and example configurations:

https://github.com/abides-sim/abides/wiki

## Quickstart
```
mkdir project
cd project

git clone https://github.com/abides-sim/abides.git
cd abides
pip install -r requirements.txt
```

# Multi-objective MARL for Game-Theoretic Market Simulation and Edge Sustainability
## Abstract
Traditional Agent-Based Models (ABMs) of financial markets rely heavily on rule-based, deterministic heuristics to model trading behaviors. While effective at replicating macro-level stylized facts, these fixed-rule frameworks lack the capacity for strategic adaptation, rendering them vulnerable to non-stationary structural shifts and regime changes. This paper formalizes the transformation of a discrete-event simulator—specifically the Agent-Based Interactive Discrete Event Simulator (ABIDES)—into a Markov Game (Partially Observable Stochastic Game) utilizing Multi-Agent Reinforcement Learning (MARL). By replacing static heuristic policies with parameterizable neural networks trained via actor-critic architectures, we model the emergent game-theoretic dynamics of competitive market making, inventory risk mitigation, and adverse selection survival under adversarial order flow.
## 1. Introduction
### 1.1 Limitations of Classical Heuristic Simulations
Simulating limit order book (LOB) dynamics via multi-agent systems requires modeling the interaction of heterogeneous market participants. In baseline configurations of simulators such as ABIDES, agents operate according to hardcoded behavioral functions:Noise Agents: Ingest random Poisson arrival rates to submit liquidity-taking market orders.Value Agents: Estimate a latent fundamental value process $V_t$ via Bayesian updating and execute orders based on perceived mispricing.Momentum Agents: Utilize moving-average cross-over rules ($p_{20} vs. p_{50}$) to determine directional bias.Naive Market Makers: Quote fixed-spread limit orders relative to an exponentially weighted moving average (EWMA) of historical volume or bid-ask spreads.While these rule-based configurations yield empirical artifacts such as fat-tailed return distributions and volatility clustering, they exhibit structural fragility. The deterministic policy mappings $\pi_i: \mathcal{S} \to \mathcal{A}$ are static. Consequently, an agent cannot adapt if its counterparties modify their execution schedules, leading to deterministic failure modes when exposed to regime shifts or toxic flow.

### 1.2 The Learning Paradigm Shift
To create an environment where trading strategies remain robust, policies must be endogenized through optimization. By modeling agents as reinforcement learning practitioners, the simulation transitions from a static environment to a non-stationary, competitive game-theoretic environment.

## 2. Mathematical Formalization
We extend the discrete-event matching engine into a multi-agent generalization of a Markov Decision Process (MDP), defined as a Partially Observable Stochastic Game (POSG) tuple:
$$\mathcal{M} = \left\langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}_{i \in \mathcal{N}}, \{\mathcal{O}_i\}_{i \in \mathcal{N}}, \mathcal{P}, \{\mathcal{R}_i\}_{i \in \mathcal{N}}, \gamma \right\rangle$$

- $\mathcal{N} = \{1, 2, \dots, N\}$ represents the set of autonomous learning agents (e.g., market makers, execution algorithms).
- $\mathcal{S}$ is the global state space of the financial market (complete order book tree, pending message queues, latent fundamental value).
- $\mathcal{A}_i$ is the action space available to agent $i$.
- $\mathcal{O}_i$ denotes the local observation space available to agent $i$ via public order book feeds (e.g., L1/L2 LOB quotes, execution notifications).
- $\mathcal{P}: \mathcal{S} \times \mathbf{\mathcal{A}} \to \Delta(\mathcal{S})$ is the environmental transition probability function, governed by the discrete-event matching kernel, where $\mathbf{\mathcal{A}} = \prod_{i \in \mathcal{N}} \mathcal{A}_i$ is the joint action space.
- $\mathcal{R}_i: \mathcal{S} \times \mathbf{\mathcal{A}} \times \mathcal{S} \to \mathbb{R}$ is the scalar reward function for agent $i$.
- $\gamma \in [0, 1)$ is the temporal discount factor.

### 2.1 The Non-Stationarity Problem & Game Theory
In a single-agent MDP, the transition probability $\mathcal{P}(s' \mid s, a_i)$ is stationary. In a MARL framework, the transition dynamics from the perspective of agent $i$ depend on the joint action vector $\mathbf{a}_{-i}$ of all other competing agents:$$\mathcal{P}_i(s' \mid s, a_i) = \sum_{\mathbf{a}_{-i}} \mathcal{P}(s' \mid s, a_i, \mathbf{a}_{-i}) \prod_{j \neq i} \pi_j(a_j \mid o_j)$$As competing agents update their policy parameters $\theta_j$ via gradient descent, the environment's transition distribution shifts over time:$$\mathcal{P}_i^{\theta_{-i}}(s' \mid s, a_i) \neq \mathcal{P}_i^{\theta_{-i}'}(s' \mid s, a_i)$$This non-stationarity invalidates standard single-agent convergence guarantees. The multi-agent learning system searches for a Stochastic Game Nash Equilibrium—a joint policy profile $\mathbf{\pi}^* = (\pi_1^*, \dots, \pi_N^*)$ such that no agent can unilaterally increase its expected discounted return:$$\mathbb{E}_{\mathbf{\pi}^*}\left[ \sum_{t=0}^{\infty} \gamma^t \mathcal{R}_i(s_t, \mathbf{a}_t) \right] \geq \mathbb{E}_{(\pi_i, \mathbf{\pi}_{-i}^*)}\left[ \sum_{t=0}^{\infty} \gamma^t \mathcal{R}_i(s_t, a_{i,t}, \mathbf{a}_{-i,t}^*) \right], \quad \forall \pi_i \in \Pi_i, \forall i \in \mathcal{N}$$

## 3. System Architecture: ABIDES as a Physics Kernel
ABIDES serves as the underlying microstructure physics engine. It enforces price-time priority execution, models message delays (network latency), and maintains the continuous double auction (CDA) matching state.

### 3.1 State Space Design 
($\mathcal{O}_i$)At step $t$, the state tensor $\mathbf{x}_t \in \mathcal{O}_i$ supplied to the reinforcement learning policy encompasses both market-level and agent-specific attributes:Order Book Depth Vector: Normalized volume and prices up to $K$ depth levels:$$\mathbf{D}_t = \left[ \{p_{b,t}^{(k)}, v_{b,t}^{(k)}, p_{a,t}^{(k)}, v_{a,t}^{(k)}\}_{k=1}^K \right]$$
### Microstructure Indicators:
- Bid-Ask Spread: $S_t = p_{a,t}^{(1)} - p_{b,t}^{(1)}$
- Order Book Imbalance (OBI):$$OBI_t = \frac{v_{b,t}^{(1)} - v_{a,t}^{(1)}}{v_{b,t}^{(1)} + v_{a,t}^{(1)}} \in [-1, 1]$$
- Micro-Price Drift:$$P_t^{\text{micro}} = \frac{v_{b,t}^{(1)} p_{a,t}^{(1)} + v_{a,t}^{(1)} p_{b,t}^{(1)}}{v_{b,t}^{(1)} + v_{a,t}^{(1)}} - P_t^{\text{mid}}$$
- Agent Portfolio State:
    - Inventory ($I_t$): Current net physical position in the asset.
    - Unrealized PnL ($uPnL_t$): Mark-to-market valuation relative to initial capital.
### 3.2 Action Space Design 
($\mathcal{A}_i$)For a specialized Market Maker agent, the action space dictates liquidity provision relative to the prevailing mid-price $P_t^{\text{mid}}$:$$\mathcal{A}_i = \left\{ \delta_{\text{bid}}, \delta_{\text{ask}}, q_{\text{bid}}, q_{\text{ask}}, \phi \right\}$$Where:
- $\delta_{\text{bid}}, \delta_{\text{ask}} \in \mathbb{R}^+$ represent distance offsets from $P_t^{\text{mid}}$ for placing limit orders.
- $q_{\text{bid}}, q_{\text{ask}} \in \mathbb{Z}^+$ specify order volume sizing.
- $\phi \in \{0, 1\}$ represents an instruction to clear open orders (cancellation action).
### 3.3 Objective & Reward Function Formulation
Standard profit maximization is insufficient due to inventory risk. In classical continuous-time models (e.g., Avellaneda-Stoikov), the market maker's utility is modeled with constant absolute risk aversion (CARA). We translate this into a discrete-step reward function that penalizes inventory accumulation and volatility exposure:$$\mathcal{R}_{i,t} = \Delta \text{PnL}_t - \gamma_{\text{inv}} \cdot I_t^2 \cdot \sigma_t^2 - \gamma_{\text{exec}} \cdot \mathbf{1}_{\{\text{toxic fill}\}}$$Where:
- $\Delta \text{PnL}_t = (C_t + I_t \cdot P_t^{\text{mid}}) - (C_{t-1} + I_{t-1} \cdot P_{t-1}^{\text{mid}})$ represents step-over-step mark-to-market wealth changes ($C_t$ is cash balance).
- $\gamma_{\text{inv}} \cdot I_t^2 \cdot \sigma_t^2$ is an inventory risk penalty scaled by local market volatility $\sigma_t^2$.
- $\gamma_{\text{exec}}$ imposes an explicit penalty for adverse selection (getting filled immediately prior to a large directional price move).
## Classical vs MARL Model
| Attribute | Classical Heuristic (e.g., POV / Basic ABIDES) | Closed-Form Mathematical (Avellaneda-Stoikov) | MARL Adaptive Policy (Proposed) |
| :--- | :--- | :--- | :--- |
| **Strategy Source** | Hardcoded conditional rules | Closed-form calculus solution under assumptions | Deep Neural Network ($\pi_\theta(a \mid s)$) optimized via RL |
| **Market Assumptions** | Assumes predictable order flow patterns | Assumes mid-price follows Arithmetic Brownian Motion | Non-parametric; learns directly from empirical simulator interactions |
| **Adaptability** | Zero adaptability; vulnerable to toxic flow | Fixed parameter setup; degrades under heavy regime shifts | Continuous adaptation via multi-agent competitive interaction |
| **Microstructure Aware** | Low (uses simple lagged indicators) | Moderate (incorporates intensity $\lambda$ and volatility $\sigma$) | High (ingests full L2 order book depth, OBI, and latency) |
| **Game Dynamics** | Single-agent execution in static noise | Single-agent optimization against stochastic environment | Multi-agent equilibrium discovery (Stochastic Game) |

## 5. Sim-to-Real Vulnerabilities & Methodological Edge
### 5.1 The Sim-to-Real Gap & Reward Exploitation
While MARL yields highly adaptive strategies, it introduces failure modes specific to deep reinforcement learning in artificial environments:
1. Overfitting to Simulator Physics: An RL policy will exploit artifacts in the simulator (e.g., discretization errors in queue priority logic or latency approximations) to generate artificial alpha that collapses in real markets.

2. Adverse Selection Collusion: In multi-agent settings, policy updates can converge to unintended local optima (e.g., implicit collusion or complete withdrawal of liquidity during simulated shocks).

### 5.2 Edge Sustainability via Non-Parametric Learning
The fundamental value of this approach lies in eliminating rigid assumptions about market behavior. By training an agent inside an environment populated by both heuristic background traders (noise, value, momentum) and competing RL agents, the policy organically discovers strategies that:
- Skew bid/ask quotes based on current inventory levels (reconstructing Avellaneda-Stoikov dynamics without explicit mathematical programming).
- Widen spreads dynamically during periods of high order book imbalance.
- Minimize execution latency hazards by controlling quote duration.

By transforming ABIDES from a static heuristic generator into an adversarial multi-agent game, we construct a platform for training trading algorithms capable of learning, surviving, and maintaining a statistically defensible edge within complex market environments.