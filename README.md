# Q-Learning on FrozenLake from Scratch

A **tabular Q-Learning implementation from scratch** that trains an agent to navigate the FrozenLake environment using **reinforcement learning**, **epsilon-greedy exploration**, and the **Q-learning update rule**.

The project implements the complete learning pipeline, from initializing the Q-table and selecting actions to training, extracting a greedy policy, and evaluating the trained agent.

## Project Overview

FrozenLake is a reinforcement learning environment where an agent must navigate a grid and reach the goal while avoiding holes.

The agent learns through trial and error:

**State → Action → Reward → Next State → Q-value Update**

Instead of using a pre-built Q-learning implementation, the core algorithm is implemented manually to understand how each component works.

## Key Concepts

**Reinforcement Learning**

**Q-Learning**

**Q-Table**

**Temporal Difference (TD) Learning**

**Epsilon-Greedy Exploration**

**Exploration vs. Exploitation**

**Discount Factor**

**Learning Rate**

**Greedy Policy**

**Training Episodes**

**Success Rate Evaluation**

## Implementation

1. **`init_q_table`**: Initializes the Q-table.
2. **`max_q_value`**: Finds the maximum Q-value for a state.
3. **`greedy_action`**: Selects the action with the highest Q-value.
4. **`sample_random_action`**: Selects a random action for exploration.
5. **`should_explore`**: Determines whether the agent explores or exploits.
6. **`epsilon_greedy_action`**: Implements epsilon-greedy action selection.
7. **`decay_epsilon`**: Gradually reduces exploration over training.
8. **`td_target`**: Calculates the Temporal Difference target.
9. **`td_error`**: Calculates the TD error.
10. **`q_learning_update`**: Updates the Q-table using the Q-learning rule.
11. **`interaction_step`**: Handles one environment interaction.
12. **`run_training_episode`**: Runs a complete training episode.
13. **`train_q_learning`**: Trains the agent over multiple episodes.
14. **`extract_greedy_policy`**: Extracts the learned policy from the Q-table.
15. **`run_greedy_episode`**: Runs the agent using the learned greedy policy.
16. **`evaluate_success_rate`**: Measures how consistently the trained agent reaches the goal.

## Q-Learning

The agent updates its Q-values using the Temporal Difference target:

**Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]**

Where:

**α** = learning rate

**γ** = discount factor

**r** = reward received

**s** = current state

**a** = selected action

**s'** = next state

Over repeated interactions with the environment, the Q-table gradually learns which actions are more valuable in each state.

## Exploration Strategy

The agent uses **epsilon-greedy exploration**.

With probability **ε**, it chooses a random action.

With probability **1 − ε**, it chooses the best-known action.

The value of ε decays during training, allowing the agent to gradually transition from exploration toward exploitation.

## Project Structure

```text
q-learning-on-frozenlake/
│
├── scaffold.py
├── README.md
└── ...
```

## How to Run

Clone the repository and run:

```bash
python scaffold.py
```

Follow the project steps to complete and execute the Q-learning pipeline.

## What This Project Demonstrates

This project focuses on understanding **how Q-learning works internally**, rather than relying on a high-level reinforcement learning library.

By implementing the individual components, it demonstrates the complete flow:

```text
Environment
     ↓
State
     ↓
Epsilon-Greedy Action
     ↓
Reward + Next State
     ↓
TD Target
     ↓
TD Error
     ↓
Q-Table Update
     ↓
Improved Policy
```

## Learning Outcome

This project provides a practical understanding of:

**How an agent learns without labeled training data**

**How Q-values represent expected future rewards**

**Why exploration is necessary**

**How the Q-table changes during training**

**How a learned policy is extracted**

**How reinforcement learning performance can be evaluated**

## Acknowledgements

Built as a hands-on implementation based on the **Deep-ML** reinforcement learning challenge.

## Future Improvements

**Visualizing the learned Q-table**

**Plotting success rate across training episodes**

**Comparing different learning rates and discount factors**

**Testing different epsilon-decay strategies**

**Extending the implementation to larger environments**

**Implementing SARSA for comparison with Q-learning**
