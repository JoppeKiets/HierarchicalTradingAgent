"""Feedback modules for the multi-agent trading pipeline.

Submodules:
  - feature_feedback:  Aggregates Analyst feature trust scores over time.
  - attention_prior:   Computes per-sub-model attention biases from
                       Critic/Executor feedback for walk-forward windows.
"""
