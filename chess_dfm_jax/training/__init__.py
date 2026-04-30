"""Training scaffolds."""

from .checkpoints import load_training_checkpoint, save_training_checkpoint
from .jepa import (
    JEPAConfig,
    LC0JEPA,
    TokenTransitionHead,
    build_synthetic_transition_batch,
    build_transition_batch,
    create_jepa_components,
    extract_train_state,
    restore_train_state,
    train_step,
    transition_jepa_loss,
)
from .joint_latent_sasa import (
    JointLatentSASAConfig,
    JointLatentSASAModel,
    create_joint_components,
    eval_joint_stage1_step,
    eval_joint_stage2_step,
    joint_contrastive_loss_fn,
    joint_stage1_loss_fn,
    joint_stage2_loss_fn,
    train_joint_stage1_step,
    train_joint_stage2_step,
)

__all__ = [
    "JEPAConfig",
    "JointLatentSASAConfig",
    "JointLatentSASAModel",
    "LC0JEPA",
    "TokenTransitionHead",
    "build_synthetic_transition_batch",
    "build_transition_batch",
    "create_jepa_components",
    "create_joint_components",
    "eval_joint_stage1_step",
    "eval_joint_stage2_step",
    "extract_train_state",
    "joint_contrastive_loss_fn",
    "joint_stage1_loss_fn",
    "joint_stage2_loss_fn",
    "load_training_checkpoint",
    "restore_train_state",
    "save_training_checkpoint",
    "train_step",
    "train_joint_stage1_step",
    "train_joint_stage2_step",
    "transition_jepa_loss",
]
