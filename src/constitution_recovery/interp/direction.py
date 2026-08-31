"""Persona-direction extraction and ablation: the pure math, testable without a GPU.

Diff-in-means between target and baseline activations gives a candidate style
direction per layer; ablation projects it out of the residual stream. Used by
scripts/persona_direction.py for the causal test of the channel-capture finding.
"""

import torch


def diff_in_means(target_acts, base_acts):
    """(n, d) activation banks -> unit direction (d,) pointing base -> target."""
    v = target_acts.float().mean(0) - base_acts.float().mean(0)
    return v / v.norm()


def separation(target_acts, base_acts, direction):
    """How well the direction separates the two banks: gap between the projected
    means in pooled-std units. ~0 = useless; > 2 = cleanly separable."""
    pt = target_acts.float() @ direction
    pb = base_acts.float() @ direction
    pooled = torch.sqrt((pt.var() + pb.var()) / 2)
    return ((pt.mean() - pb.mean()) / (pooled + 1e-8)).item()


def project_out(x, direction):
    """Remove the direction from activations x (..., d). Norm-preserving in the
    orthogonal complement -- everything not along v is untouched."""
    v = direction / direction.norm()
    v = v.to(x.dtype)
    return x - (x @ v).unsqueeze(-1) * v


def ablation_hook(direction):
    """Forward hook for a decoder layer: strips the direction from its output
    residual stream. Handles layers that return tuples."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (project_out(output[0], direction.to(output[0].device)),) + output[1:]
        return project_out(output, direction.to(output.device))
    return hook
