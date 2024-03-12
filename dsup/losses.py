import jax
import jax.numpy as jnp


def huber_loss(
    targets: jax.Array, predictions: jax.Array, kappa: float = 1.0
) -> jax.Array:
    """Huber loss.
    Equivalent to L1 loss when |targets - predictions| > kappa, and L2 loss otherwise.

    Args:
        targets: target values
        predictions: predicted values
        kappa: threshold for L2 loss

    Returns:
        Huber loss
    """
    err = jnp.abs(targets - predictions)
    return jnp.where(err <= kappa, 0.5 * err**2, 0.5 * kappa**2 + kappa * (err - kappa))


def quantile_huber_loss(
    targets: jax.Array, predictions: jax.Array, kappa: float = 1.0
) -> jax.Array:
    """Quantile huber loss, as introduced in the QR-DQN paper.

    Args:
        targets: target quantile values
        predictions: predicted quantile values
        kappa: threshold parameter for huber loss

    Returns:
        Quantile Huber loss
    """
    num_atoms = predictions.shape[-1]
    quantile_midpoints = (jnp.arange(num_atoms) + 0.5) / num_atoms
    pairwise_errors = targets[None, :] - predictions[:, None]
    huber_loss = (jnp.abs(pairwise_errors) <= kappa).astype(
        jnp.float32
    ) * 0.5 * pairwise_errors**2 + (jnp.abs(pairwise_errors) > kappa).astype(
        jnp.float32
    ) * kappa * (jnp.abs(pairwise_errors) - 0.5 * kappa)
    tau_diff = jnp.abs(
        quantile_midpoints[:, None] - (pairwise_errors < 0).astype(jnp.float32)
    )
    return jnp.sum(jnp.mean(tau_diff * huber_loss, axis=-1))
