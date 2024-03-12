import jax

batch_select_actions = jax.vmap(lambda v_a, a: v_a[a])
