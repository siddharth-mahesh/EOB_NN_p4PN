import jax
import jax.numpy as jnp
from EOB_NN_p4PN.EOB_NNp4PN.eob_nnp4pn_training_module_prelim import Neural_EOB
import equinox as eqx

jax.config.update("jax_enable_x64", True)

x_train = jnp.load("seob_x_train_prelim.npy")

print("Checking the variance weights block directly:")
model = Neural_EOB(srate=4096)
model_zero = model
model_zero = eqx.tree_at(lambda m: m.A_scale, model_zero, 0.0)
model_zero = eqx.tree_at(lambda m: m.D_scale, model_zero, 0.0)
model_zero = eqx.tree_at(lambda m: m.Q_scale, model_zero, 0.0)
model_zero = eqx.tree_at(lambda m: m.f_scale, model_zero, 0.0)
model_zero = eqx.tree_at(lambda m: m.delta_scale, model_zero, 0.0)

print("Calling vmap(model_zero) on x_train...")
try:
    y_base = jax.vmap(model_zero)(x_train)
    print("Success. Output shape:", y_base.shape)
except Exception as e:
    import traceback
    traceback.print_exc()

