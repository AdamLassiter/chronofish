# Chronofish Neural

Owns the position and action encoder, policy/value model, checkpoint format,
inference backends, and fitting. CPU and Vulkan use compatible model
checkpoints. Training and Vulkan are crate features. The [trainer
guide](../trainer/README.md) covers selecting and checking devices.

```sh
cargo test -p chronofish-neural
```
