# Writing DQN Together

This code was written mostly from scratch over the course of an hour in a breakout session. (Only `ReplayBuffer` and `mlp` were written beforehand.) It's messy and has minimal features, but shows that writing a basic RL algorithm is very doable!

## Bugs encountered

When we first ran the code, it immediately errored, unsurprisingly. I worked my way through a few shape errors. Most notably, I originally tried to use `tf.batch_gather` to select the correct Q value to update based on which action was taken (line 71), but I realized that it tries to use the same array across the whole batch, so I had to revert to using `tf.one_hot` and `tf.reduce_sum`. (An alternative would be to construct the correct set of indices and use `tf.gather_nd`.)

After working our way through a few of those bugs (and remembering to actually update the target network -- thanks to the participant who reminded me of that!), the code *sort of* worked: after training for a little while, it occasionally would find a policy that kept the CartPole up for the full 200 timesteps, but would then collapse again to totally broken performance. We thought maybe the hyperparameters were off, so we fiddled around with those a bit, but it didn't meaningfully change the behavior. Finally, as time was running out, Josh Achiam came in and rescued us, pointing out that I had entirely failed to set `obs = next_obs` after calling `env.step`! As a result, the network was operating entirely blindly -- all states looked exactly like the initial state. I'm baffled it learned anything at all!

Lesson learned: **Even if your RL algorithm sort of works, it's highly likely there's still a bug lurking in there!**
