# Shield experiments (loose)

I've synthesized a shield for the Random Walk example (rw_shield.json).

I've written a custom ActorCriticPolicy for SB3 PPO, that employs a shield
during training and during execution. I then trained a PPO model with this
shielded policy (100,000 timesteps), and a regular PPO model for comparison.

Using VIPER, these were then converted to decision trees. They perform more or
less equally (at least, there is no pattern in which one achieves a bit higher
or lower performance). From the verbose output during the learning phase though,
it did look like the shielded version were significantly quicker at achieving
the expected best score.

On the other hand, if the shield is not enforced during execution, the shielded
model performs absolutely horrendously.

## Files

- rw_shield.json - the shield as a minimized decision tree

- shielded_RandomWalk-v0_ppo.zip - shielded PPO strategy
- viper_shielded_RW.pk - the shielded VIPER generated strategy (in sklearn format)
- viper_shielded_RW.json - same as above but as in stratetrees format
- shielded_rw_example.png - visualization of the shielded strategy partitioning

- unshielded_RandomWalk-v0_ppo.zip - shielded PPO strategy
- viper_unshielded_RW.pk - the unshielded VIPER generated strategy (in sklearn format)
- viper_unshielded_RW.json - same as above but as in stratetrees format
- unshielded_rw_example.png - visualization of the unshielded strategy partitioning
