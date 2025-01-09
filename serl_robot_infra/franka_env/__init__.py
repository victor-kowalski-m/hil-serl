from gymnasium.envs.registration import register


register(
    id="FrankaPCBInsert-Vision-v0",
    entry_point="franka_env.envs.pcb_env:FrankaPCBInsert",
    max_episode_steps=100,
)
register(
    id="FrankaCableRoute-Vision-v0",
    entry_point="franka_env.envs.cable_env:FrankaCableRoute",
    max_episode_steps=500,
)