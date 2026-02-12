from gymnasium.envs.registration import register, registry

if "Unimal-v0" not in registry:
    try:
        register(
            id="Unimal-v0",
            entry_point="derl.envs.tasks.task:make_env",
            max_episode_steps=1000,
        )
    except Exception:
        pass  # metamorph also registers Unimal-v0