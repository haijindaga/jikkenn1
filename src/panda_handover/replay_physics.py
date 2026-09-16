"""Opt-in replay diagnostics using Isaac Sim's public physics API."""


def configure_replay_physics(context, manager, mode):
    if mode not in ("default", "cpu"):
        raise ValueError(f"unknown replay physics mode: {mode}")
    before = replay_physics_state(context)
    if mode == "cpu":
        manager.set_physics_sim_device("cpu")
        context.enable_gpu_dynamics(False)
        context.enable_fabric(False)
    after = replay_physics_state(context)
    validate_replay_physics(mode, after)
    return {"requested_mode": mode, "before": before, "configured": after}


def replay_physics_state(context):
    return {
        "device": str(context.device),
        "gpu_dynamics_enabled": bool(context.is_gpu_dynamics_enabled()),
        "gpu_pipeline_enabled": bool(context.use_gpu_pipeline),
        "fabric_enabled": fabric_is_enabled(context),
    }


def fabric_is_enabled(context):
    try:
        return bool(context.use_fabric)
    except TypeError as error:
        # Isaac Sim 5.1's property calls is_fabric_enabled() without the
        # required (but unused) 'enable' argument. Read the same extension
        # state as that getter, without changing any simulator settings.
        if "is_fabric_enabled" not in str(error) or "enable" not in str(error):
            raise
        return _fabric_extension_enabled()


def _fabric_extension_enabled():
    import omni.kit.app

    return bool(
        omni.kit.app.get_app().get_extension_manager().is_extension_enabled(
            "omni.physx.fabric"
        )
    )


def validate_replay_physics(mode, state):
    if mode == "cpu" and (
        state["device"] != "cpu"
        or state["gpu_dynamics_enabled"]
        or state["gpu_pipeline_enabled"]
        or state["fabric_enabled"]
    ):
        raise RuntimeError(f"CPU replay physics / Fabric OFF was not applied: {state}")
