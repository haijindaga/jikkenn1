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
        "fabric_enabled": bool(context.use_fabric),
    }


def validate_replay_physics(mode, state):
    if mode == "cpu" and (
        state["device"] != "cpu"
        or state["gpu_dynamics_enabled"]
        or state["gpu_pipeline_enabled"]
        or state["fabric_enabled"]
    ):
        raise RuntimeError(f"CPU replay physics / Fabric OFF was not applied: {state}")
