# Official Panda / Robotiq variant trial

This is a separate **gripper replacement and open/close smoke test**. It is not
yet a new end-to-end robot profile or a hammer grasp-success experiment.

Isaac Sim 5.1's official Panda USD ships a `Gripper=Robotiq_2F_85` variant:

- [Official robot assets](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html)
- [Official variant API and exact names](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.experimental.utils/docs/index.html)
- [Robotiq drive/mimic-joint example](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/rig_closed_loop_structures.html)

Use this already-mounted **85**, rather than assembling the previously suggested
140 onto Panda ourselves. Both grippers are supported by GraspGenX, but the
official Panda mounting is available for 85. No new mesh or mount is authored.

## Run

After transferring the new files to the Linux repository:

```bash
cd /home/suzutaro/projects/jikkenn1
conda activate env_isaaclab

head_plan=outputs/hammer_head_handover_fixedjoint_e2e_v1/curobo_grasp_lift_trials/candidate_039
head_capture=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"]["capture"])' "$head_plan/grasp_lift_plan_check.json")

python scripts/isaac_try_panda_robotiq.py \
  --scene-usd scenes/hammer_01.usda \
  --capture "$head_capture" \
  --output "outputs/panda_robotiq85_$(date +%Y%m%d_%H%M%S)" \
  --simulation-only
```

The script opens, closes, and reopens the gripper (three seconds per phase), then
keeps the GUI open with **Open / Close** buttons. `--headless` runs the finite
smoke test and exits. It reads declared joint limits and drive gains; a
zero-stiffness velocity drive remains velocity-controlled. It does not import
the Panda `Franka` hand controller, alter force/stiffness/damping/friction,
attach an object, or run any old grasp trajectory. It records actual master
joint motion and arm tracking, not an assumed visual/grasp success.

## Non-destructive composition

The new output `scene.usda` sublayers the existing scene. In **this new layer
only**, `/World/Panda` is inactive; a fresh reference to the official Panda USD
appears at `/World/PandaRobotiq` with the Robotiq variant. This avoids old
Panda-finger drive overrides leaking into the replacement. The root transform
is copied, and the seven arm joint angles are read from the existing capture
by **joint name**, not guessed or inherited from a different home pose.

The table, object references, lights and camera are inherited unchanged. The
source scene hash is checked and source assets are never saved. The output
depends on the original source scene and official asset paths; it is not a
portable packaged asset. Output-directory reuse is refused. CPU PhysX / Fabric
OFF applies only to the test process. Normal Panda scripts remain unchanged.

The JSON `gripper_swap_check.json` records the asset, variant, joint inventory,
authored gains, CPU settings, arm angles, phase results and failure traceback.
`success` means an open/close smoke test passed, **not that the hammer was picked**.
Unknown master-joint or drive conventions fail rather than guessing commands.

## Next gate

First confirm the replacement and opening/closing in the installed Isaac Sim.
Then prepare a matching Panda+Robotiq cuRobo model and grasp-frame transform,
regenerate GraspGenX candidates with `robotiq_2f_85`, and adapt capture/masking,
collision filtering and replay to that same profile. The original Panda
candidate 039 is not a valid interchangeable-gripper grasp test. Do not pass
this test scene into the current Panda-only end-to-end pipeline.

The local Windows environment cannot execute Isaac Sim; runtime and grasp
compatibility remain unverified until the Linux test is performed.
