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
zero-stiffness velocity drive uses standard `ArticulationAction` velocity targets
at an explicitly recorded smoke-test speed (joint span / phase duration, capped
by the authored maximum velocity), stopping near the limit. This small limit-stop
adapter is not a claimed official Panda+Robotiq grasp controller. It does not import
the Panda `Franka` hand controller, alter force/stiffness/damping/friction,
attach an object, or run any old grasp trajectory. It records actual master
joint motion and arm tracking, not an assumed visual/grasp success.

## Non-destructive composition

The new output `scene.usda` sublayers the existing scene and selects the official
Robotiq variant **at the same `/World/Panda` path**. It does not create a second
robot, deactivate the original, author a mounting joint, or replace references.
Root-stage metadata is copied explicitly: USD sublayering alone does not inherit
the original up axis or units. A non-metre or non-Z-up source is rejected rather
than silently converted. Metadata, robot-root attributes, and non-robot prim
attributes/relationships are compared before simulation. Relative object
references remain anchored to the source layers.

The seven arm joint angles are read from the existing capture by **joint name**,
not guessed or inherited from a different home pose. Existing scene-level
overrides are retained, not silently removed. If original Panda finger DOFs remain
or additional gripper DOFs have independent drives, the motion test stops and
saves the inventory for review; it does not guess how to clean up the scene.

The table, object references, lights and camera are inherited unchanged. The
source scene hash is checked and source assets are never saved. The output
depends on the original source scene and official asset paths; it is not a
portable packaged asset. Output-directory reuse is refused. Scene physics settings
are retained by default. Optional `--replay-physics cpu` enables CPU PhysX / Fabric
OFF only in this test process. Normal Panda scripts remain unchanged.

The JSON `gripper_swap_check.json` records the asset, variant, joint inventory,
authored gains, USD joint/mimic schemas and attributes, physics settings, arm
angles, phase results and failure traceback. Joint inventory is written before
interpreting control conventions. The Isaac Sim 5.1 tensor DOF convention is
Rotation=0, Translation=1; the old smoke test incorrectly expected Rotation=1.
`success` means an open/close smoke test passed, **not that the hammer was picked**.
Unknown master-joint or drive conventions fail rather than guessing commands.

For an inventory-only pass add `--inspect-only`: the script loads the variant,
initializes/reset the simulator, saves the inventory and exits without sending
position/velocity commands. It is not a read-only simulator startup (World reset
still initializes physics), and `inspection_complete` does not mean open/close
or grasp success.

## Next gate

First confirm the replacement and opening/closing in the installed Isaac Sim.
Then prepare a matching Panda+Robotiq cuRobo model and grasp-frame transform,
regenerate GraspGenX candidates with `robotiq_2f_85`, and adapt capture/masking,
collision filtering and replay to that same profile. The original Panda
candidate 039 is not a valid interchangeable-gripper grasp test. Do not pass
this test scene into the current Panda-only end-to-end pipeline.

The local Windows environment cannot execute Isaac Sim. Pure-Python tests cover
control contracts and CLI defaults; additional real-USD composition tests run
when `pxr` is available and otherwise skip. Runtime and grasp compatibility remain
unverified until the Linux test is performed.
