"""
This script was created during the 2025 Hugging Face LeRobot Hackathon.
It is therefore hacky as fuck, but it works if you run it with python.sh in an isaac sim 4.5.

In general this script:
1. Sets up the robot
2. Reads in the scene config from a yaml
3. Sets up the scene
4. Calculates a trajectory step by step
5. Renders every step
6. Writes motor positions and images to files (motor positions have to be calibrated and converted to deg during upload later)

If you want to use this script glhf. But do not waste to much time as this was not done professionally or with much time
for beautification at hand.
"""

#!/usr/bin/env python3


import os
import glob
import time
import yaml
import numpy as np

# ── 1) Init sim BEFORE any isaacsim imports ──
from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": False})

# ── 2) Isaac Sim core APIs ──
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.articulations import Articulation
from isaacsim.core.api.objects import VisualCuboid, DynamicCuboid
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import omni.isaac.core.utils.prims as prims_utils

# from omni.isaac.usd import _usd


# ── 3) Lula IK imports ──
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
)
import omni.kit.commands as kitcmd
import omni.usd
from pxr import UsdGeom, UsdShade, Sdf, Gf
import random, uuid, numpy as np


# ── 4) ROS 2 imports ──
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import argparse
import cv2
import traceback

parser = argparse.ArgumentParser(description="Sim Joint Publisher")
parser.add_argument(
    "--base_yaml_dir",
    type=str,
    required=True,
    help="Directory containing task YAML files",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to save output results",
)
parser.add_argument(
    "--no_path",
    action="store_true",
    help="Run the tasks in an infinite loop",
)

args = parser.parse_args()

ROS_PUBLISH_MOVEMENT = False  # Set True if you want to publish JointState messages

JOINT_INDEX_TO_NAMES = {
    0: "shoulder_pan.pos",
    1: "shoulder_lift.pos",
    2: "elbow_flex.pos",
    3: "wrist_flex.pos",
    4: "wrist_roll.pos",
    5: "gripper.pos",
}


def move_in_n_steps(
    robot,
    art_kin,
    world,
    goal_pos,
    goal_ori,
    publisher,
    ros_node,
    interval,
    yaml_file,
    camera1,
    overhead_cam,
    idx,
    prompt,
    loop,
    N=100,
    start_steps=1,
):
    """
    Interpolate from current robot pose to the IK solution for (goal_pos, goal_ori)
    in N steps.  At each step we:
      - solve IK
      - interpolate
      - apply to sim
      - publish JointState
      - step/render
      - spin_once + sleep to keep ROS alive
    """
    # 1) Solve IK
    action_goal, ok = art_kin.compute_inverse_kinematics(goal_pos, None)
    if not ok:
        if ROS_PUBLISH_MOVEMENT:
            ros_node.get_logger().warn("move_in_n_steps: goal unreachable!")
        return

    joint_indices = action_goal.joint_indices.copy()
    q_goal = action_goal.joint_positions.copy()
    # 2) current full joint state
    q_start_full = robot.get_joint_positions()
    q_start = q_start_full[joint_indices]

    previous_q = q_start_full.copy()  # Store the initial position for saving

    # 3) loop & interpolate
    for step in range(1, N + 1):
        α = step / N
        q_i = q_start + α * (q_goal - q_start)

        # apply
        robot.apply_action(
            ArticulationAction(
                joint_positions=q_i,
                joint_indices=joint_indices,
            )
        )
        world.step(render=True)
        save_data(
            task_idx=step + start_steps,
            yaml_file=yaml_file,
            camera1=camera1,
            overhead_cam=overhead_cam,
            state_position=previous_q,
            action=robot.get_joint_positions(),
            idx=idx,
            prompt=prompt,
            loop=loop,
        )

        previous_q = robot.get_joint_positions().copy()

        if ROS_PUBLISH_MOVEMENT:
            # publish this intermediate pose
            msg = JointState()
            msg.header.stamp = ros_node.get_clock().now().to_msg()
            # full vector: fill in only controlled DOFs, zero elsewhere
            full_q = q_start_full.copy()
            full_q[joint_indices] = q_i
            msg.name = robot.dof_names
            msg.position = full_q.tolist()
            publisher.publish(msg)

            # keep ROS alive
            rclpy.spin_once(ros_node, timeout_sec=0.0)
        # time.sleep(interval)


def save_data(
    task_idx,
    yaml_file,
    camera1,
    overhead_cam,
    state_position,
    action,
    idx,
    prompt,
    loop,
):
    """
    Save images and robot state data for the given task index.
    Args:
        task_idx (int): Index of the task.
        yaml_file (str): Path to the YAML file for the task.
        camera1 (Camera): Camera object for the gripper camera.
        overhead_cam (Camera): Camera object for the overhead camera.
        state_position (np.ndarray): Current joint positions of the robot.
        action (np.ndarray): Action applied to the robot.
        idx (int): Index of the task in the YAML file list.
    """
    # Create subfolder named after the YAML file (without extension)
    yaml_base = os.path.splitext(os.path.basename(yaml_file))[0]
    subfolder = os.path.join(args.output_path, yaml_base + f"_loop_{loop}")
    os.makedirs(subfolder, exist_ok=True)

    # Capture images from cameras
    gripper_img = camera1.get_rgb()  # shape: (H, W, 4)
    overhead_img = overhead_cam.get_rgb()

    if (
        gripper_img is None
        or overhead_img is None
        or not gripper_img.size > 0
        or not overhead_img.size > 0
    ):
        print(
            f"Warning: No image captured for task {task_idx}. Skipping saving images."
        )
        return

    # Convert RGBA to BGR for OpenCV and drop alpha
    gripper_bgr = cv2.cvtColor((gripper_img * 1).astype(np.uint8), cv2.COLOR_RGB2BGR)
    overhead_bgr = cv2.cvtColor((overhead_img * 1).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save images
    cv2.imwrite(os.path.join(subfolder, f"gripper_cam_{task_idx}.png"), gripper_bgr)
    cv2.imwrite(os.path.join(subfolder, f"overhead_cam_{task_idx}.png"), overhead_bgr)

    # Save robot state and action for the entire epoch (one CSV file per YAML/epoch, one line per step)
    state_file = os.path.join(subfolder, f"joint_data.csv")
    write_header = not os.path.exists(state_file)
    with open(state_file, "a") as f:
        if write_header:
            f.write(
                "task_idx,"
                + "task,"
                + ",".join(
                    [
                        f"state_{JOINT_INDEX_TO_NAMES[i]}"
                        for i in range(len(state_position))
                    ]
                )
                + ","
                + ",".join(
                    [f"action_{JOINT_INDEX_TO_NAMES[i]}" for i in range(len(action))]
                )
                + "\n"
            )
        f.write(
            f"{task_idx},"
            + f'"{prompt}",'
            + ",".join([str(x) for x in state_position])
            + ","
            + ",".join([str(x) for x in action])
            + "\n"
        )


def add_random_items_to_world(world, num_items=5):
    items = [
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "006_mustard_bottle",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "011_banana",
        "019_pitcher_base",
        "021_bleach_cleanser",
        "024_bowl",
        "025_mug",
    ]
    for i in range(num_items):
        item = np.random.choice(items)
        usd_path = f"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/{item}.usd"
        prim_path = f"/World/item_{i}"
        item = stage_utils.add_reference_to_stage(
            usd_path=usd_path, prim_path=prim_path
        )

        # Sample a random position outside 0.5m and within 1.0m radius from the center (xy-plane)
        while True:
            xy = np.random.uniform(-2.0, 2.0, size=2)
            r = np.linalg.norm(xy)
            if 0.5 < r < 2.0:
                break
        z = np.random.uniform(0.0, 0.05)  # Keep z close to ground
        position = np.array([xy[0], xy[1], z])
        item_prim = world.scene.stage.GetPrimAtPath(prim_path)

        xformAPI = UsdGeom.XformCommonAPI(item_prim)
        xformAPI.SetTranslate(
            (
                position[0],
                position[1],
                position[2],  # Adjust z to be on the ground plane
            )
        )
        # Randomly rotate the item around the z-axis
        rotation = np.random.uniform(0, 360)

        # Gf.Vec3f takes three floats
        xformAPI.SetRotate(
            Gf.Vec3f(0.0, 0.0, float(rotation)),  # cast to float
            UsdGeom.XformCommonAPI.RotationOrderXYZ,  # optional, default is XYZ
        )


def move_to_cube(
    robot,
    art_kin,
    world,
    spec,
    n_items,
    publisher,
    ros_node,
    interval,
    yaml_file,
    camera1,
    overhead_cam,
    idx,
    loop,
):
    """
    Move the robot to a cube specified in the YAML file.
    Args:
        robot (Articulation): The robot to control.
        art_kin (ArticulationKinematicsSolver): The kinematics solver for the robot.
        world (World): The simulation world.
        publisher (rclpy.publisher): ROS publisher for joint states.
        ros_node (rclpy.node.Node): ROS node for publishing.
        interval (float): Time interval for each step.
        yaml_file (str): Path to the YAML file containing task specifications.
        camera1 (Camera): Camera object for the gripper camera.
        overhead_cam (Camera): Camera object for the overhead camera.
        idx (int): Index of the task in the YAML file list.
        loop (int): Current loop iteration.
    """
    cube_cfg = spec["cube"]
    # convert 0–255 colour to 0–1
    try:
        color = np.array(cube_cfg["rgb_colour"], dtype=float) / 255.0
    except (TypeError, KeyError, ValueError):
        print(f"Warning: Invalid RGB color in {yaml_file}. Using default red.")
        color = np.array([1.0, 0.0, 0.0])  # Default to red if not specified
    try:
        position = np.array([float(x) for x in cube_cfg["position"]])
    except (TypeError, KeyError, ValueError):
        print(f"Warning: Invalid position in {yaml_file}. Using default [0,0,0].")
        position = np.array([0.0, 0.0, 0.0])
    try:
        orientation = np.array([float(x) for x in cube_cfg["orientation"]])
    except (TypeError, KeyError, ValueError):
        print(f"Warning: Invalid orientation in {yaml_file}. Using default [0,0,0,1].")
        orientation = np.array([0.0, 0.0, 0.0, 1.0])
    try:
        length = float(cube_cfg["length"])
    except (TypeError, KeyError, ValueError):
        print(f"Warning: Invalid length in {yaml_file}. Using default 0.05.")
        length = 0.05
    try:
        width = float(cube_cfg["width"])
    except (TypeError, KeyError, ValueError):
        print(f"Warning: Invalid width in {yaml_file}. Using default 0.05.")
        width = 0.05
    try:
        height = float(cube_cfg["height"])
    except (TypeError, KeyError, ValueError):
        print(f"Warning: Invalid height in {yaml_file}. Using default 0.05.")
        height = 0.05
    # print(position, orientation, length, width, height)
    try:
        position[-1] += height / 2.0  # Adjust z to be on the ground plane
    except TypeError:
        print(
            f"Warning: Invalid position in {yaml_file}. Using default [0,0,{height/2}]."
        )
        position = np.array([0.0, 0.0, height / 2.0])
        exit(1)
    prompt = spec.get("prompt", f"Task {idx}: move to cube")
    print(f"Running task #{idx+1}: {prompt}")
    # 3) Spawn the cube
    cube_prim = f"/World/task_cube_{idx}"
    cube = world.scene.add(
        VisualCuboid(
            prim_path=cube_prim,
            name=f"task_cube_{idx}",
            position=position,
            scale=np.array([length, width, height]),  # meters
            orientation=orientation,
            color=color,
        )
    )
    # cube = world.scene.add(
    #     DynamicCuboid(
    #         prim_path="/World/fancy_cube",
    #         name="fancy_cube",
    #         position=np.array([0.2, -0.2, 0.0]),  # meters
    #         scale=np.array([0.05, 0.05, 0.05]),  # 2×5×2 cm
    #         color=np.array([1.0, 0.0, 0.0]),  # red
    #     )
    # )
    # 4) Let the robot move to that position
    move_in_n_steps(
        robot=robot,
        art_kin=art_kin,
        world=world,
        goal_pos=position,
        goal_ori=cube.get_world_pose()[1],
        publisher=publisher,
        ros_node=ros_node,
        interval=interval,
        yaml_file=yaml_file,
        camera1=camera1,
        overhead_cam=overhead_cam,
        idx=idx,
        prompt=prompt,
        loop=loop,
        N=100,
    )
    world.scene.remove_object(f"task_cube_{idx}")


def pick_and_place_cube(
    robot,
    art_kin,
    world,
    spec,
    n_items,
    publisher,
    ros_node,
    interval,
    yaml_file,
    camera1,
    overhead_cam,
    idx,
    loop,
):
    start_cfg = spec["start_cube"]
    target_cfg = spec["landing_zone"]

    def parse_cube(cfg, default_color, default_pos, default_ori, default_lwh):
        try:
            color = np.array(cfg.get("rgb_colour", default_color), dtype=float) / 255.0
        except Exception:
            color = np.array(default_color, dtype=float) / 255.0
        try:
            position = np.array([float(x) for x in cfg.get("position", default_pos)])
        except Exception:
            position = np.array(default_pos, dtype=float)
        try:
            orientation = np.array(
                [float(x) for x in cfg.get("orientation", default_ori)]
            )
        except Exception:
            orientation = np.array(default_ori, dtype=float)
        try:
            length = float(cfg.get("length", default_lwh[0]))
        except Exception:
            length = default_lwh[0]
        try:
            width = float(cfg.get("width", default_lwh[1]))
        except Exception:
            width = default_lwh[1]
        try:
            height = float(cfg.get("height", default_lwh[2]))
        except Exception:
            height = default_lwh[2]
        position[-1] += height / 2.0
        return color, position, orientation, length, width, height

    # Parse cubes
    start_color, start_pos, start_ori, start_l, start_w, start_h = parse_cube(
        start_cfg, [255, 0, 0], [0, 0, 0], [0, 0, 0, 1], [0.05, 0.05, 0.05]
    )
    target_color, target_pos, target_ori, target_l, target_w, target_h = parse_cube(
        target_cfg, [0, 255, 0], [0, 0, 0], [0, 0, 0, 1], [0.05, 0.05, 0.05]
    )

    # Add start and target cubes
    start_prim = f"/World/start_cube_{idx}"
    target_prim = f"/World/target_cube_{idx}"
    # cube = world.scene.add(
    #     DynamicCuboid(
    #         prim_path="/World/fancy_cube",
    #         name="fancy_cube",
    #         position=np.array([0.2, -0.2, 0.0]),  # meters
    #         scale=np.array([0.05, 0.05, 0.05]),  # 2×5×2 cm
    #         color=np.array([1.0, 0.0, 0.0]),  # red
    #     )
    # )
    material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    # Compute angle to origin in xy-plane
    angle = np.arctan2(start_pos[1], start_pos[0])  # radians
    # Convert angle to quaternion (rotation about z axis)
    quat = rot_utils.euler_angles_to_quats(
        np.array([0, 0, np.degrees(angle)]), degrees=True
    )
    start_ori = quat
    start_cube = DynamicCuboid(
        prim_path=start_prim,
        name=f"start_cube_{idx}",
        position=start_pos,
        scale=np.array([start_l, start_w, start_h]),
        orientation=start_ori,
        color=start_color,
    )
    start_cube.apply_physics_material(material)
    world.scene.add(start_cube)
    # Add the target cube as a DynamicCuboid with high friction to make it "sticky"/grippable
    world.scene.add(
        VisualCuboid(
            prim_path=target_prim,
            name=f"target_cube_{idx}",
            position=target_pos,
            scale=np.array([0.1, 0.1, target_h]),
            orientation=target_ori,
            color=target_color,
        )
    )

    prompt = spec.get("prompt", f"Task {idx}: push cube to cube")
    # Warmup the cameras
    # warmup_steps = 100
    # for _ in range(warmup_steps):
    #     world.step(render=True)
    #     time.sleep(0.025)
    #
    # # Remove cubes after task
    # world.scene.remove_object(f"start_cube_{idx}")
    # world.scene.remove_object(f"target_cube_{idx}")
    # return

    # 1. Open the gripper
    robot.apply_action(
        ArticulationAction(
            joint_positions=[0.4],  # Open position (adjust as needed)
            joint_indices=[5],  # Gripper joint index
        )
    )
    world.step(render=True)
    save_data(
        task_idx=1,
        yaml_file=yaml_file,
        camera1=camera1,
        overhead_cam=overhead_cam,
        state_position=robot.get_joint_positions(),
        action=robot.get_joint_positions(),
        idx=idx,
        prompt=prompt,
        loop=loop,
    )

    # 2. Move to start cube XY, but 0.1m above the cube
    above_start_pos = start_pos.copy()
    above_start_pos[2] += 0.1
    move_in_n_steps(
        robot=robot,
        art_kin=art_kin,
        world=world,
        goal_pos=above_start_pos,
        goal_ori=start_ori,
        publisher=publisher,
        ros_node=ros_node,
        interval=interval,
        yaml_file=yaml_file,
        camera1=camera1,
        overhead_cam=overhead_cam,
        idx=idx,
        prompt=prompt,
        loop=loop,
        N=60,
        start_steps=2,
    )

    # 3. Move down to the start cube position
    move_in_n_steps(
        robot=robot,
        art_kin=art_kin,
        world=world,
        goal_pos=start_pos,
        goal_ori=start_ori,
        publisher=publisher,
        ros_node=ros_node,
        interval=interval,
        yaml_file=yaml_file,
        camera1=camera1,
        overhead_cam=overhead_cam,
        idx=idx,
        prompt=prompt,
        loop=loop,
        N=40,
        start_steps=62,
    )

    # 4. Close the gripper
    gripper_start = 0.4  # Open position
    gripper_end = -0.3  # Closed position
    gripper_steps = 30
    for step in range(1, gripper_steps + 1):
        alpha = step / gripper_steps
        gripper_pos = gripper_start + alpha * (gripper_end - gripper_start)
        robot.apply_action(
            ArticulationAction(
                joint_positions=[gripper_pos],
                joint_indices=[5],  # Gripper joint index
            )
        )
        world.step(render=True)
        save_data(
            task_idx=step + 102,
            yaml_file=yaml_file,
            camera1=camera1,
            overhead_cam=overhead_cam,
            state_position=robot.get_joint_positions(),
            action=robot.get_joint_positions(),
            idx=idx,
            prompt=prompt,
            loop=loop,
        )

    # 5. Lift the cube slightly after closing the gripper (pick up)
    lift_pos = start_pos.copy()
    lift_pos[2] += 0.15  # Lift by 8cm (adjust as needed)
    move_in_n_steps(
        robot=robot,
        art_kin=art_kin,
        world=world,
        goal_pos=lift_pos,
        goal_ori=start_ori,
        publisher=publisher,
        ros_node=ros_node,
        interval=interval,
        yaml_file=yaml_file,
        camera1=camera1,
        overhead_cam=overhead_cam,
        idx=idx,
        prompt=prompt,
        loop=loop,
        N=100,
        start_steps=132,  # Start after gripper close
    )

    # wait 20 steps
    for step in range(1, 20):
        world.step(render=True)
        save_data(
            task_idx=step + 232,
            yaml_file=yaml_file,
            camera1=camera1,
            overhead_cam=overhead_cam,
            state_position=robot.get_joint_positions(),
            action=robot.get_joint_positions(),
            idx=idx,
            prompt=prompt,
            loop=loop,
        )

    # 6. Move to target cube position (simulate carrying/pushing)
    target_pos[2] = start_pos[2]
    move_in_n_steps(
        robot=robot,
        art_kin=art_kin,
        world=world,
        goal_pos=target_pos,
        goal_ori=target_ori,
        publisher=publisher,
        ros_node=ros_node,
        interval=interval,
        yaml_file=yaml_file,
        camera1=camera1,
        overhead_cam=overhead_cam,
        idx=idx,
        prompt=prompt,
        loop=loop,
        N=80,
        start_steps=252,  # Start after lift
    )

    # open gripper in 30 steps to release the cube onto the target cube
    gripper_start = -0.3  # Closed position
    gripper_end = 0.5  # Open position
    gripper_steps = 30
    for step in range(1, gripper_steps + 1):
        alpha = step / gripper_steps
        gripper_pos = gripper_start + alpha * (gripper_end - gripper_start)
        robot.apply_action(
            ArticulationAction(
                joint_positions=[gripper_pos],
                joint_indices=[5],  # Gripper joint index
            )
        )
        world.step(render=True)
        save_data(
            task_idx=step + 332,
            yaml_file=yaml_file,
            camera1=camera1,
            overhead_cam=overhead_cam,
            state_position=robot.get_joint_positions(),
            action=robot.get_joint_positions(),
            idx=idx,
            prompt=prompt,
            loop=loop,
        )

    # 7. Move back to above the target cube
    above_target_pos = target_pos.copy()
    above_target_pos[2] += 0.2  # Adjust height to be above the target cube
    move_in_n_steps(
        robot=robot,
        art_kin=art_kin,
        world=world,
        goal_pos=above_target_pos,
        goal_ori=target_ori,
        publisher=publisher,
        ros_node=ros_node,
        interval=interval,
        yaml_file=yaml_file,
        camera1=camera1,
        overhead_cam=overhead_cam,
        idx=idx,
        prompt=prompt,
        loop=loop,
        N=30,
        start_steps=362,  # Start after gripper open
    )

    # Return to zero in 30 steps
    sim_zero = np.array([0.0, -1.8, 1.6, 1.2, -1.7, -0.2])
    q_start = robot.get_joint_positions()
    N = 30
    for step in range(1, N + 1):
        alpha = step / N
        q_i = q_start + alpha * (sim_zero - q_start)
        robot.apply_action(
            ArticulationAction(
                joint_positions=q_i,
                joint_indices=list(range(len(sim_zero))),
            )
        )
        world.step(render=True)
        save_data(
            task_idx=step + 392,
            yaml_file=yaml_file,
            camera1=camera1,
            overhead_cam=overhead_cam,
            state_position=q_start,
            action=q_i,
            idx=idx,
            prompt=prompt,
            loop=loop,
        )
        q_start = q_i.copy()

    # Remove cubes after task
    world.scene.remove_object(f"start_cube_{idx}")
    world.scene.remove_object(f"target_cube_{idx}")


def main():
    # ── Optional ROS init ──
    if ROS_PUBLISH_MOVEMENT:
        rclpy.init()
        ros_node = rclpy.create_node("sim_joint_state_publisher")
        publisher = ros_node.create_publisher(JointState, "/so101/sim_joint_states", 10)
        ros_node.get_logger().info("ROS node for sim joint publishing started.")
    else:
        ros_node = publisher = None

    # ── Build Isaac Sim world once ──
    world = World()
    world.scene.add_default_ground_plane()

    # ── Load your SO101 USD + Articulation once ──
    usd_path = os.path.expanduser(
        "~/lerobot/so101/SO-ARM100/Simulation/SO101/so101_new_calib/so101_new_calib.usd"
    )
    prim_path = "/World/so101"
    stage_utils.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    robot = Articulation(prim_path=prim_path, name="so101_sim_arm")
    world.scene.add(robot)
    stage = world.scene.stage
    # robot.
    shader_path = "/World/so101/Looks/material_a_3d_printed/Shader"
    shader_prim = stage.GetPrimAtPath(shader_path)
    ground_plane_path = "/World/defaultGroundPlane/Looks/theGrid/Shader"
    ground_plane_prime = stage.GetPrimAtPath(ground_plane_path)

    # ── Cameras ──
    # (you already have these—make sure you call initialize() on any you add)
    camera1 = Camera(
        prim_path="/World/so101/gripper/GripperCamera",
        name="gripper_cam",
        resolution=(500, 500),
    )
    camera1.initialize()  # Initialize the gripper camera
    overhead_cam = Camera(
        prim_path="/World/OverheadCamera",
        name="overhead_cam",
        position=np.array([0.0, -0.15, 2.0]),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
        resolution=(500, 500),
    )
    overhead_cam.initialize()

    # Ensure output_path exists
    os.makedirs(args.output_path, exist_ok=True)

    # ── Kinematics solver ──
    desc = os.path.expanduser(
        "~/lerobot/so101/SO-ARM100/Simulation/SO101/robot_description_new.yaml"
    )
    urdf = os.path.expanduser(
        "~/lerobot/so101/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    )
    lula = LulaKinematicsSolver(robot_description_path=desc, urdf_path=urdf)
    art_kin = ArticulationKinematicsSolver(robot, lula, "gripper_tip")

    # ── Timing ──
    rate_hz = 300
    interval = 1.0 / rate_hz

    # ── Iterate through all YAML tasks ──
    yaml_files = sorted(glob.glob(os.path.join(args.base_yaml_dir, "*.yaml")))
    yaml_files = [
        f for f in yaml_files if os.path.basename(f).startswith("pick_and_place_cube")
    ]
    for loop in range(2):
        for idx, yaml_file in enumerate(yaml_files):
            yaml_base = os.path.splitext(os.path.basename(yaml_file))[0]
            subfolder = os.path.join(args.output_path, yaml_base + f"_loop_{loop}")
            if os.path.exists(subfolder):
                print(f"Skipping {yaml_file} because output folder already exists.")
                continue

            print(f"Processing YAML file: {yaml_file}")

            shader_prim.GetAttribute("inputs:diffuse_color_constant").Set(
                (np.random.rand(), np.random.rand(), np.random.rand())
            )
            # Choose between a random color or a random shade of white for the ground plane
            if np.random.rand() < 0.5:
                # Random color
                ground_plane_prime.GetAttribute("inputs:diffuse_tint").Set(
                    (
                        np.random.uniform(1.5, 2.0),
                        np.random.uniform(0.0, 0.5),
                        np.random.uniform(0.0, 0.2),
                    )
                )
                ground_plane_prime.GetAttribute("inputs:albedo_add").Set(0.0)
            else:
                if np.random.rand() < 0.5:
                    # Random shade of white (all channels similar, high value)
                    ground_plane_prime.GetAttribute("inputs:diffuse_tint").Set(
                        (
                            np.random.uniform(1.5, 2.0),
                            np.random.uniform(1.5, 2.0),
                            np.random.uniform(1.5, 2.0),
                        )
                    )
                    ground_plane_prime.GetAttribute("inputs:albedo_add").Set(
                        np.random.uniform(1.5, 2.0)
                    )
                else:
                    # Random shade of gray (all channels similar, low value)
                    ground_plane_prime.GetAttribute("inputs:diffuse_tint").Set(
                        (
                            np.random.uniform(0.8, 1.0),
                            np.random.uniform(0.8, 1.0),
                            np.random.uniform(0.8, 1.0),
                        )
                    )
                    ground_plane_prime.GetAttribute("inputs:albedo_add").Set(
                        np.random.uniform(-0.5, -0.3)
                    )

            n_items = 40
            # ── Add random items to the world ──
            add_random_items_to_world(world, num_items=n_items)

            # 2) Parse task spec
            with open(yaml_file, "r") as f:
                spec = yaml.safe_load(f)
            if not isinstance(spec, dict):
                print(
                    f"Error: YAML file {yaml_file} did not parse as a dictionary. Please check the file format."
                )
                continue
            task_type = spec.get("task_type", None)
            if task_type is None:
                print(f"Error: 'task_type' not found in {yaml_file}.")
                continue

            # 1) Reset world & robot to known home pose
            world.reset()
            robot.initialize()

            # Set the gripper camera to robo pos
            robot.apply_action(
                ArticulationAction(
                    joint_positions=[-1.6],
                    joint_indices=[4],
                )
            )

            # Warmup the cameras
            warmup_steps = 10
            for _ in range(warmup_steps):
                world.step(render=True)
                time.sleep(0.01)
                if ROS_PUBLISH_MOVEMENT:
                    rclpy.spin_once(ros_node, timeout_sec=0.0)

            if task_type == "move_to_cube":
                try:
                    move_to_cube(
                        robot=robot,
                        art_kin=art_kin,
                        world=world,
                        spec=spec,
                        n_items=n_items,
                        publisher=publisher,
                        ros_node=ros_node,
                        interval=interval,
                        yaml_file=yaml_file,
                        camera1=camera1,
                        overhead_cam=overhead_cam,
                        idx=idx,
                        loop=loop,
                    )

                except Exception as e:
                    # hacky but needs to run through all tasks
                    print(f"Error in task #{idx+1} with YAML file {yaml_file}: {e}")
                    print(f"Error processing task #{idx+1}: {e}")
                    continue

            elif task_type == "pick_and_place_cube":
                try:
                    pick_and_place_cube(
                        robot=robot,
                        art_kin=art_kin,
                        world=world,
                        spec=spec,
                        n_items=n_items,
                        publisher=publisher,
                        ros_node=ros_node,
                        interval=interval,
                        yaml_file=yaml_file,
                        camera1=camera1,
                        overhead_cam=overhead_cam,
                        idx=idx,
                        loop=loop,
                    )
                except Exception as e:
                    print(f"Error in task #{idx+1} with YAML file {yaml_file}: {e}")
                    traceback.print_exc()
                    # hacky but needs to run through all tasks
                    continue

            for i in range(n_items):
                prim_path = f"/World/item_{i}"
                prim = stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    stage.RemovePrim(prim_path)
            # prims_utils.delete_prim(cube_prim)
            for _ in range(50):
                world.step(render=True)

    # ── Clean up ──
    if ROS_PUBLISH_MOVEMENT:
        ros_node.get_logger().info("All tasks done. Shutting down ROS.")
        ros_node.destroy_node()
        rclpy.shutdown()

    sim_app.close()


if __name__ == "__main__":
    main()
