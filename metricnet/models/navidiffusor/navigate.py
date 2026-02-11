import argparse
import os
import time
import tracemalloc
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
# ROS
import rospy
import torch
import torch.nn as nn
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from geometry_msgs.msg import Point, PoseStamped
# from metricnet.visualizing.action_utils import plot_trajs_and_points
from guide import PathGuide, PathOpt
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
# UTILS
from topic_names import (IMAGE_TOPIC, POS_TOPIC, SAMPLED_ACTIONS_TOPIC,
                         SUB_GOAL_TOPIC, VISUAL_MARKER_TOPIC, WAYPOINT_TOPIC)
from utils import (load_model, msg_to_pil, rotate_point_by_quaternion,
                   to_numpy, transform_images)
from visualization_msgs.msg import Marker

from metricnet.training.utils import get_action

# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 
ACTION_STATS = {}
ACTION_STATS['min'] = np.array([-2.5, -4])
ACTION_STATS['max'] = np.array([5, 4])

# GLOBALS
context_queue = []
context_size = None  
subgoal = []

robo_pos = None
robo_orientation = None
rela_pos = None
# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_plt_param(uc_actions, gc_actions, goal_pos):
    traj_list = np.concatenate([
        uc_actions,
        gc_actions,
    ], axis=0)
    traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
    traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

    point_list = [np.array([0, 0]), goal_pos]
    point_colors = ["green", "red"]
    point_alphas = [1.0, 1.0]
    return traj_list, traj_colors, traj_alphas, point_list, point_colors, point_alphas

def action_plot(uc_actions, gc_actions, goal_pos):
    traj_list, traj_colors, traj_alphas, point_list, point_colors, point_alphas = get_plt_param(uc_actions, gc_actions, goal_pos)
    fig, ax = plt.subplots(1, 1)
    # plot_trajs_and_points(
    #     ax,
    #     traj_list,
    #     point_list,
    #     traj_colors,
    #     point_colors,
    #     traj_labels=None,
    #     point_labels=None,
    #     quiver_freq=0,
    #     traj_alphas=traj_alphas,
    #     point_alphas=point_alphas, 
    # )

    save_path = os.path.join(f"output_goal_{rela_pos}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"output image saved as {save_path}")

def Marker_process(points, id, selected_num, length=8):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns= "points"
    marker.id = id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    if selected_num == id:
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
    else:
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
    for i in range(length):
        p = Point()
        p.x = points[2 * i]
        p.y = points[2 * i + 1]
        p.z = 0
        marker.points.append(p)
    return marker

def Marker_process_goal(points, marker, length=1):
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns= "points"
    marker.id = 0
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    
    for i in range(length):
        p = Point()
        p.x = points[2 * i]
        p.y = points[2 * i + 1]
        p.z = 1
        marker.points.append(p)
    return marker

def callback_obs(msg):
    obs_img = msg_to_pil(msg)
    if obs_img.mode == 'RGBA':
        obs_img = obs_img.convert('RGB')
    else:
        obs_img = obs_img 
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

def pos_callback(msg):
    global robo_pos, robo_orientation
    robo_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    robo_orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, 
                        msg.pose.orientation.z, msg.pose.orientation.w])

def main(args: argparse.Namespace):
    global context_size, robo_pos, robo_orientation, rela_pos

     # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    if args.pos_goal:
        with open(os.path.join(TOPOMAP_IMAGES_DIR, args.dir, "position.txt"), 'r') as file:
            lines = file.readlines()

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    pathguide = PathGuide(device, ACTION_STATS)
    pathopt = PathOpt()
     # load topomap
    topomap_filenames = sorted([filename for filename in os.listdir(os.path.join(
                            TOPOMAP_IMAGES_DIR, args.dir)) if filename.endswith('.png')],
                            key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(topomap_filenames)
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = args.init_node
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node

     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)

    if args.pos_goal:
        pos_curr_msg = rospy.Subscriber(
            POS_TOPIC, PoseStamped, pos_callback, queue_size=1)
        subgoal_pub = rospy.Publisher(
            SUB_GOAL_TOPIC, Marker, queue_size=1)
        robogoal_pub = rospy.Publisher(
            '/goal1', Marker, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
    marker_pub = rospy.Publisher(VISUAL_MARKER_TOPIC, Marker, queue_size=10)

    print("Registered with master node. Waiting for image observations...")

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    scale = 4.0
    scale_factor = scale * MAX_V / RATE
    # navigation loop
    while not rospy.is_shutdown():
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                if args.guide:
                    pathguide.get_cost_map_via_tsdf(context_queue[-1])
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1) 
                obs_images = obs_images.to(device)
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                if args.pos_goal:
                    mask = torch.ones(1).long().to(device)
                    goal_pos = np.array([float(lines[end].split()[0]), float(lines[end].split()[1]), float(lines[end].split()[2])])
                    rela_pos = goal_pos - robo_pos
                    rela_pos = rotate_point_by_quaternion(rela_pos, robo_orientation)[:2]
                    print('rela_pos: ', rela_pos)
                    marker_robogoal = Marker()
                    Marker_process_goal(rela_pos[:2], marker_robogoal, 1)
                    robogoal_pub.publish(marker_robogoal)
                else:
                    mask = torch.zeros(1).long().to(device)  
                goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]
                goal_image = torch.concat(goal_image, dim=0)
                obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                if args.pos_goal:
                    goal_poses = np.array([[float(lines[i].split()[0]), float(lines[i].split()[1]), float(lines[i].split()[2])] for i in range(start, end + 1)])
                    min_idx = np.argmin(np.linalg.norm(goal_poses - robo_pos, axis=1))
                    sg_idx = min_idx
                else:
                    dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                    dists = to_numpy(dists.flatten())
                    min_idx = np.argmin(dists)
                    sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                time4 = time.time()
                closest_node = min_idx + start
                print("closest node:", closest_node)
                
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    
                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)
                
                start_time = time.time()
                for k in noise_scheduler.timesteps[:]:
                    with torch.no_grad():
                        # predict noise
                        noise_pred = model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                    if args.guide:
                        interval1 = 6
                        period = 1
                        if k <= interval1:
                            if k % period == 0:
                                    if k > 2:
                                        grad, cost_list = pathguide.get_gradient(naction, goal_pos=rela_pos, scale_factor=scale_factor)
                                        grad_scale = 1.0
                                        naction -= grad_scale * grad
                                    else:
                                        if k>=0 and k <= 2:
                                            naction_tmp = naction.detach().clone()
                                            for i in range(1):
                                                grad, cost_list = pathguide.get_gradient(naction_tmp, goal_pos=rela_pos, scale_factor=scale_factor)
                                                naction_tmp -= grad
                                            naction = naction_tmp

                naction = to_numpy(get_action(naction))
                naction_selected, selected_num = pathopt.select_trajectory(naction, l=args.waypoint, angle_threshold=45)
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
                for i in range(8):
                    marker = Marker_process(sampled_actions_msg.data[i * 16 + 1 : (i + 1) * 16 + 1] * scale_factor, i, selected_num)
                    marker_pub.publish(marker)
                print("published sampled actions")
                sampled_actions_pub.publish(sampled_actions_msg)

                chosen_waypoint = naction_selected[args.waypoint]
            elif (len(context_queue) > model_params["context_size"]):
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(topomap[start: end + 1]):
                    transf_obs_img = transform_images(context_queue, model_params["image_size"])
                    goal_data = transform_images(sg_img, model_params["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                # look for closest node
                if args.pos_goal:
                    goal_poses = np.array([[float(lines[i].split()[0]), float(lines[i].split()[1]), float(lines[i].split()[2])] for i in range(start, end + 1)])
                    closest_node = np.argmin(np.linalg.norm(goal_poses - robo_pos, axis=1))
                else:
                    closest_node = np.argmin(distances)
                # chose subgoal and output waypoints
                if distances[closest_node] > args.close_threshold:
                    chosen_waypoint = waypoints[closest_node][args.waypoint]
                    sg_img = topomap[start + closest_node]
                else:
                    chosen_waypoint = waypoints[min(
                        closest_node + 1, len(waypoints) - 1)][args.waypoint]
                    sg_img = topomap[start + min(closest_node + 1, len(waypoints) - 1)]     

        if model_params["normalize"]:
            chosen_waypoint[:2] *= (scale_factor / scale)

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)

        torch.cuda.empty_cache()

        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--init-node",
        "-i",
        default=0,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--guide",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--point-goal",
        default=False,
        type=bool,
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)

