#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default='Duckietown-udem1-v0')
parser.add_argument("--map-name", default="udem1") #udem1
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()

if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

# driver = ManualDriver(key_handler=key_handler)
driver = AutoDriver(environment=env)

node1 = GraphNode(x_center=0, y_center=0)
node2 = GraphNode(x_center=1, y_center=1)

graph = Graph()
graph.add_node(node1)
graph.add_node(node2)

graph.link_last_two_nodes()

node1.print_neighbors()
node2.print_neighbors()

def update(dt):
    action = driver.get_action()

    obs, reward, done, info = env.step(action)

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

pyglet.app.run()

env.close()
