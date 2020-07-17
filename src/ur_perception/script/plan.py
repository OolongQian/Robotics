#!/usr/bin/env python
from __future__ import print_function

import argparse

import numpy as np
from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import And, print_solution
from pddlstream.language.function import FunctionInfo
from pddlstream.language.generator import from_gen_fn, from_fn
from pddlstream.utils import read, INF, get_file_path

import rospy
import tf
from ur_perception.msg import ScenePoses
import sys
import copy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import PoseStamped
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from scene_poses_tf_listener import id2name


class MoveGroupUR5(object):
    def __init__(self):
        super(MoveGroupUR5, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_ur5', anonymous=True)

        # Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        # kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        # Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        # for getting, setting, and updating the robot's internal understanding of the
        # surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        # Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        # to a planning group (group of joints).  In this tutorial the group is the primary
        # arm joints in the Panda robot, so we set the group's name to "panda_arm".
        # If you are using a different robot, change this value to the name of your robot
        # arm planning group.
        # This interface can be used to plan and execute motions:
        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        # Create a `DisplayTrajectory`_ ROS publisher which is used to display
        # trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)

        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")

        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        self.waiting = True
        self.poses = {}
        self.tfListener = tf.TransformListener()

    def subscribe_6dPose(self, scene_poses):
        for i in range(len(scene_poses.poses)):
            pose = scene_poses.poses[i]
            class_id = scene_poses.class_ids[i]

            pos = pose.position
            ori = pose.orientation

            self.poses[id2name[class_id + 1]] = (pos, ori)

        self.waiting = False

    def get_object_6dPose(self):
        # Get each object's name and 6d pose based on Intel REALSENCE
        # Return a dictionary {'name': (translation, rotation)}
        # Rotation term is in formulation of quaternion
        print('Initializing...')
        # This initializing time depends on the speed of predicating 6d pose.
        while self.waiting:
            rospy.Subscriber("/scene_poses", ScenePoses, self.subscribe_6dPose)

        # Convert the 6d pose w.r.t world.
        for obj in self.poses.keys():
            pos, ori = self.poses[obj]
            ps = PoseStamped()
            ps.header.frame_id = 'camera_link'
            ps.pose.position = pos
            ps.pose.orientation = ori
            target_pose = self.tfListener.transformPose("world", ps).pose
            self.poses[obj] = (target_pose.position, target_pose.orientation)

        return self.poses


def get_problem():
    move_group = MoveGroupUR5()
    poses = move_group.get_object_6dPose()
    print(poses)
    exit()
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = None
    constant_map = {}
    stream_map = {}

    robot_name = 'ur5'
    robot_conf = [0, 0, 1]
    
    obj1 = 'obj1'
    obj2 = 'obj2'
    obj1_pose = [1, 1, 0]
    obj2_pose = [2, 2, 0]
    free = [3, 3, 0]

    init = [
        ('Robot', robot_name),
        ('AtConf', robot_name, robot_conf),
        ('Conf', robot_name, robot_conf),
        ('CanMove', robot_name),
        ('HandEmpty', robot_name),
        ('Obj', obj1),
        ('Pose', obj1, obj1_pose),
        ('AtPose', obj1, obj1_pose),
        ('Obj', obj2),
        ('Pose', obj2, obj2_pose),
        ('AtPose', obj2, obj2_pose),
        ('FreePos', free),
    ]

    goal = [
        # ('AtConf', robot_name, robot_conf),
        ('AtPose', obj1, obj2_pose),
        ('AtPose', obj2, obj1_pose),
    ]
    goal = And(*goal)

    return domain_pddl, constant_map, stream_pddl, stream_map, init, goal


def main(max_time = 180):
    problem = get_problem()
    solution = solve_focused(problem, planner='ff-wastar2', \
                             success_cost=INF, max_time=max_time, debug=False, \
                             unit_efforts=True, effort_weight=1, search_sample_ratio=0)
    print_solution(solution)
    plan, cost, evaluations = solution
    if plan is None:
        print('Unable to find a solution in under {} seconds'.format(max_time))
        return None

# def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--problem', default='load_manipulation', help='The name of the problem to solve')
    # parser.add_argument('-c', '--cfree', action='store_true', help='Disables collisions when planning')
    # parser.add_argument('-v', '--visualizer', action='store_true', help='Use the drake visualizer')
    # parser.add_argument('-s', '--simulate', action='store_true', help='Simulates the system')
    # args = parser.parse_args()
    # rospy.init_node('planner', anonymous=True)
    # plan()


if __name__ == "__main__":
    main()