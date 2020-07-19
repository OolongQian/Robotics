#!/usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import sys
import copy
from math import pi

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import And, print_solution
from pddlstream.language.function import FunctionInfo
from pddlstream.language.generator import from_gen_fn, from_fn
from pddlstream.utils import read, INF, get_file_path

import moveit_commander
import moveit_msgs.msg
from moveit_commander.conversions import pose_to_list

from robotiq_85_msgs.msg import GripperCmd, GripperStat
from scene_poses_tf_listener import id2name
from ur_perception.msg import ScenePoses


class UR5(object):
    def __init__(self):
        super(UR5, self).__init__()
        # Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        # kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        # Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        # to a planning group (group of joints).  In this tutorial the group is the primary
        # arm joints in the Panda robot, so we set the group's name to "panda_arm".
        # If you are using a different robot, change this value to the name of your robot
        # arm planning group.
        # This interface can be used to plan and execute motions:
        arm_name = "manipulator"
        arm = moveit_commander.MoveGroupCommander(arm_name)
        gripper_name = "gripper"
        gripper = moveit_commander.MoveGroupCommander(gripper_name)

        # Create a `DisplayTrajectory`_ ROS publisher which is used to display
        # trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)

        # Settings about the gripper
        rospy.Subscriber("/gripper/stat", GripperStat, self._update_gripper_stat, queue_size=10)
        self._gripper_pub = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=10)
        self._gripper_stat = GripperStat()
        self._gripper_cmd = GripperCmd()

        # We can also print the name of the end-effector link for this group:
        eef_link = arm.get_end_effector_link()
        print("============ End effector link for arm: %s" % eef_link)
        eef_link = gripper.get_end_effector_link()
        print("============ End effector link for gripper: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")

        self.robot = robot
        self.arm = arm
        self.gripper = gripper
        self.display_trajectory_publisher = display_trajectory_publisher
        self.group_names = group_names
        self.init_pose = self.arm.get_current_pose().pose

        self.waiting = True
        self.object_poses = {}
        self.tfListener = tf.TransformListener()

    def subscribe_6d_pose(self, scene_poses):
        for i in range(len(scene_poses.poses)):
            pose = scene_poses.poses[i]
            class_id = scene_poses.class_ids[i]

            pos = pose.position
            ori = pose.orientation

            self.object_poses[id2name[class_id + 1]] = (pos, ori)

        self.waiting = False

    def get_object_6d_pose(self):
        # Get each object's name and 6d pose based on Intel REALSENCE
        # Return a dictionary in form of {'name': (translation, rotation)}
        # Rotation term is in formulation of quaternion
        print('Initializing...')
        # This initializing time depends on the speed of predicating 6d pose.
        while self.waiting:
            rospy.Subscriber("/scene_poses", ScenePoses, self.subscribe_6d_pose)

        # Convert the 6d pose w.r.t world.
        for obj in self.object_poses.keys():
            pos, ori = self.object_poses[obj]
            ps = PoseStamped()
            ps.header.frame_id = 'camera_link'
            ps.pose.position = pos
            ps.pose.orientation = ori
            target_pose = self.tfListener.transformPose("world", ps).pose
            self.object_poses[obj] = (target_pose.position, target_pose.orientation)

        return self.object_poses

    def execute_plan(self, plan):
        print("Press Enter to visualize the trajectory in RViz.")
        raw_input()
        self.display_trajectory(plan)
        # print("Press Enter to move the real hardware.")
        # raw_input()
        # self.arm.execute(plan, wait=True)

    def display_trajectory(self, plan):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory)

    def go_to_joint_state(self, goal):
        self.arm.set_joint_value_target(goal)
        plan = self.arm.plan()
        self.execute_plan(plan)

    def go_to_pose_goal_demo(self):
        current_pose = self.arm.get_current_pose().pose
        target_pose = copy.deepcopy(current_pose)
        target_pose.position.z = 0.6
        self.arm.set_pose_target(current_pose)
        plan = self.arm.plan()
        self.execute_plan(plan)

        self.arm.set_pose_target(target_pose)
        plan = self.arm.plan()
        self.execute_plan(plan)

    def go_to_pose(self, goal):
        self.arm.set_pose_target(goal)
        plan = self.arm.plan()
        self.execute_plan(plan)

    def go_to_position(self, goal):
        self.arm.set_position_target(goal)
        plan = self.arm.plan()
        self.execute_plan(plan)

    def _update_gripper_stat(self, stat):
        self._gripper_stat = stat

    def gripper_self_check(self, pos=0.0):
        while not self._gripper_stat.is_ready:
            continue
        if abs(self._gripper_stat.position - pos) < 0.002:
            return False
        else:
            return True

    def gripper_goto(self, pos=0.0, speed=1.0, force=1.0):
        # The speed of running this code is faster than transporting gripper_commands in hardware
        # Thus, we must define 3 states to distinguish each stages:
        # 1) wait until gripper is ready, then publish the command.
        # 2) wait until gripper's state is moving.
        # 3) wait until gripper stops moving.
        # Only after these three stages can we finish this function.
        need_move = self.gripper_self_check(pos)
        if not need_move:
            return
        state = 0
        while state < 3:
            if state == 0:
                ready = False
                while not self._gripper_stat.is_ready:
                    continue
                self._gripper_cmd.position = pos
                self._gripper_cmd.speed = speed
                self._gripper_cmd.force = force
                self._gripper_pub.publish(self._gripper_cmd)
                state += 1
            if state == 1:
                while not self._gripper_stat.is_moving:
                    continue
                state += 1
            if state == 2:
                while self._gripper_stat.is_moving:
                    continue
                state += 1

    def open_gripper(self):
        self.gripper_goto(pos=0.085)
        print(self._gripper_stat, '\n')

    def close_gripper(self):
        self.gripper_goto(pos=0)
        print(self._gripper_stat, '\n')

    def get_current_pose(self):
        return self.arm.get_current_pose().pose

    def get_init_pose(self):
        return self.init_pose


def get_problem(robot):
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = None
    constant_map = {}
    stream_map = {}

    robot_name = 'ur5'
    robot_conf = robot.get_current_pose()
    init = [
        ('Robot', robot_name),
        ('AtConf', robot_name, robot_conf),
        ('Conf', robot_name, robot_conf),
        ('CanMove', robot_name),
        ('HandEmpty', robot_name),
        ('FreePos', robot_conf)
    ]

    objects = robot.get_object_6d_pose()
    for obj_name in objects.keys():
        init += [
            ('Obj', obj_name), # Currently, we only care about translation
            ('Pose', obj_name, objects[obj_name][0]),
            ('AtPose', obj_name, objects[obj_name][0]),
        ]

    print('The robot detects the following objects: ')
    obj_names = list(objects.keys())
    for i, obj_name in enumerate(obj_names):
        print('\t%d %s' % (i + 1, obj_name))
    print('Which one do you want to grasp?')
    idx = raw_input()
    idx = int(idx)
    while idx <= 0 or idx > len(obj_names):
        print('Wrong input, try again: ')
        idx = raw_input()
        idx = int(idx)

    goal = [
        ('AtConf', robot_name, robot_conf),
        ('AtGrasp', robot_name, obj_names[idx - 1]),
    ]

    goal = And(*goal)

    return domain_pddl, constant_map, stream_pddl, stream_map, init, goal


def interpret_plan(robot, plan):
    print('=================')
    print(plan)
    for i, (name, args) in enumerate(plan):
        print(name)
        if name == 'movetoobj':
            goal = args[3]
            robot.go_to_position([goal.x, goal.y, goal.z])
        elif name == 'movetofreepos':
            current_pose = robot.get_current_pose()
            current_pose.position = args[2].position
            robot.go_to_pose(current_pose)
        elif name == 'grasp':
            robot.close_gripper()
        elif name == 'place':
            robot.open_gripper()


def main(max_time = 180):
    robot = UR5()
    robot.open_gripper()
    problem = get_problem(robot)
    solution = solve_focused(problem, planner='ff-wastar2',
                             success_cost=INF, max_time=max_time, debug=False,
                             unit_efforts=True, effort_weight=1, search_sample_ratio=0)
    # print_solution(solution)
    plan, cost, evaluations = solution
    if plan is None:
        print('Unable to find a solution in under {} seconds'.format(max_time))
        return None
    interpret_plan(robot, plan)
    return True

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
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('planning_ur5', anonymous=True)
    main()

