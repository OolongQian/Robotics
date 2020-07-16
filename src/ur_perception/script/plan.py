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


def get_problem():
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


def plan(max_time = 180):
    problem = get_problem()
    solution = solve_focused(problem, planner='ff-wastar2', \
                             success_cost=INF, max_time=max_time, debug=False, \
                             unit_efforts=True, effort_weight=1, search_sample_ratio=0)
    print_solution(solution)
    plan, cost, evaluations = solution
    if plan is None:
        print('Unable to find a solution in under {} seconds'.format(max_time))
        return None

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--problem', default='load_manipulation', help='The name of the problem to solve')
    # parser.add_argument('-c', '--cfree', action='store_true', help='Disables collisions when planning')
    # parser.add_argument('-v', '--visualizer', action='store_true', help='Use the drake visualizer')
    # parser.add_argument('-s', '--simulate', action='store_true', help='Simulates the system')
    # args = parser.parse_args()
    rospy.init_node('planner', anonymous=True)
    plan()


if __name__ == "__main__":
    main()