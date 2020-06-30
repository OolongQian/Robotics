# Robotics
This is a repository that aims to realize basic functionality of UR5 robot. 

Need to checkout universal_robot submodule to melodic-devel branch.

I have also created v1.0-stable branch from commit aade6801e7df66679b1fe9d162da0d03b4742dd4 to continue my development of mmdetection. 

Under perception folder, we provide the implementation that, given depth images and object models, outputs 6D pose for these objects presented. 
Its implementation is adapted from cocoAPI, mmdetection, and DenseFusion repositories. Please the README under perception folder.  

## Usage
Since I've written

```buildoutcfg
source ~/sucheng/Robotics/devel/setup.bash
```
into ~/.bashrc, we don't need to source it in each new terminal.

```
1 roscore
2 roslaunch ur_perception camera_setup.launch
3 roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.1.102
4 roslaunch ur_perception start_scene_poses.launch
5 rviz
```

## Current progress
* Understand the whole story from scratch.

* Fix the index bug in DenseFusion on external images.

* Get the relative transfrom from tool0 (the machine gripper, need further calibration)
to each given object.

## Short-term plan
* Use moveit to let UR5 touch the object.

* Learn the structure of pddlStream.