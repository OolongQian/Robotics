#!/usr/bin/env python
import roslib

import rospy
import tf
from ur_perception.msg import ScenePoses

# TODO : refactor this utility function in ur_perception package.
id2name = {1: '002_master_chef_can',
           2: '003_cracker_box',
           3: '004_sugar_box',
           4: '005_tomato_soup_can',
           5: '006_mustard_bottle',
           6: '007_tuna_fish_can',
           7: '008_pudding_box',
           8: '009_gelatin_box',
           9: '010_potted_meat_can',
           10: '011_banana',
           11: '019_pitcher_base',
           12: '021_bleach_cleanser',
           13: '024_bowl',
           14: '025_mug',
           15: '035_power_drill',
           16: '036_wood_block',
           17: '037_scissors',
           18: '040_large_marker',
           19: '051_large_clamp',
           20: '052_extra_large_clamp',
           21: '061_foam_brick'}

br = tf.TransformBroadcaster()


def dynamicTfBroadcast(scene_poses):
    for i in range(len(scene_poses.poses)):
        pose = scene_poses.poses[i]
        class_id = scene_poses.class_ids[i]

        "assume each class instance is unique. " \
        "actually, tracking is required."
        pos = pose.position
        ori = pose.orientation
        br.sendTransform((pos.x, pos.y, pos.z), (ori.x, ori.y, ori.z, ori.w), rospy.Time.now(),
                         id2name[class_id], "camera_link")

        print "update 6d pose for {}".format(id2name[class_id])


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('scene_poses_tf_publisher', anonymous=True)
    rospy.Subscriber("scene_poses", ScenePoses, dynamicTfBroadcast)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
