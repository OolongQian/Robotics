#!/usr/bin/env python

import scipy.io as scio
import os
import rospy
import tf
from ur_perception.msg import ScenePoses
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs.msg

"""this piece of code publishes object mesh to rviz, translated by eval prediction result."""

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

pld_publisher = rospy.Publisher("/scene_partial_pld_topic", PointCloud, queue_size=666)

def pldPublish(scene_poses):
    for i in range(len(scene_poses.point_clouds)):
        cls_id = scene_poses.class_ids[i]
        if cls_id != 11:
            continue
        pld = scene_poses.point_clouds[i]

        "assume each class instance is unique. " \
        "actually, tracking is required."
        pld_msg = PointCloud()
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_link"
        pld_msg.header = header
        pld_msg.points = pld.points

        print pld_msg
        print "publish!"
        pld_publisher.publish(pld_msg)



def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('scene_partial_pld_publisher')
    rospy.Subscriber("scene_poses", ScenePoses, pldPublish)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()

