#!/usr/bin/env python

import scipy.io as scio
import os
import rospy
import tf
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

br = tf.TransformBroadcaster()
vis_id = rospy.get_param("/vis_id")

if __name__ == "__main__":
    rospy.init_node("df_iterative_publisher")
    rospy.sleep(0.5)

    poses_path = "/home/vinjohn/sucheng/Robotics/perception/DenseFusion/experiments/eval_result/ycb/Densefusion_iterative_result"
    masked_pld_path = "/home/vinjohn/sucheng/Robotics/perception/DenseFusion/experiments/eval_result/ycb/masked_plds"
    filenames = os.listdir(masked_pld_path)

    filename = filenames[vis_id]
    dataname = "_".join(filename.split("_")[:3])
    index = int(filename.split("_")[4])
    clsid = int(filename.split("_")[-1][:-4])
    clsname = id2name[clsid]

    cls_pose = scio.loadmat(os.path.join(poses_path, dataname))['poses'][index]

    rot, trans = cls_pose[:4], cls_pose[4:]

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        br.sendTransform(trans, rot, rospy.Time.now(),
                         id2name[clsid] + "_iterative", "map")
        rate.sleep()
