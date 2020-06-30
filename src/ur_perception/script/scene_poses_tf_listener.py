#!/usr/bin/env python
# import roslib
# roslib.load_manifest('ur_perception')
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


def listener():
    rospy.init_node('scene_poses_tf_listener')
    lsr = tf.TransformListener()
    rate = rospy.Rate(10)
    test_obj_name = id2name[12] # just for a test
    print "Initializing..."
    while not rospy.is_shutdown():
        try:
            (trans, rot) = lsr.lookupTransform(test_obj_name, 'tool0', rospy.Time(0))
            print(trans)
            print(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        rate.sleep()


if __name__ == '__main__':
    listener()
