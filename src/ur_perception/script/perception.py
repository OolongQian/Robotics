import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import os
import cv2
import subprocess
import shutil
import scipy.io as scio

bridge = CvBridge()

"""
Dataset class information 

1: '002_master_chef_can', 
2: '003_cracker_box', 
3: '004_sugar_box', 
4: '005_tomato_soup_can', 
5: '006_mustard_bottle',
6:'007_tuna_fish_can', 
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
21: '061_foam_brick')
"""


class Eyes:
    def __init__(self, ur_perception_ros_path, perception_repos_path):
        self.ur_perception_ros_path = ur_perception_ros_path
        self.perception_repos_path = perception_repos_path
        self.external_inputs_path = "external_inputs"
        self.external_outputs_path = "external_outputs"

        rospy.init_node("eyes", anonymous=True)
        self.cameraInfoTopic = "/realsense/camera_info"
        self.cameraRgbTopic = "/realsense/rgb"
        self.cameraDepthTopic = "/realsense/depth"
        self.cameraLinkName = "/camera_link"
        # rospy.wait_for_message is equivalent to ros::spinOnce.
        self.cameraInfo = rospy.wait_for_message(self.cameraInfoTopic, CameraInfo, timeout=5)

    def blink(self):
        "use rostopic type [topic_name] to check for message type."
        "use rosmsg show [message_name] to check for messsage data structure."
        rgb = rospy.wait_for_message(self.cameraRgbTopic, Image, timeout=5)
        depth = rospy.wait_for_message(self.cameraDepthTopic, Image, timeout=5)

        try:
            "raw ros image message needs to be processed by openCVS_bridge -> ndarray."
            cv_rgb = bridge.imgmsg_to_cv2(rgb, "passthrough")
            cv_depth = bridge.imgmsg_to_cv2(depth, "passthrough")
            "can display these images."
            # self.myImshow(cv_rgb, 1)
            # self.myImshow(cv_depth, 2)
            # plt.show()
            self.detectAndEstimate(cv_rgb, cv_depth)
        except CvBridgeError:
            rospy.logerr("cv bridge error")

    def myImshow(self, img, figure_id):
        plt.figure(figure_id)
        plt.imshow(img)

    def makeTmpDirs(self):
        pass
        if os.path.exists(os.path.join(self.ur_perception_ros_path, "data")):
            shutil.rmtree(os.path.join(self.ur_perception_ros_path, "data"))
        data_inputs = os.path.join(self.ur_perception_ros_path, "data", self.external_inputs_path)
        data_outputs = os.path.join(self.ur_perception_ros_path, "data", self.external_outputs_path)
        os.makedirs(os.path.join(data_inputs, "images"))
        os.makedirs(os.path.join(data_inputs, "depths"))
        os.makedirs(os.path.join(data_outputs))

    def rmTmpDirs(self):
        pass
        # if os.path.exists(os.path.join(self.ur_perception_ros_path, "data")):
        #     shutil.rmtree(os.path.join(self.ur_perception_ros_path, "data"))

    def detectAndEstimate(self, cv_rgb, cv_depth):
        """ invoke external neural network inference program to do estimation over images."""

        "prepare tmp directory."
        self.makeTmpDirs()
        data_inputs = os.path.join(self.ur_perception_ros_path, "data", self.external_inputs_path)
        data_outputs = os.path.join(self.ur_perception_ros_path, "data", self.external_outputs_path)

        "write rgb and depth into input folder."
        rgb_name = os.path.join(data_inputs, "images", "tmp.png")
        depth_name = os.path.join(data_inputs, "depths", "tmp.png")
        cv2.imwrite(rgb_name, cv_rgb)
        cv2.imwrite(depth_name, cv_depth)

        "invoke a subprocess to do neural network inference."
        interpreter = "/home/vinjohn/miniconda3/envs/densefusion/bin/python"
        nn_inference_script = os.path.join(self.perception_repos_path, "denseFusionOnExternalImages.py")
        arguments = {"--ur_perception_ros_path": "/home/vinjohn/sucheng/Robotics/src/ur_perception/data",
                     "--perception_repos_path": "/home/vinjohn/sucheng/Robotics/perception"}

        command = "{} {}".format(interpreter, nn_inference_script)
        for k, v in arguments.items():
            command += " {} {}".format(k, v)

        "use subprocess.Popen to invoke external program, and retrieve its output and error."
        res = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, errors = res.communicate()
        print output
        if res.returncode:
            print "RUN failed\n\n%s\n\n" % errors
        if errors:
            print "\n\n%s\n\n" % errors

        pose_mat_path = os.path.join(data_outputs, "df_outputs", "df_iterative_result", "tmp.png.mat")
        pose = scio.loadmat(pose_mat_path)["poses"]

        self.rmTmpDirs()


if __name__ == "__main__":
    ur_perception_ros_path = "/home/vinjohn/sucheng/Robotics/src/ur_perception"
    perception_repos_path = "/home/vinjohn/sucheng/Robotics/perception"

    eyes = Eyes(ur_perception_ros_path, perception_repos_path)
    eyes.blink()
