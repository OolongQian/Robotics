#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import cv2
import subprocess
import scipy.io as scio
from ur_perception.msg import ScenePoses
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud
import shutil
import std_msgs.msg
from geometry_msgs.msg import Point32
import PIL.Image as PilImage
bridge = CvBridge()


class Eyes:
    def __init__(self, ur_perception_ros_path, perception_repos_path):
        self.ur_perception_ros_path = ur_perception_ros_path
        self.perception_repos_path = perception_repos_path
        self.external_inputs_path = "external_inputs"
        self.external_outputs_path = "external_outputs"

        self.cameraInfoTopic = "/realsense/camera_info"
        self.cameraRgbTopic = "/realsense/rgb"
        self.cameraDepthTopic = "/realsense/depth"
        self.cameraLinkName = "/camera_link"

        "rospy.wait_for_message is equivalent to ros::spinOnce." \
        "only able to receive message when node is inited."
        self.cameraInfo = rospy.wait_for_message(self.cameraInfoTopic, CameraInfo, timeout=5)

    def getRgbd(self):
        "raw ros image message needs to be processed by openCVS_bridge -> ndarray."
        ros_rgb = rospy.wait_for_message(self.cameraRgbTopic, Image, timeout=5)
        ros_depth = rospy.wait_for_message(self.cameraDepthTopic, Image, timeout=5)
        rgb = bridge.imgmsg_to_cv2(ros_rgb, "passthrough")
        depth = bridge.imgmsg_to_cv2(ros_depth, "passthrough")
        return rgb, depth

    def makeTmpDirs(self):
        if os.path.exists(os.path.join(self.ur_perception_ros_path, "data")):
            shutil.rmtree(os.path.join(self.ur_perception_ros_path, "data"))
        data_inputs = os.path.join(self.ur_perception_ros_path, "data", self.external_inputs_path)
        data_outputs = os.path.join(self.ur_perception_ros_path, "data", self.external_outputs_path)
        os.makedirs(os.path.join(data_inputs, "images"))
        os.makedirs(os.path.join(data_inputs, "depths"))
        os.makedirs(os.path.join(data_outputs))

    def rmTmpDirs(self):
        if os.path.exists(os.path.join(self.ur_perception_ros_path, "data")):
            shutil.rmtree(os.path.join(self.ur_perception_ros_path, "data"))

    def detectAndEstimate(self, cv_rgb, cv_depth):
        """ invoke external neural network inference program to do estimation over images."""

        "prepare tmp directory."
        self.makeTmpDirs()
        data_inputs = os.path.join(self.ur_perception_ros_path, "data", self.external_inputs_path)
        data_outputs = os.path.join(self.ur_perception_ros_path, "data", self.external_outputs_path)

        "write rgb and depth into input folder."
        rgb_name = os.path.join(data_inputs, "images", "tmp.tiff")
        depth_name = os.path.join(data_inputs, "depths", "tmp.tiff")
        "NOTE : use cv2.imwrite --> 0-255 int"
        # cv2.imwrite(rgb_name, cv_rgb)
        # cv2.imwrite(depth_name, cv_depth)
        PilImage.fromarray(cv_rgb).save(rgb_name)
        PilImage.fromarray(cv_depth).save(depth_name)

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

        pose_mat_path = os.path.join(data_outputs, "df_outputs", "df_iterative_result", "tmp.tiff.mat")
        # pose_mat_path = os.path.join(data_outputs, "df_outputs", "df_wo_refine_result", "tmp.tiff.mat")
        pose = scio.loadmat(pose_mat_path)["poses"]

        plds_path = os.path.join(data_outputs, "df_outputs", "masked_plds")
        plds = []
        for pld_name in os.listdir(plds_path):
            pld_path = os.path.join(plds_path, pld_name)
            pts = []
            with open(pld_path, 'r') as f:
                for line in f.readlines():
                    pt = line.strip().split(' ')
                    assert len(pt) == 3
                    pts.append((float(pt[0]), float(pt[1]), float(pt[2])))
            plds.append(pts)

        self.rmTmpDirs()
        return pose, plds


# license removed for brevity
def talker():
    rospy.init_node('scene_poses_publisher', anonymous=True)
    pub = rospy.Publisher('scene_poses', ScenePoses, queue_size=10)
    rate = rospy.Rate(0.1)  # 0.2hz

    ur_perception_ros_path = "/home/vinjohn/sucheng/Robotics/src/ur_perception"
    perception_repos_path = "/home/vinjohn/sucheng/Robotics/perception"
    eyes = Eyes(ur_perception_ros_path, perception_repos_path)

    while not rospy.is_shutdown():
        scene_poses = ScenePoses()

        cls_poses, plds = eyes.detectAndEstimate(*eyes.getRgbd())

        try:
            for i in range(cls_poses.shape[0]):
                cls_pose = cls_poses[i]
                pld = plds[i]
                cls_id, rot, trans = cls_pose[0], cls_pose[1:5], cls_pose[5:]

                scene_poses.class_ids.append(cls_id)

                pose_msg = Pose()
                pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = trans[0], trans[1], trans[2]
                pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w = rot[0], \
                                                                                                                 rot[1], \
                                                                                                                 rot[2], \
                                                                                                                 rot[3]
                scene_poses.poses.append(pose_msg)

                pld_msg = PointCloud()
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "camera_link"
                for pt in pld:
                    pld_msg.points.append(Point32(*pt))

                scene_poses.point_clouds.append(pld_msg)

            pub.publish(scene_poses)
        except IndexError:
            print "point cloud detection index out of bound, may be because mmdetection high-level inference API exception occurred. " \
                  "I have no time to debug. because point cloud publisher is just for illustration, therefore nevermind. "
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
