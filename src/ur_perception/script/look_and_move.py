from cv_bridge import CvBridge

bridge = CvBridge()

"""
Robot execution GETTING_STARTED: 
roscore 
roslaunch ur_perception camera_setup.launch 
roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.1.102  // drive robot hardware and publish physical information. 

EXECUTE this script.
"""

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
21: '061_foam_brick'
"""

# TODO : refactor this utility function in ur_perception package.
# id2name = {1: '002_master_chef_can',
#            2: '003_cracker_box',
#            3: '004_sugar_box',
#            4: '005_tomato_soup_can',
#            5: '006_mustard_bottle',
#            6: '007_tuna_fish_can',
#            7: '008_pudding_box',
#            8: '009_gelatin_box',
#            9: '010_potted_meat_can',
#            10: '011_banana',
#            11: '019_pitcher_base',
#            12: '021_bleach_cleanser',
#            13: '024_bowl',
#            14: '025_mug',
#            15: '035_power_drill',
#            16: '036_wood_block',
#            17: '037_scissors',
#            18: '040_large_marker',
#            19: '051_large_clamp',
#            20: '052_extra_large_clamp',
#            21: '061_foam_brick'}
#
#
# def clsId2clsName(cls_id):
#     """convert mmdetection predicted output class id to class name.
#     all classes can be inspected under ur_perception/models.
#     cls_id starts from 1, and 1 is background class, therefore 002_master_chef_can has class id 2."""
#     return id2name[cls_id - 1]
#
#
#
# class Eyes:
#     def __init__(self, ur_perception_ros_path, perception_repos_path):
#         self.ur_perception_ros_path = ur_perception_ros_path
#         self.perception_repos_path = perception_repos_path
#         self.external_inputs_path = "external_inputs"
#         self.external_outputs_path = "external_outputs"
#
#         self.cameraInfoTopic = "/realsense/camera_info"
#         self.cameraRgbTopic = "/realsense/rgb"
#         self.cameraDepthTopic = "/realsense/depth"
#         self.cameraLinkName = "/camera_link"
#
#         "rospy.wait_for_message is equivalent to ros::spinOnce." \
#         "only able to receive message when node is inited."
#         self.cameraInfo = rospy.wait_for_message(self.cameraInfoTopic, CameraInfo, timeout=5)
#
#     def getRgbd(self):
#         "raw ros image message needs to be processed by openCVS_bridge -> ndarray."
#         ros_rgb = rospy.wait_for_message(self.cameraRgbTopic, Image, timeout=5)
#         ros_depth = rospy.wait_for_message(self.cameraDepthTopic, Image, timeout=5)
#         rgb = bridge.imgmsg_to_cv2(ros_rgb, "passthrough")
#         depth = bridge.imgmsg_to_cv2(ros_depth, "passthrough")
#         return rgb, depth
#
#     def blink(self):
#         "use rostopic type [topic_name] to check for message type."
#         # "use rosmsg show [message_name] to check for messsage data structure."
#         try:
#             "can display these images."
#             return self.detectAndEstimate(*self.getRgbd())
#         except CvBridgeError:
#             rospy.logerr("cv bridge error")
#
#     def myImshow(self, img, figure_id):
#         plt.figure(figure_id)
#         plt.imshow(img)
#
#     def makeTmpDirs(self):
#         pass
#         # if os.path.exists(os.path.join(self.ur_perception_ros_path, "data")):
#         #     shutil.rmtree(os.path.join(self.ur_perception_ros_path, "data"))
#         # data_inputs = os.path.join(self.ur_perception_ros_path, "data", self.external_inputs_path)
#         # data_outputs = os.path.join(self.ur_perception_ros_path, "data", self.external_outputs_path)
#         # os.makedirs(os.path.join(data_inputs, "images"))
#         # os.makedirs(os.path.join(data_inputs, "depths"))
#         # os.makedirs(os.path.join(data_outputs))
#
#     def rmTmpDirs(self):
#         pass
#         # if os.path.exists(os.path.join(self.ur_perception_ros_path, "data")):
#         #     shutil.rmtree(os.path.join(self.ur_perception_ros_path, "data"))
#
#     def detectAndEstimate(self, cv_rgb, cv_depth):
#         """ invoke external neural network inference program to do estimation over images."""
#
#         "prepare tmp directory."
#         self.makeTmpDirs()
#         data_inputs = os.path.join(self.ur_perception_ros_path, "data", self.external_inputs_path)
#         data_outputs = os.path.join(self.ur_perception_ros_path, "data", self.external_outputs_path)
#
#         "write rgb and depth into input folder."
#         rgb_name = os.path.join(data_inputs, "images", "tmp.tiff")
#         depth_name = os.path.join(data_inputs, "depths", "tmp.tiff")
#         cv2.imwrite(rgb_name, cv_rgb)
#         cv2.imwrite(depth_name, cv_depth)
#
#         "invoke a subprocess to do neural network inference."
#         interpreter = "/home/vinjohn/miniconda3/envs/densefusion/bin/python"
#         nn_inference_script = os.path.join(self.perception_repos_path, "denseFusionOnExternalImages.py")
#         arguments = {"--ur_perception_ros_path": "/home/vinjohn/sucheng/Robotics/src/ur_perception/data",
#                      "--perception_repos_path": "/home/vinjohn/sucheng/Robotics/perception"}
#
#         command = "{} {}".format(interpreter, nn_inference_script)
#         for k, v in arguments.items():
#             command += " {} {}".format(k, v)
#
#         "use subprocess.Popen to invoke external program, and retrieve its output and error."
#         res = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#         output, errors = res.communicate()
#         print output
#         if res.returncode:
#             print "RUN failed\n\n%s\n\n" % errors
#         if errors:
#             print "\n\n%s\n\n" % errors
#
#         pose_mat_path = os.path.join(data_outputs, "df_outputs", "df_iterative_result", "tmp.tiff.mat")
#         pose = scio.loadmat(pose_mat_path)["poses"]
#
#         self.rmTmpDirs()
#         return pose
#
#     def remember(self):
#         save_path = '../visual_validate'
#         rgb, depth = self.getRgbd()
#         poses = self.detectAndEstimate(rgb, depth)
#         cv2.imwrite(os.path.join(save_path, "rgb.png"), rgb)
#         cv2.imwrite(os.path.join(save_path, "depth.png"), depth)
#         scio.savemat('{0}/{1}.mat'.format(save_path, "class_pose"),
#                      {'poses': poses})
#
#
# from moveit_python import MoveGroupInterface, PlanningSceneInterface
# import geometry_msgs.msg
# from tf import TransformListener
#
#
# class Terminator:
#     def __init__(self, ur_perception_ros_path, perception_repos_path):
#         self.ur_perception_ros_path = ur_perception_ros_path
#         self.perception_repos_path = perception_repos_path
#         self.eyes = Eyes(self.ur_perception_ros_path, self.perception_repos_path)
#
#         "planning scene interface creates a 'virtual environment around robot'."
#         self.p = PlanningSceneInterface(frame="base_link")
#         "group is ur5's planning group, can be seen with ur5_moveit_config demo.launch." \
#         "frame sets up the FOR for our coordinate system." \
#         "MoveGroupInterface is used for goal specification and control."
#         self.g = MoveGroupInterface(group="manipulator", frame="base_link")
#
#         self.tf_listener = TransformListener()
#
#     def buildScene(self, cls_and_poses):
#         for i in range(cls_and_poses.shape[0]):
#             obj_pose = cls_and_poses[i]
#             cls_id, rot, trans = obj_pose[0], obj_pose[1:5], obj_pose[5:]
#             cls_name = clsId2clsName(cls_id)
#             instance_name = "index_{}_classname_{}".format(i, cls_name)
#
#             camera_pose = geometry_msgs.msg.PoseStamped()
#             camera_pose.header.frame_id = "camera_link"
#             camera_pose.pose.position.x, camera_pose.pose.position.y, camera_pose.pose.position.z = trans[1], trans[0], \
#                                                                                                     trans[2]
#             camera_pose.pose.orientation.x, camera_pose.pose.orientation.y, camera_pose.pose.orientation.z, camera_pose.pose.orientation.w = rot
#             time.sleep(2)  # leave some time for tf_listener to prepare.
#             base_pose = self.tf_listener.transformPose("base_link", camera_pose)
#
#             self.addMesh(clsId2clsName(cls_id), instance_name, base_pose.pose.position, base_pose.pose.orientation)
#
#     def addMesh(self, cls_name, instance_name, point, quaternion):
#         print "begin add object {} of class {}".format(instance_name, cls_name)
#         mesh_name = "../models/{}/textured.obj".format(cls_name)
#         self.p.addMesh(name=instance_name, pose=geometry_msgs.msg.Pose(point, quaternion), filename=mesh_name)
#
#
# def eyeTest(ur_perception_ros_path, perception_repos_path):
#     eyes = Eyes(ur_perception_ros_path, perception_repos_path)
#     eyes.blink()
#
#
# def buildSceneTest(ur_perception_ros_path, perception_repos_path):
#     terminator = Terminator(ur_perception_ros_path, perception_repos_path)
#     oo_scene = terminator.eyes.blink()
#     terminator.buildScene(oo_scene)
#
#
# def remember4Validate(ur_perception_ros_path, perception_repos_path):
#     terminator = Terminator(ur_perception_ros_path, perception_repos_path)
#     terminator.eyes.remember()
#
#
# if __name__ == "__main__":
#     rospy.init_node("terminator_moveit_py", anonymous=True)
#
#     ur_perception_ros_path = "/home/vinjohn/sucheng/Robotics/src/ur_perception"
#     perception_repos_path = "/home/vinjohn/sucheng/Robotics/perception"
#
#     # eyeTest(ur_perception_ros_path, perception_repos_path)
#     buildSceneTest(ur_perception_ros_path, perception_repos_path)
#     # remember4Validate(ur_perception_ros_path, perception_repos_path)
#
#     # terminator = Terminator(ur_perception_ros_path, perception_repos_path)
#     # terminator.eyes.blink()
#     # terminator.addCube()
#     # terminator.removeCube()
#     # terminator.addMesh()
#     # terminator.removeMesh()
