#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

#include <moveit_visual_tools/moveit_visual_tools.h>
#include <tf/transform_listener.h>


int main(int argc, char** argv) {
    // Init ros node
    ros::init(argc, argv, "move_group_ur5");
    ros::NodeHandle node_handle;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // Init tf listener
    tf::TransformListener listener;
    tf::StampedTransform transform;

    // Set planning group in ur5
    static const std::string PLANNING_GROUP = "manipulator";
    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // Get the joint model group
    const robot_state::JointModelGroup* joint_model_group =
        move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

    // Set the visual info in RViz
    namespace rvt = rviz_visual_tools;
    moveit_visual_tools::MoveItVisualTools visual_tools("base_link");
    visual_tools.deleteAllMarkers();

    // Remote control is an introspection tool that allows users to step through a high level script
    // via buttons and keyboard shortcuts in RViz
    visual_tools.loadRemoteControl();

    // Set markers in RViz
    Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 1.75;
    visual_tools.publishText(text_pose, "MoveGroupUR5", rvt::WHITE, rvt::XLARGE);

    // Batch publishing is used to reduce the number of messages being sent to RViz for large visualizations
    visual_tools.trigger();

    // Print the name of reference frame for this robot.
    ROS_INFO_NAMED("ur5", "Planning frame: %s", move_group.getPlanningFrame().c_str());
    // Print the name of the end effector link.
    ROS_INFO_NAMED("ur5", "End effector link %s", move_group.getEndEffectorLink().c_str());
    // Get a list of all the groups in the robot.
    ROS_INFO_NAMED("ur5", "Available Planning Groups:");
    std::copy(move_group.getJointModelGroupNames().begin(), move_group.getJointModelGroupNames().end(),
              std::ostream_iterator<std::string>(std::cout, ", "));

    // Variables in planning
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success;

    // Start
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");

    // Get current joint positions
    moveit::core::RobotStatePtr current_state = move_group.getCurrentState();
    std::vector<double> joint_group_positions;
    current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);
    std::vector<double> initial_joint_group_positions = joint_group_positions;

    // The story is that, we predict the object's 6-D pose w.r.t frame "camera_link", but not the absolute 6-d pose.
    // Thus, we need to convert the 6-d pose w.r.t "camera_link" to the 6-d pose w.r.t "world".

    // We first get the initial pose w.r.t "camera_link" from tf listener.
    // The while(true) structure is that, I find this tf listener cannot always get the tf info immediately, even though
    // the tf actually exists.
    geometry_msgs::Pose target_pose_camera;
    while (true) {
        try {
            listener.lookupTransform("021_bleach_cleanser", "camera_link", ros::Time(0), transform);
            break;
        }
        catch (tf::TransformException &ex) {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
        }
    }
    target_pose_camera.position.x = transform.getOrigin().x();
    target_pose_camera.position.y = transform.getOrigin().y();
    target_pose_camera.position.z = transform.getOrigin().z();
    target_pose_camera.orientation.x = transform.getRotation().x();
    target_pose_camera.orientation.y = transform.getRotation().y();
    target_pose_camera.orientation.z = transform.getRotation().z();
    target_pose_camera.orientation.w = transform.getRotation().w();

    // Then, use a PoseStamped to cover the Pose
    // PoseStamped = Pose + Header(stamp, frame_id, seq), we only care about frame_id here.
    geometry_msgs::PoseStamped target_pose_stamped_camera;
    target_pose_stamped_camera.header.frame_id = "camera_link";
    target_pose_stamped_camera.pose = target_pose_camera;

    // Create another PoseStamped, waiting to hold the absolute pose
    geometry_msgs::PoseStamped& target_pose_stamped = target_pose_stamped_camera;

    // Get the object's pose w.r.t. the frame "world".
    while (true) {
        try {
            listener.transformPose("world", target_pose_stamped_camera, target_pose_stamped);
            break;
        }
        catch (tf::TransformException &ex) {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
        }
    }

    move_group.setPoseTarget(target_pose_stamped.pose);

    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start planning.");

    // Plan and verify success
    success = (move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO_NAMED("ur5", "Visualizing plan 1 (touch object) %s", success ? "" : "FAILED");

    // Visualize the plan in RViz
    visual_tools.deleteAllMarkers();
    visual_tools.publishText(text_pose, "Joint Space Goal", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    visual_tools.trigger();
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to move the robot");

    // Let the real robot move
    if (success)
        move_group.move();

    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to plan the way back");

    // Get current position and plan for the way back.
    move_group.setStartState(*move_group.getCurrentState());
    move_group.setJointValueTarget(initial_joint_group_positions);

    // Plan and verify success
    success = (move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO_NAMED("ur5", "Visualizing plan 2 (back) %s", success ? "" : "FAILED");

    // Visualize the plan in RViz
    visual_tools.deleteAllMarkers();
    visual_tools.publishText(text_pose, "Joint Space Goal", rvt::WHITE, rvt::XLARGE);
    visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    visual_tools.trigger();
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to move the robot");

    if (success)
        move_group.move();

    // END
    ros::shutdown();
    return 0;
}