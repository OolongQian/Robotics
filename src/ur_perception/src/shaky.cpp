#include <ros/ros.h>

// MoveIt
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/kinematic_constraints/utils.h>

#include <moveit/planning_interface/planning_interface.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/PlanningScene.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

#include <boost/scoped_ptr.hpp>

int main(int argc, char **argv) {
    const std::string node_name = "motion_planning_tutorial";
    ros::init(argc, argv, "shaky");
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::NodeHandle node_handle("~");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");

    robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
    ROS_INFO("Model frame: %s", kinematic_model->getModelFrame().c_str());

    robot_state::RobotStatePtr kinematic_state(new robot_state::RobotState(kinematic_model));
    kinematic_state->setToDefaultValues();

    const robot_state::JointModelGroup *joint_model_group = kinematic_model->getJointModelGroup("manipulator");
    const std::vector <std::string> &joint_names = joint_model_group->getVariableNames();

    //  std::vector<double> joint_values;
//  kinematic_state->copyJointGroupPositions(joint_model_group, joint_values);
//  for (std::size_t i = 0; i < joint_names.size(); ++i)
//  {
//    ROS_INFO("Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
//  }

//  joint_values[0] = 5.57;
//  kinematic_state->setJointGroupPositions(joint_model_group, joint_values);

//  ROS_INFO_STREAM("Current state is " << (kinematic_state->satisfiesBounds() ? "valid" : "not valid"));
//
//  kinematic_state->enforceBounds();
//  ROS_INFO_STREAM("Current state is " << (kinematic_state->satisfiesBounds() ? "valid" : "not valid"));

//  kinematic_state->setToRandomPositions(joint_model_group);
    const Eigen::Isometry3d &end_effector_state = kinematic_state->getGlobalLinkTransform("ee_link");
    ROS_INFO_STREAM("Translation: \n" << end_effector_state.translation() << "\n");
    ROS_INFO_STREAM("Rotation: \n" << end_effector_state.rotation() << "\n");

    double timeout = 0.1;
    bool found_ik = kinematic_state->setFromIK(joint_model_group, end_effector_state, timeout);
    std::vector<double> joint_values;

    if (found_ik) {
        kinematic_state->copyJointGroupPositions(joint_model_group, joint_values);
        for (std::size_t i = 0; i < joint_names.size(); ++i) {
            ROS_INFO("Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
        }
    } else {
        ROS_INFO("Did not find IK solution");
    }

    const std::string PLANNING_GROUP = "manipulate";
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    planning_scene->getCurrentStateNonConst().setToDefaultValues(joint_model_group, "ready");
    boost::scoped_ptr<pluginlib::ClassLoader<planning_interface::PlannerManager>> planner_plugin_loader;
    planning_interface::PlannerManagerPtr planner_instance;
    std::string planner_plugin_name;
    if (!node_handle.getParam("planning_plugin", planner_plugin_name))
        ROS_FATAL_STREAM("Could not find planner plugin name");
    try
    {
        planner_plugin_loader.reset(new pluginlib::ClassLoader<planning_interface::PlannerManager>(
                "moveit_core", "planning_interface::PlannerManager"));
    }
    catch (pluginlib::PluginlibException& ex)
    {
        ROS_FATAL_STREAM("Exception while creating planning plugin loader " << ex.what());
    }
    try
    {
        planner_instance.reset(planner_plugin_loader->createUnmanagedInstance(planner_plugin_name));
        if (!planner_instance->initialize(robot_model, node_handle.getNamespace()))
            ROS_FATAL_STREAM("Could not initialize planner instance");
        ROS_INFO_STREAM("Using planning interface '" << planner_instance->getDescription() << "'");
    }
    catch (pluginlib::PluginlibException& ex)
    {
        const std::vector<std::string>& classes = planner_plugin_loader->getDeclaredClasses();
        std::stringstream ss;
        for (std::size_t i = 0; i < classes.size(); ++i)
            ss << classes[i] << " ";
        ROS_ERROR_STREAM("Exception while loading planner '" << planner_plugin_name << "': " << ex.what() << std::endl
                                                             << "Available plugins: " << ss.str());
    }

    namespace rvt = rviz_visual_tools;
    moveit_visual_tools::MoveItVisualTools visual_tools("panda_link0");
    visual_tools.loadRobotStatePub("/display_robot_state");
    visual_tools.enableBatchPublishing();
    visual_tools.deleteAllMarkers();  // clear all old markers
    visual_tools.trigger();

    /* Remote control is an introspection tool that allows users to step through a high level script
     via buttons and keyboard shortcuts in RViz */
    visual_tools.loadRemoteControl();

    /* RViz provides many types of markers, in this demo we will use text, cylinders, and spheres*/
    Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 1.75;
    visual_tools.publishText(text_pose, "Motion Planning API Demo", rvt::WHITE, rvt::XLARGE);

    /* Batch publishing is used to reduce the number of messages being sent to RViz for large visualizations */
    visual_tools.trigger();

    /* We can also use visual_tools to wait for user input */
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");

    ros::shutdown();
    return 0;
}
