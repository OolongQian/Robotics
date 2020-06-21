#include <ros/ros.h> 
#include <tf/transform_broadcaster.h>

/*
I suddenly find that YLX tells me to use tf static_transform_publisher <-- tf package provides this functionality. 
rosrun tf static_transform_publisher ＼
        0.0263365 0.0389317 0.0792014 -0.495516 0.507599 -0.496076 0.500715 \
        /ee_link   /camera_link   100
*/

int main(int argc, char** argv){
  ros::init(argc, argv, "camera_tf_broadcaster");
  ros::NodeHandle node;

  tf::TransformBroadcaster br;
  tf::Transform transform;

  ros::Rate rate(10.0);
  while (node.ok()){
    transform.setOrigin( tf::Vector3(0.0263365, 0.0389317, 0.0792014) );
    transform.setRotation( tf::Quaternion(-0.495516, 0.507599, -0.496076, 0.500715) );
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "ee_link", "camera_link"));
    rate.sleep();
  }
  return 0;
};
