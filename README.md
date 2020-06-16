# Robotics
This is a repository that aims to realize basic functionality of UR5 robot. 

Need to checkout universal_robot submodule to melodic-devel branch, and checkout mmdetection to branch aade6801e7df66679b1fe9d162da0d03b4742dd4, which is release v1.0.0.  

Under perception folder, we provide the implementation that, given depth images and object models, outputs 6D pose for these objects presented. 
Its implementation is adapted from cocoAPI, mmdetection, and DenseFusion repositories. Please the README under perception folder.  
