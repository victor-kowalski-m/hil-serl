# Source the setup.bash file for the first ROS workspace
source ~/catkin_serl/devel/setup.bash

# Set ROS master URI to localhost
export ROS_MASTER_URI=http://localhost:11311

# Run the first instance of franka_server.py in the background
python franka_server.py \
    --robot_ip=172.16.1.2 \
    --gripper_type=Franka \
    --reset_joint_target=0,0,0,-1.9,-0,2,0 \
    --flask_url=0.0.0.0 \ # 127.0.0.1 \
    --ros_port=11311
    
# curl -X POST 127.0.0.1:5000/activate_gripper
# curl -X POST -H "Content-Type: application/json" -d '{"arr":[0.42287451979677902,0.2919328194561711,0.31935349992446304,0.9994928165271177,0.026686381118793564,0.016239465720793874,0.006182760433241234]}' 127.0.0.1:5000/pose

# rs-enumerate-devices
