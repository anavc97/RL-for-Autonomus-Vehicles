# Tuning A Path Tracking Controller for an Autonomous Vehicle Using Reinforcement Learning

- Base code for project: "Tuning A Path Tracking Controller for an Autonomous Vehicle Using Reinforcement Learning"
- This system (fig.1) uses a reinforcement learning agent to tune the parameters of a path tracking controller (fig.2)

--------![Full_arch](https://user-images.githubusercontent.com/57260454/220219032-6f9c628c-399e-4698-9566-0dc1bba5ef6c.jpg)
Figure 1




--------![controller](https://user-images.githubusercontent.com/57260454/220218751-f5405d5a-bf8c-4bcb-a0f6-6bb19b5544a4.jpg)
Figure 2

# Requirements

This code requires the following programs to be installed:

1. CARLA SIMULATOR (version 0.9.9.4)
2. CARLA ROS Bridge: https://github.com/anavc97/ros-bridge
3. ROS (I use Noetic)

---

# Instructions of Use

1. After installing CARLA: copy all files from the main folder to /[path]/[to]/CARLA_0.9.9.4/PythonAPI/examples/

2. In /[path]/[to]/CARLA_0.9.9.4/, run 
```
bash -xv ./CarlaUE4.sh -opengl -quality-level=Low 
```
(flags might need adjustment)

3. In another terminal, in /[path]/[to]/CARLA_0.9.9.4/, run 

```
source ~/carla-ros-bridge/catkin_ws/devel/setup.bash
roslaunch carla_ackermann_control carla_ackermann_control.launch
```

4. In /[path]/[to]/CARLA_0.9.9.4/PythonAPI/examples/, run 

```
source ~/carla-ros-bridge/catkin_ws/devel/setup.bash
```

followed by the python script you want to run

5. (optional) Copy file *carla_ros_bridge_with_example_ego_vehicle.launch* to */[path]/[to]/carla-ros-bridge/ros-bridge/carla_ros_bridge/launch/* (or just edit the file already in that folder)


---

# Description of files

- manual_control.py: test simulator manually
- carla_ros_bridge_with_example_ego_vehicle.launch: edit vehicle's initial position
**Note:** **During training** the parameter *synchronous_mode_wait_for_vehicle_control_command* needs to be *True*.
- QL_LC.py: train the agent to perform the Lane Changing Maneuver (parameters: length of training step (LOOP_TIME), number of episodes (EPISODES), reference path file (REFPATH_FILE))
- QL_R.py: train the agent to perform the Roundabout Navigation Maneuver (parameters: length of training step (LOOP_TIME), number of episodes (EPISODES), reference path file (REFPATH_FILE))
- system.py: validate system with a full path (parameters: sets of gains for each maneuver (function agent())
- test_control_repeat.py: validate system in each maneuver (parameters: list of gains to train (K_list), number of tests per gain (N_LAPS))
**Note:** Do not forget to configure the launch file with the appropriate initial position for that maneuver

**MAP**: Town 3

LANE CHANGE:
"-84.9,124.6056,0,0,0,-91.4"

ROUNDABOUT:
"4.82,-41.115,0.5,0,0,89.88"
