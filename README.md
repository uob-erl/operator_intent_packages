# Bayesian-Based-Human-Operator-Intent-Recognition

Repository containing the code for the Bayesian-Based Human-Operator Intent Recognition experiment that took place in Extreme Robotics Laboratory (ERL). The experiment attempted to investigate human intent recongition while human-operators were assigned to remotely control a mobile robot in multiple Search & Rescue scenarios. A Bayesian-Based framework is adopted to perform intent recognition (i.e. most probable goal that human wants to progress the robot towards) based on environmental data (e.g. path length, angle) gathered by the mobile robot on-the-fly.

For more information regarding the concept and the principles that govern the research project a good start could be https://github.com/uob-erl/fuzzy_mi_controller .

# Setting up & getting started
If you want to recreate the experiment and test the algorithm, follow the steps below :

1) Create a ROS workspace and add this repository to 'src' directory

2) Add navigation stack to 'src' directory :
```sh
$ git clone https://github.com/ros-planning/navigation.git
$ cd navigation
$ git checkout origin/melodic-devel
```

3) Add husky drivers to 'src' directory :
```sh
$ git clone https://github.com/uob-erl/husky.git
$ cd husky
$ git checkout origin/learning_effect_exp
```

4) Add package related to simulating robots in Gazebo to 'src' directory :
```sh
$ git clone https://github.com/uob-erl/erl_gazebo.git
```

5) Add fuzzy_MI_controller to 'src' directory :
```sh
$ git clone https://github.com/uob-erl/fuzzy_mi_controller.git
$ cd fuzzy_mi_controller
$ git checkout origin/operator_intent
```

6) Build the workspace with `$ catkin_make`

# Running the simulation
To run the simulated Husky mobile robot and intent recognition (assuming ROS and Gazebo are correctly installed and working) :
```sh
$ roslaunch experiments_launch husky_gazebo_mi_experiment.launch
$ rosrun bayesian_operator_intent XXXXXX.py
```

# bayesian_operator_intent/bayes_scenarios/
Folder containing python scripts XXXXXX.py that are intended to be directly executed according to the area of the map the user is interested in performing intent recognition.
HERE recursive Bayesian intent recognition is performed on-the-fly. 

# bayesian_operator_intent/values_scripts/
Folder containing python scripts XXXXXX.py that are intended to be directly executed according to the area of the map the user is interested in. 
NOT intent recognition is performed HERE. These scripts compute the necessary values/data that can be collected through 'rosbag functionality' and used in OFFLINE scripts to perform intent recognition and evaluate the algorithm (check https://github.com/dimipan/bayesian_experiment_offline).

# bayesian_operator_intent/main_folder/
Folder containing the updated version of the code (written in a clear/compact way) for future use.



