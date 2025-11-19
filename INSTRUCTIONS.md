# Complete ROS Project Guide - Master Reference

**Course:** CSC 752 - Machine Intelligence  
**Environment:** Ubuntu 20.04, ROS Noetic, Isaac Sim, Docker  
**Workspace:** `~/catkin_ws`  

---

## 🖥️ Environment Overview

### System Setup
- **OS:** Ubuntu 20.04.6 LTS
- **ROS Version:** ROS Noetic
- **Python:** Python 3.8+
- **Simulator:** NVIDIA Isaac Sim
- **Robot:** Toyota HSR (Human Support Robot)
- **Container:** Docker (user: csc752)

### Workspace Structure
```
~/catkin_ws/
├── src/                          # Source code (all packages here)
│   ├── robot_localization/       # EKF/UKF package (Assignment 4)
│   ├── hsrb_rosnav/              # HSR navigation configs
│   ├── hsr-omniverse/            # Isaac Sim integration
│   ├── cscXXX-final-project
│   └── ...
├── build/                        # Build artifacts (auto-generated)
├── devel/                        # Development space (auto-generated)
│   └── setup.bash               # Source this after compilation
└── logs/                         # ROS logs

/home/csc752/hsr_robocanes_omniverse/
└── cscXXX-final-project-start.py         # Isaac Sim start scripts
```

---

## 🔧 Basic ROS Commands

### Essential Commands (Copy-Paste Ready)

#### 1. Start ROS Core
```bash
# ALWAYS run this first in a dedicated terminal
roscore
```
**When to use:** Start of every ROS session  
**Keep running:** Yes, never close this terminal  
Or instead of roscore we can include evrerything in the roslaunch (as needed)

---

#### 2. Navigation Commands
```bash


# Navigate to src directory
cd src/

# Navigate to specific package
cd ~/catkin_ws/src/cscXXX-final-project-start.py

# Quick navigation using relative paths
cd src/                    # From catkin_ws
cd ../..                   # Go up two directories
```

---

#### 3. Compilation Commands
```bash
# Compile entire workspace
cd ..
c

# Source after compilation (ALWAYS DO THIS)
s

```

**Shortcuts:**
- `c` = compile (if you create alias: `alias c='cd ~/catkin_ws && catkin_make && source devel/setup.bash'`)
- `s` = source (alias: `alias s='source ~/catkin_ws/devel/setup.bash'`)

---

#### 4. Package Management
```bash
# Find package location
rospack find package_name

# List all packages
rospack list

# List package dependencies
rospack depends package_name

# Check if package exists
rospack find robot_localization
```

---

#### 5. Launch Files
```bash
# Launch a launch file
roslaunch package_name launch_file.launch

# Examples:
roslaunch hsr-omniverse hsr.launch
roslaunch assignment-4-pstepanovum hsrb_bag.launch

# Launch with arguments
roslaunch package_name file.launch arg1:=value1 arg2:=value2

# Check launch file syntax
roslaunch --check package_name file.launch
```

---

#### 6. Running Nodes
```bash
# Run a Python script as ROS node
rosrun package_name script_name.py

# Examples:
rosrun assignment-4-pstepanovum predefposes.py
rosrun assignment-5-pstepanovum hsr_pf_localization.py

# Make script executable first
chmod +x scripts/script_name.py

# Run Python script directly
python3 scripts/script_name.py
cd src/package_name
python3 assignment5-start.py
```

---

#### 7. Topic Commands
```bash
# List all active topics
rostopic list

# Echo topic messages (see data)
rostopic echo /topic_name
rostopic echo /hsrb/odom
rostopic echo /odometry/filtered

# Show topic info
rostopic info /topic_name

# Show message rate (Hz)
rostopic hz /hsrb/odom

# Show topic type
rostopic type /hsrb/odom

# Publish to topic (testing)
rostopic pub /topic_name message_type "data"
```

---

#### 8. Node Commands
```bash
# List active nodes
rosnode list

# Get node info
rosnode info /node_name
rosnode info /ekf_localization

# Kill a node
rosnode kill /node_name

# Ping a node
rosnode ping /node_name
```

---

#### 9. Service Commands
```bash
# List services
rosservice list

# Call a service
rosservice call /service_name

# Get service type
rosservice type /service_name
```

---

#### 10. Parameter Server
```bash
# List parameters
rosparam list

# Get parameter value
rosparam get /parameter_name
rosparam get /use_sim_time

# Set parameter
rosparam set /parameter_name value
rosparam set /use_sim_time true

# Load parameters from file
rosparam load config/params.yaml

# Dump parameters to file
rosparam dump output.yaml
```

---

#### 11. Bag Files (Recording/Playback)
```bash
# Record topics to bag file
rosbag record -O output.bag /topic1 /topic2

# Record all topics
rosbag record -a

# Play bag file
rosbag play bagfile.bag

# Play with clock (for simulation time)
rosbag play bagfile.bag --clock

# Get bag info
rosbag info bagfile.bag

# Play at different speed
rosbag play bagfile.bag -r 2.0  # 2x speed
rosbag play bagfile.bag -r 0.5  # 0.5x speed
```

---

#### 12. Visualization Tools
```bash
# RViz (3D visualization)
rviz
rviz -d config_file.rviz

# rqt (Qt-based tools)
rqt

# rqt_multiplot (plotting)
rqt_multiplot

# rqt_graph (node graph)
rqt_graph

# rqt_console (log viewer)
rqt_console

# TF tree viewer
rosrun rqt_tf_tree rqt_tf_tree

# Image viewer
rosrun image_view image_view image:=/camera/image_raw
```

---

#### 13. Debugging Commands
```bash
# Check ROS environment
printenv | grep ROS

# Check master URI
echo $ROS_MASTER_URI

# Check if roscore is running
rostopic list

# Check computation graph
rqt_graph

# View logs
roscd log
tail -f latest.log

# Clean logs
rosclean check
rosclean purge
```

---

#### 14. Process Management
```bash
# Kill all ROS processes (clean restart)
pkill -9 -f ros

# Kill specific processes
pkill -9 roscore
pkill -9 roslaunch
pkill -9 rviz

# Check running ROS processes
ps aux | grep ros

# Kill by PID
kill -9 <PID>
```

---

#### 15. Git Commands (for assignments)
```bash
# Clone assignment
git clone https://classroom.github.com/a/ASSIGNMENT_ID

# Check status
git status

# Add files
git add .
git add specific_file.py

# Commit
git commit -m "Descriptive message"

# Push
git push origin main

# Pull updates
git pull origin main

# Check branch
git branch

# Create new branch
git checkout -b new_branch_name
```

---

## 📁 Project Structure

### Standard ROS Package Structure
```
package_name/
├── CMakeLists.txt           # Build configuration
├── package.xml              # Package metadata and dependencies
├── README.md                # Documentation
├── requirements.txt         # Python dependencies (if any)
├── launch/                  # Launch files (.launch)
│   ├── main.launch
│   └── simulation.launch
├── config/                  # Configuration files (.yaml, .json)
│   ├── params.yaml
│   └── localization.yaml
├── scripts/                 # Python scripts (executable)
│   ├── node1.py
│   ├── node2.py
│   └── init_package.sh
├── src/                     # C++ source files (if any)
├── include/                 # C++ header files (if any)
├── msg/                     # Custom message definitions
├── srv/                     # Custom service definitions
├── bags/                    # ROS bag files
│   └── data.bag
├── maps/                    # Map files (.pgm, .yaml)
│   ├── map.pgm
│   └── map.yaml
├── urdf/                    # Robot descriptions
├── rviz/                    # RViz config files
└── docs/                    # Additional documentation
```

---

## 🔄 Typical Workflow

### Pattern 1: New Assignment Setup
```bash
# Step 1: Navigate and clone
cd ~/catkin_ws/src
git clone https://classroom.github.com/a/ASSIGNMENT_ID

# Step 2: Navigate into package
cd assignment-X-username

# Step 3: Initialize package (if init script exists)
chmod +x scripts/init_package.sh
bash scripts/init_package.sh
# OR
python3 scripts/init_package.py

# Step 4: Install dependencies (if requirements.txt exists)
pip3 install -r requirements.txt

# Step 5: Compile
cd ~/catkin_ws
catkin_make

# Step 6: Source
source devel/setup.bash

# Step 7: Verify
rospack find assignment-X-username
```

---

### Pattern 2: Running Simulation with Isaac Sim

**Terminal Layout (5 terminals):**

**Terminal 1: ROS Core**
```bash
roscore
```

**Terminal 2: Isaac Sim**
```bash
cd /home/csc752/hsr_robocanes_omniverse
python3 assignmentX-start.py
```
Wait ~2 minutes for Isaac Sim to load

**Terminal 3: Verify System**
```bash
rosnode list          # Check nodes
rostopic list         # Check topics
rostopic echo /hsrb/odom -n 1  # Test data
```

**Terminal 4: Run Scripts/Nodes**
```bash
cd ~/catkin_ws
source devel/setup.bash
rosrun package_name script.py
```

**Terminal 5: Visualization**
```bash
rviz
# OR
rqt_multiplot
```

---

### Pattern 3: Rosbag Playback (No Isaac Sim)

**Terminal 1: Launch Everything**
```bash
cd ~/catkin_ws
source devel/setup.bash
roslaunch package_name launch_file.launch
```
This typically includes:
- Setting `use_sim_time=true`
- Loading robot description
- Starting necessary nodes
- Playing bag file with `--clock`

**Terminal 2: Visualization**
```bash
rqt_multiplot
# Configure plot for topics
```

**Terminal 3: Monitoring**
```bash
rostopic hz /odometry/filtered
rostopic echo /odometry/filtered
```

---

### Pattern 4: Development Cycle
```bash
# 1. Edit code
nano scripts/my_script.py
# OR use VS Code, vim, etc.

# 2. Save changes

# 3. Compile (if needed for C++ or CMakeLists changes)
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# 4. Test
rosrun package_name my_script.py

# 5. Debug if needed
rostopic echo /debug_topic
rosnode info /my_node

# 6. Repeat 1-5 until working

# 7. Commit
git add scripts/my_script.py
git commit -m "Implement feature X"
git push origin main
```

---

## 🎯 Assignment Patterns

### Assignment 4: EKF Localization

**What it does:**
- Uses Extended Kalman Filter for robot localization
- Fuses odometry and IMU data
- Outputs filtered pose estimate

**Key topics:**
- Input: `/hsrb/odom`, `/hsrb/base_imu/data`
- Output: `/odometry/filtered`

**Workflow:**
1. Setup robot_localization package
2. Configure localization.yaml
3. Create launch file for bag playback
4. Plot trajectories in rqt_multiplot
5. Take screenshots

**Commands:**
```bash
# Part A: Live simulation
roslaunch hsr-omniverse hsr.launch
rosrun assignment-4-username predefposes.py
rqt_multiplot  # Plot /hsrb/odom

# Part B: Rosbag
roslaunch assignment-4-username hsrb_bag.launch
rqt_multiplot  # Plot /odometry/filtered
```

---

### Assignment 5: Particle Filter Localization

**What it does:**
- Implements particle filter for localization
- Uses laser scan and odometry
- Estimates robot pose in known map

**Key functions to implement:**
1. Measurement model (particle weights)
2. Resampling (universal stochastic)
3. Mean position (state estimate)

**Workflow:**
1. Clone assignment
2. Implement three functions
3. Test with Isaac Sim
4. Answer theory questions
5. Submit

**Commands:**
```bash
cd ~/catkin_ws/src/assignment-5-username
python3 assignment5-start.py

# Watch in RViz:
# - Particles converge
# - Yellow arrow tracks robot
```

---

## 🔧 Common Issues & Solutions

### Issue 1: "Unable to communicate with master"
**Cause:** roscore not running  
**Fix:**
```bash
# Terminal 1
roscore
```

---

### Issue 2: "Package not found"
**Cause:** Not compiled or not sourced  
**Fix:**
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rospack find package_name
```

---

### Issue 3: "Permission denied" on script
**Cause:** Script not executable  
**Fix:**
```bash
chmod +x scripts/script_name.py
```

---

### Issue 4: Multiple move_base nodes conflict
**Cause:** Isaac Sim and launch file both starting navigation  
**Fix:**
- Use ONLY Isaac Sim for live simulation
- Use launch file ONLY for rosbag playback
- Don't mix them!

---

### Issue 5: "No messages on topic"
**Causes & Fixes:**

**For live simulation:**
```bash
# Check if robot is publishing
rostopic hz /hsrb/odom
# If no data, check Isaac Sim is running and robot is loaded
```

**For rosbag:**
```bash
# Check use_sim_time is set
rosparam get /use_sim_time  # Should be "true"

# Check bag is playing with --clock
rosbag play file.bag --clock
```

---

### Issue 6: rqt_multiplot not showing data
**Fix:**
```bash
# 1. Make sure topic exists
rostopic list | grep topic_name

# 2. Check topic is publishing
rostopic hz /topic_name

# 3. In rqt_multiplot, click Play ▶️ button

# 4. Check topic name is exact (case-sensitive)
# Correct: /odometry/filtered
# Wrong: /odometry/Filtered
```

---

### Issue 7: Import errors in Python
**Fix:**
```bash
# Install missing packages
pip3 install package_name

# Check if requirements.txt exists
pip3 install -r requirements.txt

# Verify installation
python3 -c "import package_name; print('OK')"
```

---

### Issue 8: Git push fails
**Cause:** Authentication or wrong remote  
**Fix:**
```bash
# Check remote
git remote -v

# Set up credentials (use token not password)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Use HTTPS with token or SSH key
```

---

### Issue 9: Catkin_make fails
**Causes & Fixes:**

**Missing dependencies:**
```bash
# Install ROS package
sudo apt-get install ros-noetic-package-name

# Check package.xml for dependencies
cat package.xml | grep depend
```

**Syntax error in CMakeLists.txt:**
```bash
# Check for typos, missing parentheses
nano CMakeLists.txt
```

**Corrupted build:**
```bash
cd ~/catkin_ws
catkin_make clean
catkin_make
```

---

### Issue 10: RViz crashes or freezes
**Fix:**
```bash
# Kill and restart
pkill -9 rviz
rviz

# Reset config
rm ~/.rviz/default.rviz
rviz
```

---

## 🛠️ Tools & Utilities

### RViz Configuration

**Add displays for robot localization:**
1. RobotModel - visualize robot
2. TF - coordinate frames
3. LaserScan - topic: `/hsrb/base_scan`
4. Odometry - topic: `/hsrb/odom`
5. Odometry - topic: `/odometry/filtered`
6. Map - topic: `/map`
7. Path - for trajectories

**Set Fixed Frame:** `map` or `odom`

---

### rqt_multiplot Configuration

**For trajectory plotting:**

**Part A (live sim):**
- Topic: `/hsrb/odom`
- X field: `pose/pose/position/x`
- Y field: `pose/pose/position/y`

**Part B (rosbag with EKF):**
- Topic: `/odometry/filtered`
- X field: `pose/pose/position/x`
- Y field: `pose/pose/position/y`

**Save config:**
- File → Save Configuration
- Load with: `rqt_multiplot --multiplot-config file.xml`

---

### Screenshot Tools
```bash
# Entire screen
gnome-screenshot

# Select window
gnome-screenshot -w

# Select area
gnome-screenshot -a

# Delayed screenshot
gnome-screenshot -d 5

# Save to specific location
gnome-screenshot -f ~/catkin_ws/src/package/screenshots/image.png
```

---

## 📋 Quick Reference

### Essential 3-Step Pattern

**1. Setup**
```bash
cd ~/catkin_ws/src
git clone REPO_URL
cd package_name
bash scripts/init_package.sh  # if exists
pip3 install -r requirements.txt  # if exists
```

**2. Compile**
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

**3. Run**
```bash
# Option A: Live simulation
roscore                          # Terminal 1
python3 assignmentX-start.py     # Terminal 2
rosrun package_name script.py    # Terminal 3

# Option B: Launch file
roslaunch package_name file.launch
```

---

### Most Common Commands (Daily Use)

```bash
# Navigation
cd ~/catkin_ws
cd src/package_name

# Build
catkin_make
source devel/setup.bash

# Run
roscore
roslaunch package_name file.launch
rosrun package_name script.py

# Debug
rostopic list
rostopic echo /topic_name
rosnode list
rosnode info /node_name

# Visualize
rviz
rqt_multiplot
rqt_graph

# Git
git status
git add .
git commit -m "message"
git push origin main
```

---

### File Paths Quick Reference

```bash
# Catkin workspace
~/catkin_ws/

# Your packages
~/catkin_ws/src/assignment-X-username/

# Isaac Sim scripts
/home/csc752/hsr_robocanes_omniverse/

# ROS logs
~/.ros/log/

# RViz configs
~/.rviz/

# Common configs
~/catkin_ws/src/package/config/
~/catkin_ws/src/package/launch/
```

---

### Topic Name Patterns

**Odometry topics:**
- `/hsrb/odom` - Raw wheel odometry
- `/odometry/filtered` - EKF filtered odometry
- `/hsrb/laser_odom` - Laser-based odometry

**Sensor topics:**
- `/hsrb/base_scan` - Laser scan
- `/hsrb/base_imu/data` - IMU data
- `/hsrb/head_rgbd_sensor/rgb/image_rect_color` - Camera

**Navigation topics:**
- `/move_base/goal` - Navigation goals
- `/move_base/result` - Navigation results
- `/cmd_vel` or `/hsrb/command_velocity` - Velocity commands

**Mapping topics:**
- `/map` - Occupancy grid map
- `/map_metadata` - Map metadata

**TF topics:**
- `/tf` - Dynamic transforms
- `/tf_static` - Static transforms

---

## 🎓 Learning Resources

**ROS Wiki:**
- http://wiki.ros.org/
- http://wiki.ros.org/ROS/Tutorials

**Package Documentation:**
- robot_localization: http://docs.ros.org/kinetic/api/robot_localization/html/
- navigation: http://wiki.ros.org/navigation
- tf: http://wiki.ros.org/tf

**Particle Filter Resources:**
- Probabilistic Robotics (Thrun, Burgard, Fox)
- Particle Filter tutorials online

---

## 💡 Pro Tips

### Efficiency Tips

**1. Create aliases in ~/.bashrc:**
```bash
alias cw='cd ~/catkin_ws'
alias cs='cd ~/catkin_ws/src'
alias cm='cd ~/catkin_ws && catkin_make && source devel/setup.bash'
alias s='source ~/catkin_ws/devel/setup.bash'
alias c='catkin_make'
```

**2. Use tab completion:**
- Type `rosrun ass` then press TAB → autocompletes package name
- Type `cd src/ass` then press TAB → autocompletes directory

**3. Use terminal multiplexers:**
```bash
# Install tmux
sudo apt-get install tmux

# Split terminal windows instead of opening multiple terminals
```

**4. Keep a terminal cheat sheet:**
- Bookmark common commands
- Keep this README open while working

---

### Debugging Strategy

**1. Check system status:**
```bash
rostopic list    # Are topics available?
rosnode list     # Are nodes running?
rqt_graph        # Is graph connected correctly?
```

**2. Check data flow:**
```bash
rostopic hz /topic_name     # Is data being published?
rostopic echo /topic_name   # What does data look like?
```

**3. Check logs:**
```bash
rqt_console     # GUI log viewer
roscd log       # Navigate to log directory
tail -f latest.log  # Watch logs in real-time
```

**4. Isolate the problem:**
- Run components one at a time
- Check each step before proceeding
- Use echo to verify data at each stage

---

**Standard submission commands:**
```bash
cd /src/assignment-X-username
git status
git add .
git commit -m "Complete Assignment X: [brief description]"
git push origin main

# Verify on GitHub:
# Open browser → GitHub Classroom → Check files are there
```

---

## 📞 Getting Help

**When stuck:**

1. **Read error messages carefully**
   - Most errors tell you exactly what's wrong
   
2. **Check this README**
   - Common issues section
   - Command reference
   
3. **Google the error**
   - Copy exact error message
   - Search with "ROS" keyword
   
4. **Check ROS Answers**
   - https://answers.ros.org/
   
5. **Ask for help with context**
   - What you're trying to do
   - What command you ran
   - Complete error message
   - What you've already tried

---

## 🎯 Summary: The Universal Workflow

**Every ROS project follows this pattern:**

```bash
# 1. SETUP
cd ~/catkin_ws/src
git clone REPO_URL
cd package_name
# Run initialization if needed
# Install dependencies if needed

# 2. BUILD
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# 3. VERIFY
rospack find package_name

# 4. RUN
# Start roscore
# Start simulation OR launch file
# Run your nodes
# Visualize results

# 5. DEVELOP
# Edit code
# Compile if needed
# Test
# Debug
# Repeat

# 6. SUBMIT
git add .
git commit -m "message"
git push origin main
```

**Remember:** Read assignment requirements → Setup → Implement → Test → Submit

---

**End of Master README**