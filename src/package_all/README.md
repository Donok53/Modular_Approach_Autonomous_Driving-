# package_all

## 1. Requirements
### lio-localizer
- [gtsam](https://gtsam.org/get_started/)
    ```
    $ sudo apt-get install -y software-properties-common
    $ sudo add-apt-repository ppa:borglab/gtsam-release-4.0
    $ sudo apt-get update
    $ sudo apt-get install -y libgtsam-dev libgtsam-unstable-dev
    ```
- [Ceres-2.1.0](http://ceres-solver.org/installation.html)
    ```
    # Requirements
    $ sudo apt-get install -y cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
    ```
    ```
    $ mkdir ~/ceres_ws && cd ~/ceres_ws
    $ git clone -b 2.1.0 --single-branch https://ceres-solver.googlesource.com/ceres-solver
    $ mkdir ceres-bin && cd ceres-bin
    $ cmake ../ceres-solver
    $ make -j3
    $ make test
    $ make install
    ```
- [nanoflann](https://github.com/jlblancoc/nanoflann.git)
    ```
    $ mkdir ~/nanoflann_ws && cd ~/nanoflann_ws
    $ git clone https://github.com/jlblancoc/nanoflann.git
    $ cd nanoflann
    $ git checkout tags/v1.4.3
    $ mkdir build && cd build && cmake ..
    $ make
    $ sudo make install
    ```
### astar_map
    ```
    $ pip install utm
    ```
### all
- [ROS](http://wiki.ros.org/ROS/Installation) (tested with noetic).
    ```
    $ sudo apt-get install -y curl lsb-release gnupg g++
    $ sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    $ curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
    $ sudo apt-get update
    $ sudo apt-get install -y ros-noetic-desktop ros-noetic-pcl-conversions
    ```
## 2. install
    ```
    $ cd ~/catkin_ws/src
    $ git clone http://git.aimlab.co.kr/wave-ai/package_all.git
    $ cd ~/catkin_ws
    $ catkin_make -DCMAKE_BUILD_TYPE=Release
    ```

