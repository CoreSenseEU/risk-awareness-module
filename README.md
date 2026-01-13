# CoreSense Risk Awareness Module (RiskAM)

[CoreSense](https://coresense.eu/) is a Horizon Europe funded project that aims to develop a theory and a derived cognitive architecture for understanding in autonomous robots. A key ingredient needed to move towards true open-world autonomy in robotics is *risk awareness*: instead of trying to model the environment and all fathomable risks beforehand, which is infeasible in open-world scenarios; we make the robot itself aware of risks. This is the aim of the **CoreSense risk awareness module (RiskAM)**. You can read the full motivation, description, and prototype feature documentation of RiskAM in [CoreSense deliverable D3.5](http://zahalka.net/wp-content/uploads/2025/04/CoreSense___CS_067_D3_5__RiskAM_deliverable.pdf).

At this stage, RiskAM is a *prototype* that works for *visual navigation* and considers *risks to humans*. You can find the [full demo videos from the CoreSense RoboCup @ Home 2023 dataset here](https://drive.google.com/drive/folders/1y_I-fNZk9aPJJtIgrDVrzYYbc89Gha_P?usp=sharing).

## Installation & prerequisites
The SW was implemented for Linux (tested on Ubuntu 24.04).

### Package installation
Make sure ROS 2 with CV bridge and Rosbag 2 - Py is properly installed on your system. Currently, RiskAM is ROS 2-distro-agnostic (it was tested on `rolling`), but we reserve the right to change this in the future according to the CoreSense project specs. [[Installation guide for `rolling`]](https://docs.ros.org/en/rolling/Installation.html)

Set up a virtual environment, install the requirements: 
    cd <riskam_root_dir>
    virtualenv env_riskam
    pip install -r requirements.txt

Link the system's `rosbag2_py` to the virtualenv:

    echo "/opt/ros/<ros2_distro>/lib/python3.x/site-packages" > <riskam_root_dir>/env_riskam/lib/python3.x/site-packages/ros2.pth

Activate the environment before each use:

    source env_riskam/bin/activate

Make sure ROS 2 is sourced before extracting data from ROS 2 bags:

    source /opt/ros/<ros2_distro>/setup.bash


### Demo dataset download
The usage will be demonstrated on the CoreSense RoboCup @ Home 2023 dataset [[Download link]](https://zenodo.org/records/13902513). Download the `2023_Dataset.zip` file to a location of your choice and unzip it.

### Image extraction from ROS 2 bags
RiskAM expects RGB images as inputs, so first we need to extract them from the provided ROS 2 bags. This requires some manual handling. First, create the `ros_datasets` directory in the project's root directory `<riskam_root_dir>`, and a `cs_robocup_2023` directory within it:

    cd <riskam_root_dir>
    mkdir -p ros_datasets/cs_robocup_2023

Then, for each run denoted `RB_##` in the dataset (you can skip `RB_04`, which is empty), unzip the respective ZIP file into the newly created directories as follows (example for `RB_01`):
    
    unzip <download_dir>/2023_Dataset/Dataset_v2/RB_01/RB_01.zip -d <riskam_root_dir>/ros_datasets/cs_robocup_2023/RB_01

For run `RB_01` specifically (not the others), you must manually uncompress the ROS 2 bags from `.zst` format:
    
    cd <riskam_root_dir>/ros_datasets/cs_robocup_2023/RB_01/mapeo1
    unzstd *.zst
    cd <riskam_root_dir>

Then, run:
    
    python scripts/extract_ros2_dataset.py cs_robocup_2023

Note that you can run this for each `RB_##` separately if you're running out of disk space; the script does not expect all runs to be present in the `ros_datasets/cs_robocup_2023` directory at once, it will extract whatever is available


## Usage
### Demo video for a specific run

Run the following script:

    python scripts/test_run_with_video.py cs_robocup_2023 RB_##

This will create the `test_results/videos` directory (if it hasn't existed already) with two videos:

- `test_results/videos/cs_robocup_2023_RB_##_raw.avi` - The actual footage of what the robot sees.
- `test_results/videos/cs_robocup_2023_RB_##_risk.avi` - The RiskAM output visualized for each frames, with the following features:
    1. Color-coded risk score in the top right corner.
    2. Bounding boxes for all detected humans.
    3. Each bounding box is color-coded according to gaze:
        - Red: Cannot detect pose or the person is likely not aware of the robot.
        - Green: The pose can be detected and the person is almost certainly aware of the robot.
        - Yellow: The pose can be detected and the person is maybe aware of the robot.
    4. The bounding box corresponding to the maximum risk score value (displayed top right) is marked with an asterisk.
    5. The points fade to dark gray with increasing (estimated) proximity.

### Experiments
To repeat the experiments discussed in the CoreSense deliverable D3.5, run:

    python scripts/run_experiments.py run cs_robocup_2023 --run RB_##

to repeat the experiments for run `RB_##`, or simply:

    scripts/run_experiments_cs_robocup_2023_all.sh

to run them all sequentially.

The output will be in the `exp_results` directory (newly created if it didn't exist already).


## Acknowledgment
This work has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101070254 CORESENSE. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the Horizon Europe programme. Neither the European Union nor the granting authority can be held responsible for them.

