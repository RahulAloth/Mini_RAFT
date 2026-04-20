# Mini_RAFT
![Stars](https://img.shields.io/github/stars/RahulAloth/Mini_RAFT?style=flat-square)
![Forks](https://img.shields.io/github/forks/RahulAloth/Mini_RAFT?style=flat-square)
![License](https://img.shields.io/github/license/RahulAloth/Mini_RAFT?style=flat-square)
![Issues](https://img.shields.io/github/issues/RahulAloth/Mini_RAFT?style=flat-square)

## 📘 Project Description

The idea for this project came from a practical engineering challenge during the development of my **Robosense sensor fusion system**.  
To build and test a full **Spatio‑Temporal Aligner** and multi‑sensor fusion logic, I needed a realistic multi‑sensor environment.  

However, at the time, I only had **two stereo cameras** available — no LiDAR.

Purchasing a LiDAR (especially a Robosense unit) during early development was not feasible, so I explored an alternative approach:

### **Can a stereo camera + CNN simulate LiDAR depth?**

This project is the result of that idea.

---

## 🎯 Input vs Output (Mini-RAFT)

<table>
  <tr>
    <td align="center"><b>Input Image</b></td>
    <td align="center"><b>Model Output</b></td>
  </tr>
  <tr>
    <td><img src="/data/data_scene_flow/testing/image_3/000000_10.png" width="420"></td>
    <td><img src="/data/data_scene_flow/output.png" width="420"></td>
  </tr>
</table>

---

## 🔄 Integration With RoboSense Fusion Project

The LiDAR‑like simulation developed in this project will be integrated into my larger **RoboSense Fusion** system:  
👉 https://github.com/RahulAloth/RoboSense-Fusion

The goal is to create a complete multi‑sensor fusion environment **without requiring a physical LiDAR** during early development.  
The stereo‑based virtual LiDAR will be packaged into a **ROS2‑compatible message format** (sensor_msgs/PointCloud2), allowing it to be published as a real LiDAR topic.

This enables:

- RViz2 visualization of the simulated LiDAR stream  
- Plug‑and‑play compatibility with existing LiDAR fusion pipelines  
- Development of the **SpatioTemporalAligner**  
- Testing of multi‑sensor synchronization and calibration  
- Early fusion logic development before real hardware arrives  

Once integrated, the stereo camera will behave like a virtual LiDAR inside ROS2, allowing the RoboSense Fusion project to run full perception and fusion stacks using only stereo input.

--- 

## 🎯 Motivation

During fusion development, I needed:

- A depth source similar to LiDAR  
- A way to test fusion logic without real LiDAR hardware  
- A method to generate 3D structure (XYZ) from stereo  
- A pipeline that runs on **reComputer J40** (Jetson Orin NX)  
- A lightweight model that can be trained on CPU  

Traditional RAFT‑Stereo is extremely accurate, but:

- too heavy for Jetson  
- too slow for real‑time  
- too expensive to train  
- not practical for CPU‑only training  

So I designed a **Mini‑RAFT** model — a compact version of RAFT‑Stereo — optimized for:

- low memory  
- CPU training  
- Jetson deployment  
- real‑time inference  
- KITTI‑style stereo depth  

---

## 🧠 Approach

The project uses a custom **Mini‑RAFT Stereo** network to estimate disparity from stereo images.  
The disparity is then converted into:

- **Depth map**  
- **Point cloud (XYZ)**  
- **LiDAR‑style front‑view projection**  

This allows a stereo camera to behave like a **virtual LiDAR**, enabling:

- fusion algorithm development  
- temporal alignment testing  
- 3D perception prototyping  
- multi‑sensor simulation  

without needing a physical LiDAR sensor.

---

## 📚 Training Data

The model is trained on:

### **KITTI Stereo 2015 Dataset**  
(Official link: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

### **Scene Flow Dataset**  
(Official link: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Scene Flow provides large synthetic stereo data for pretraining.  
KITTI provides real‑world fine‑tuning.

Both datasets include:

- left/right stereo images  
- ground‑truth disparity  
- camera intrinsics  

This makes them ideal for training a stereo depth CNN.

---

## 🏋️ Training Pipeline

The training algorithm includes:

- Stereo‑consistent geometric augmentations  
- Appearance augmentations (ColorJitter)  
- Mini‑RAFT forward pass  
- Multi‑scale disparity supervision  
- Smooth L1 loss  
- AdamW optimizer  
- CPU‑friendly batch size (1)  
- Checkpoint saving  

The entire model was trained on **CPU**, proving that a lightweight RAFT‑style network can be trained without GPU resources.

---

## 🛰️ Output Capabilities

The system produces:

- Disparity map  
- Depth map  
- Rainbow depth visualization  
- Grayscale depth visualization  
- 3D point cloud (XYZ)  
- LiDAR‑style front‑view projection  

This allows a stereo camera to act as a **virtual LiDAR**, enabling fusion development without hardware.

---

## 🧩 Why This Project Matters

This project demonstrates that:

- We can simulate LiDAR‑like depth using stereo + CNN  
- We can train a RAFT‑style model on CPU  
- We can deploy a compact stereo network on Jetson  
- We can build a multi‑sensor fusion environment without buying expensive sensors  

It is a practical, cost‑effective solution for robotics, AV research, and sensor fusion prototyping.

