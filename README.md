# **Camera Model **

---

## **Requirements**

- **Python Version:** 3.8+ recommended (works with 3.9, 3.10, 3.11)  
  Verify version with:  
  ```bash
  python --version
Required Packages:

 ```bash
  pip install numpy opencv-python matplotlib\
```

## Project Structure
project_root/
│
├─ run.py
├─ utils.py
│
├─ task1/
│  ├─ pts2d.txt
│  ├─ pts3d.txt
│  └─ task1.py
│
└─ task23/
   ├─ task23.py
   ├─ temple/
   │  ├─ data.npz
   │  ├─ im1.png
   │  └─ im2.png
   ├─ ztrans/
   │  ├─ data.npz
   │  ├─ im1.png
   │  └─ im2.png
   └─ ...
Running the Script
Open a terminal and navigate to the project root (the folder containing run.py, task1/, task23/, utils.py).

Basic command:

```bash
  python run.py --input_folder . --output_folder results
```
With specific image extension (e.g., png):

```bash
python main.py --input_folder . --output_folder results --image_ext png
```
## Example Output Files
Task 2 & 3 (Fundamental Matrix, Epipoles, Triangulation):
temple_F.txt, temple_F_raw.txt — Fundamental matrix (normalized and raw)
temple_epipole_e1.txt, temple_epipole_e2.txt — Epipoles (homogeneous coordinates)
temple_epipolar.png — Epipolar line overlay (requires im1.png and im2.png)
temple_pcd.txt, temple_pcd.png — Triangulated 3D points and visualization (requires K1 and K2 if camera intrinsics are provided)

## Task 1 (Projection Matrix & Reprojection Errors):

task1_P.txt — Estimated camera projection matrix
task1_reproj_errors.txt — Reprojection errors for each 2D-3D point correspondence


```bash
pip install --upgrade opencv-python
```
---
