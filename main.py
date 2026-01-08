#bscs23144
#Khadija MAsood
#Assignment 5


import os
import sys
import argparse
import traceback
import numpy as np
import cv2

# Import your modules (they must be in PYTHONPATH or same folder)
import task1.task1 as task1
import task23.task23 as task23
import utils


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_task1(task1_input_dir, out_dir):
    """
    Runs Task 1: estimate projection matrix P from pts3d and pts2d
    Expects files:
      task1/pts2d.txt  (N x 2)
      task1/pts3d.txt  (N x 3)
    Saves:
      results/task1_P.txt
      results/task1_reproj_errors.txt
    """
    print("\n=== TASK 1: Projection matrix estimation ===")
    pts2d_path = os.path.join(task1_input_dir, "pts2d.txt")
    pts3d_path = os.path.join(task1_input_dir, "pts3d.txt")
    if not (os.path.exists(pts2d_path) and os.path.exists(pts3d_path)):
        print("Skipping Task1: pts2d.txt or pts3d.txt not found in", task1_input_dir)
        return

    try:
        pts2d = np.loadtxt(pts2d_path)
        pts3d = np.loadtxt(pts3d_path)
        if pts2d.ndim != 2 or pts2d.shape[1] != 2:
            raise ValueError("pts2d.txt must be Nx2")
        if pts3d.ndim != 2 or pts3d.shape[1] != 3:
            raise ValueError("pts3d.txt must be Nx3")

        P = task1.find_projection(pts2d, pts3d)
        mean_err, dists = task1.reprojection_error(P, pts3d, pts2d)

        # Save results
        ensure_dir(out_dir)
        np.savetxt(os.path.join(out_dir, "task1_P.txt"), P, fmt="%.8f")
        np.savetxt(os.path.join(out_dir, "task1_reproj_errors.txt"), dists, fmt="%.8f")
        with open(os.path.join(out_dir, "task1_summary.txt"), "w") as f:
            f.write(f"P (3x4):\n{P}\n\n")
            f.write(f"Mean reprojection error (px): {mean_err:.8f}\n")
            f.write(f"Saved per-point errors to task1_reproj_errors.txt\n")

        print("Task1 completed. P saved to:", os.path.join(out_dir, "task1_P.txt"))
        print("Mean reprojection error (px):", mean_err)
    except Exception:
        print("Error running Task1:")
        traceback.print_exc()


def safe_load_npz(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        print("Failed to load npz:", path)
        traceback.print_exc()
        return None


def run_task2_and_task3(task23_root, out_dir, image_ext="png"):
    """
    Runs Task 2: fundamental matrices + epipoles + epipolar overlays
         Task 3: triangulation + point cloud visualizations
    Expects:
      task23_root/<dataset>/data.npz  (with keys pts1, pts2 and optionally K1, K2)
      and images im1.png/im2.png (optional)
    Saves outputs to results/<dataset>_*.*
    """
    print("\n=== TASK 2 & TASK 3: Fundamental matrix, epipoles, triangulation ===")

    if not os.path.isdir(task23_root):
        print("task23 root not found:", task23_root)
        return

    datasets = sorted([d for d in os.listdir(task23_root) if os.path.isdir(os.path.join(task23_root, d))])
    if not datasets:
        print("No datasets found under", task23_root)
        return

    for name in datasets:
        ds_dir = os.path.join(task23_root, name)
        npz_path = os.path.join(ds_dir, "data.npz")
        print("\n--- Dataset:", name)
        if not os.path.exists(npz_path):
            print("  data.npz not found, skipping dataset.")
            continue

        data = safe_load_npz(npz_path)
        if data is None:
            continue

        keys = list(data.keys())
        print("  Keys in data.npz:", keys)
        if "pts1" not in data or "pts2" not in data:
            print("  pts1/pts2 missing, skipping.")
            continue

        pts1 = data["pts1"].astype(float)
        pts2 = data["pts2"].astype(float)
        out_prefix = os.path.join(out_dir, name)

        # Task 2: compute F
        try:
            F = task23.find_fundamental_matrix(None, pts1, pts2)
            # normalize for reporting (F[1,1] == 1 in 0-based indexing)
            Fnorm = F.copy().astype(float)
            if abs(Fnorm[1, 1]) > 1e-12:
                Fnorm = Fnorm / Fnorm[1, 1]
            else:
                # fallback to Frobenius normalization
                Fnorm = Fnorm / np.linalg.norm(Fnorm)

            np.savetxt(out_prefix + "_F.txt", Fnorm, fmt="%.8f")
            np.savetxt(out_prefix + "_F_raw.txt", F, fmt="%.8f")
            print("  Saved F to:", out_prefix + "_F.txt")
        except Exception:
            print("  Error computing F for dataset", name)
            traceback.print_exc()
            continue

        # Task 2: compute epipoles
        try:
            e1, e2 = task23.compute_epipoles(Fnorm)
            np.savetxt(out_prefix + "_epipole_e1.txt", e1.reshape(1, -1), fmt="%.12e")
            np.savetxt(out_prefix + "_epipole_e2.txt", e2.reshape(1, -1), fmt="%.12e")
            print("  Epipoles saved:", out_prefix + "_epipole_e1.txt", out_prefix + "_epipole_e2.txt")
        except Exception:
            print("  Error computing epipoles for dataset", name)
            traceback.print_exc()
            e1 = e2 = None

        # Task 2: draw epipolar overlays (if images exist)
        im1p = os.path.join(ds_dir, f"im1.{image_ext}")
        im2p = os.path.join(ds_dir, f"im2.{image_ext}")
        if os.path.exists(im1p) and os.path.exists(im2p):
            try:
                im1 = cv2.imread(im1p, cv2.IMREAD_GRAYSCALE)
                im2 = cv2.imread(im2p, cv2.IMREAD_GRAYSCALE)
                out_img_path = out_prefix + "_epipolar.png"
                utils.draw_epipolar(im1, im2, Fnorm, pts1, pts2, epi1=e1, epi2=e2, filename=out_img_path)
                print("  Saved epipolar overlay to:", out_img_path)
            except Exception:
                print("  Error drawing epipolar overlay for", name)
                traceback.print_exc()
        else:
            print("  Images not found for epipolar overlay (expected):", im1p, im2p)

        # Task 3: triangulation if K1 and K2 present
        K1 = data["K1"] if "K1" in data else None
        K2 = data["K2"] if "K2" in data else None
        if K1 is None or K2 is None:
            print("  K1/K2 not present in data.npz — skipping triangulation for", name)
            continue

        try:
            pcd = task23.find_triangulation(K1.astype(float), K2.astype(float), F, pts1, pts2)
            if pcd is None:
                print("  Triangulation returned None for dataset", name)
            else:
                # Save pcd
                np.savetxt(out_prefix + "_pcd.txt", pcd, fmt="%.8f")
                # save a visualization
                try:
                    utils.visualize_pcd(pcd.T, filename=out_prefix + "_pcd.png")
                    print("  Saved point cloud image:", out_prefix + "_pcd.png")
                except Exception:
                    print("  Failed to save pcd image for", name)
                print("  Saved triangulation text:", out_prefix + "_pcd.txt")
        except Exception:
            print("  Error during triangulation for dataset", name)
            traceback.print_exc()
            continue


def create_arg_parser():
    p = argparse.ArgumentParser(description="Run Tasks 1-3: Projection, Fundamental, Triangulation pipeline.")
    p.add_argument("--input_folder", required=True, help="Path to project/input folder (contains task1/, task23/).")
    p.add_argument("--output_folder", required=True, help="Path to put all result files.")
    p.add_argument("--image_ext", default="png", help="Image extension used in datasets (default: png).")
    return p


def main_cli():
    parser = create_arg_parser()
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    img_ext = args.image_ext.lstrip(".")

    if not os.path.isdir(input_folder):
        print("Input folder not found:", input_folder)
        sys.exit(1)

    ensure_dir(output_folder)

    # Task1 input and results
    task1_in = os.path.join(input_folder, "task1")
    task1_out = os.path.join(output_folder, "task1")
    ensure_dir(task1_out)
    run_task1(task1_in, task1_out)

    # Task23 root and results
    task23_root = os.path.join(input_folder, "task23")
    task23_out = os.path.join(output_folder, "task23")
    ensure_dir(task23_out)
    run_task2_and_task3(task23_root, task23_out, image_ext=img_ext)

    print("\nAll tasks finished. Results written to:", output_folder)


if __name__ == "__main__":
    main_cli()