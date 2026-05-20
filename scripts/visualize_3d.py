"""Visualize 3D skeleton from pipeline output .npy files.

Usage:
    uv run python scripts/visualize_3d.py stored_vids/output/calibration_20050771_keypoints3d.npy
    uv run python scripts/visualize_3d.py stored_vids/output/calibration_20050771_keypoints3d.npy --frame 500
    uv run python scripts/visualize_3d.py stored_vids/output/calibration_20050771_keypoints3d.npy --animate
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation

# bml_movi_87 joint names (indices 0-86)
JOINT_NAMES = [
    "backneck", "upperback", "clavicle", "sternum", "umbilicus",
    "lfronthead", "lbackhead", "lback", "lshom", "lupperarm",
    "lelbm", "lforearm", "lwrithumbside", "lwripinkieside", "lfin",
    "lasis", "lpsis", "lfrontthigh", "lthigh", "lknem",
    "lankm", "Left Heel", "lfifthmetatarsal", "Left Big Toe", "lcheek",
    "lbreast", "lelbinner", "lwaist", "lthumb", "lfrontinnerthigh",
    "linnerknee", "lshin", "lfirstmetatarsal", "lfourthtoe", "lscapula",
    "lbum", "rfronthead", "rbackhead", "rback", "rshom",
    "rupperarm", "relbm", "rforearm", "rwrithumbside", "rwripinkieside",
    "rfin", "rasis", "rpsis", "rfrontthigh", "rthigh",
    "rknem", "rankm", "Right Heel", "rfifthmetatarsal", "Right Big Toe", "rcheek",
    "rbreast", "relbinner", "rwaist", "rthumb", "rfrontinnerthigh",
    "rinnerknee", "rshin", "rfirstmetatarsal", "rfourthtoe", "rscapula",
    "rbum", "Head", "mhip", "Pelvis", "Sternum",
    "Left Ankle", "Left Elbow", "Left Hip", "Left Hand", "Left Knee",
    "Left Shoulder", "Left Wrist", "Left Foot", "Right Ankle", "Right Elbow",
    "Right Hip", "Right Hand", "Right Knee", "Right Shoulder", "Right Wrist",
    "Right Foot",
]

_jidx = {name: i for i, name in enumerate(JOINT_NAMES)}

# Skeleton edges using the landmark joints (indices 67-86)
EDGES_CENTER = [
    ("Head", "backneck"), ("backneck", "Sternum"), ("Sternum", "upperback"),
    ("upperback", "Pelvis"), ("Pelvis", "mhip"),
]
EDGES_LEFT = [
    ("Left Shoulder", "Left Elbow"), ("Left Elbow", "Left Wrist"),
    ("Left Wrist", "Left Hand"),
    ("Left Shoulder", "backneck"),
    ("Left Hip", "Left Knee"), ("Left Knee", "Left Ankle"),
    ("Left Ankle", "Left Foot"),
    ("Left Hip", "Pelvis"),
]
EDGES_RIGHT = [
    ("Right Shoulder", "Right Elbow"), ("Right Elbow", "Right Wrist"),
    ("Right Wrist", "Right Hand"),
    ("Right Shoulder", "backneck"),
    ("Right Hip", "Right Knee"), ("Right Knee", "Right Ankle"),
    ("Right Ankle", "Right Foot"),
    ("Right Hip", "Pelvis"),
]

EDGES_CENTER_IDX = [(_jidx[a], _jidx[b]) for a, b in EDGES_CENTER]
EDGES_LEFT_IDX = [(_jidx[a], _jidx[b]) for a, b in EDGES_LEFT]
EDGES_RIGHT_IDX = [(_jidx[a], _jidx[b]) for a, b in EDGES_RIGHT]


def find_good_frame(kp):
    """Find a frame with high confidence (non-zero keypoints)."""
    nonzero = np.any(kp[:, :, :3] != 0, axis=(1, 2))
    if not np.any(nonzero):
        return 0
    # Pick frame with highest mean confidence
    conf = kp[:, :, 3] if kp.shape[2] == 4 else np.ones(kp.shape[:2])
    conf_per_frame = conf.mean(axis=1)
    conf_per_frame[~nonzero] = -1
    return int(np.argmax(conf_per_frame))


def plot_skeleton(ax, xyz, conf=None, conf_thresh=0.05):
    """Plot a single frame's 3D skeleton."""
    ax.clear()

    if conf is not None:
        visible = conf > conf_thresh
    else:
        visible = np.ones(xyz.shape[0], dtype=bool)

    # Draw edges
    for edges, color in [(EDGES_CENTER_IDX, "black"), (EDGES_LEFT_IDX, "red"), (EDGES_RIGHT_IDX, "blue")]:
        for i, j in edges:
            if visible[i] and visible[j]:
                ax.plot([xyz[i, 0], xyz[j, 0]],
                        [xyz[i, 1], xyz[j, 1]],
                        [xyz[i, 2], xyz[j, 2]],
                        color=color, linewidth=2)

    # Draw joints
    vis_xyz = xyz[visible]
    ax.scatter(vis_xyz[:, 0], vis_xyz[:, 1], vis_xyz[:, 2],
               c="green", s=30, depthshade=True)

    # Set equal aspect ratio
    if len(vis_xyz) > 0:
        center = vis_xyz.mean(axis=0)
        max_range = (vis_xyz.max(axis=0) - vis_xyz.min(axis=0)).max() / 2 * 1.2
        for set_lim, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
            set_lim(c - max_range, c + max_range)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")


def plot_single_frame(kp, frame_idx, output_path):
    """Plot and save a single frame."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xyz = kp[frame_idx, :, :3]
    conf = kp[frame_idx, :, 3] if kp.shape[2] == 4 else None

    plot_skeleton(ax, xyz, conf)
    ax.set_title(f"Frame {frame_idx}")
    ax.view_init(elev=10, azim=-75)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def animate_skeleton(kp, output_path, step=5, max_frames=200):
    """Create an animated gif/mp4 of the skeleton over time."""
    # Find frames with detections
    nonzero = np.any(kp[:, :, :3] != 0, axis=(1, 2))
    valid_frames = np.where(nonzero)[0]
    if len(valid_frames) == 0:
        print("No valid frames to animate.")
        return

    # Subsample
    valid_frames = valid_frames[::step][:max_frames]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=-75)

    def update(i):
        frame_idx = valid_frames[i]
        xyz = kp[frame_idx, :, :3]
        conf = kp[frame_idx, :, 3] if kp.shape[2] == 4 else None
        plot_skeleton(ax, xyz, conf)
        ax.set_title(f"Frame {frame_idx}")
        ax.view_init(elev=10, azim=-75)

    anim = FuncAnimation(fig, update, frames=len(valid_frames), interval=100)

    if output_path.endswith(".gif"):
        anim.save(output_path, writer="pillow", fps=10)
    else:
        anim.save(output_path, writer="ffmpeg", fps=10)

    print(f"Saved animation: {output_path} ({len(valid_frames)} frames)")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D skeleton from .npy")
    parser.add_argument("npy_file", help="Path to keypoints3d .npy file")
    parser.add_argument("--frame", type=int, default=None, help="Frame index (default: best frame)")
    parser.add_argument("--animate", action="store_true", help="Create animation")
    parser.add_argument("--step", type=int, default=5, help="Frame step for animation")
    parser.add_argument("-o", "--output", default=None, help="Output path (default: auto)")
    args = parser.parse_args()

    kp = np.load(args.npy_file)
    print(f"Loaded: {args.npy_file} — shape {kp.shape}")

    base = args.npy_file.rsplit(".", 1)[0]

    if args.animate:
        out = args.output or f"{base}_skeleton.gif"
        animate_skeleton(kp, out, step=args.step)
    else:
        frame_idx = args.frame if args.frame is not None else find_good_frame(kp)
        out = args.output or f"{base}_frame{frame_idx}.png"
        plot_single_frame(kp, frame_idx, out)


if __name__ == "__main__":
    main()
