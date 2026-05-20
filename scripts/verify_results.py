"""Quick sanity checks on pipeline output .npy files."""
import argparse
import os
import numpy as np


def verify(output_dir):
    files = sorted(f for f in os.listdir(output_dir) if f.endswith(".npy"))
    if not files:
        print(f"No .npy files found in {output_dir}")
        return

    print(f"Found {len(files)} result files\n")

    for f in files:
        path = os.path.join(output_dir, f)
        kp = np.load(path)
        n_frames, n_joints, n_dims = kp.shape

        xyz = kp[:, :, :3]
        conf = kp[:, :, 3] if n_dims == 4 else None

        # Basic stats
        all_zero_frames = np.all(xyz == 0, axis=(1, 2)).sum()
        nan_frames = np.any(np.isnan(xyz), axis=(1, 2)).sum()
        mean_conf = conf.mean() if conf is not None else None
        low_conf_frac = (conf < 0.1).mean() if conf is not None else None

        # Range check — are 3D coords reasonable?
        xyz_nonzero = xyz[~np.all(xyz == 0, axis=(1, 2))]
        if len(xyz_nonzero) > 0:
            mins = xyz_nonzero.min(axis=(0, 1))
            maxs = xyz_nonzero.max(axis=(0, 1))
            spread = maxs - mins
        else:
            mins = maxs = spread = np.zeros(3)

        # Movement check — is the skeleton actually moving?
        frame_deltas = np.linalg.norm(np.diff(xyz, axis=0), axis=-1).mean()

        print(f"--- {f} ---")
        print(f"  Shape: {kp.shape}  ({n_frames} frames, {n_joints} joints, {n_dims} dims)")
        print(f"  All-zero frames: {all_zero_frames}/{n_frames}")
        print(f"  NaN frames: {nan_frames}/{n_frames}")
        if conf is not None:
            print(f"  Confidence: mean={mean_conf:.3f}, low(<0.1)={low_conf_frac:.1%}")
        print(f"  3D range:  X=[{mins[0]:.1f}, {maxs[0]:.1f}]  Y=[{mins[1]:.1f}, {maxs[1]:.1f}]  Z=[{mins[2]:.1f}, {maxs[2]:.1f}]")
        print(f"  3D spread: X={spread[0]:.1f}  Y={spread[1]:.1f}  Z={spread[2]:.1f}")
        print(f"  Mean per-frame movement: {frame_deltas:.4f}")
        print()

    # Cross-video consistency
    if len(files) > 1:
        shapes = [np.load(os.path.join(output_dir, f)).shape for f in files]
        joints = set(s[1] for s in shapes)
        frames = [s[0] for s in shapes]
        print(f"Cross-video: joint counts={joints}, frame range=[{min(frames)}, {max(frames)}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default="stored_vids/output")
    verify(parser.parse_args().output_dir)
