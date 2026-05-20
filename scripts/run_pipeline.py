import argparse
import os
import sys
from datetime import datetime

import numpy as np

import pose_pipeline

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_dir = os.path.dirname(_repo_dir) + "/"
pose_pipeline.set_environmental_variables(pose_project_dir=_project_dir)
from pose_pipeline import *
from pose_pipeline.pipeline import (
    BottomUpBridging,
    BottomUpBridgingPerson,
    TrackingBboxMethodLookup,
    TopDownMethodLookup,
    LiftingMethodLookup,
)
from pose_pipeline.utils.video_format import insert_local_video
from pose_pipeline.utils.tracking import annotate_single_person, annotate_dominant_person

pose_pipeline.env.pytorch_memory_limit()
pose_pipeline.env.tensorflow_memory_limit()


def parse_args():
    parser = argparse.ArgumentParser(description="Run PosePipeline end-to-end")
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_video_dir = os.path.join(repo_dir, "stored_vids")
    parser.add_argument(
        "--video_dir",
        default=default_video_dir,
        help="Directory containing video files (default: stored_vids/)",
    )
    parser.add_argument(
        "--project", default="default_project", help="Project name for DataJoint"
    )
    parser.add_argument(
        "--tracking_method", default="MMDet_deepsort", help="Tracking method name"
    )
    parser.add_argument(
        "--top_down_method", default="Bridging_bml_movi_87", help="Top-down method name"
    )
    parser.add_argument(
        "--lifting_method", default="Bridging_bml_movi_87", help="Lifting method name"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save output .npy files (default: video_dir/output)",
    )
    parser.add_argument(
        "--skip_bottom_up",
        action="store_true",
        help="Skip bottom-up step if already done",
    )
    parser.add_argument(
        "--auto_annotate",
        action="store_true",
        help="Auto-select the dominant track (most frames) for multi-track videos",
    )
    return parser.parse_args()


def import_videos(video_dir, project):
    print(f"\n=== Step 1: Importing videos from {video_dir} ===")
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    files = [
        f
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[1].lower() in video_extensions
    ]

    if not files:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    for f in files:
        insert_local_video(
            f,
            datetime.now(),
            os.path.join(video_dir, f),
            video_project=project,
            skip_duplicates=True,
        )

    proj_filt = {"video_project": project}
    print(f"Imported {len(files)} videos. Video table:")
    print(Video & proj_filt)
    return proj_filt


def run_video_info(proj_filt):
    print("\n=== Step 2: Populating VideoInfo ===")
    VideoInfo.populate(proj_filt, suppress_errors=True)


def run_bottom_up(proj_filt):
    print("\n=== Step 3: Running bottom-up pose estimation ===")
    video_keys = (Video & proj_filt).fetch("KEY")

    BottomUpBridging.populate(proj_filt, suppress_errors=True)

    for v in video_keys:
        v_copy = v.copy()
        v_copy["bottom_up_method_name"] = "Bridging_OpenPose"
        BottomUpMethod.insert1(v_copy, skip_duplicates=True)

    BottomUpPeople.populate(proj_filt, suppress_errors=True)
    BlurredVideo.populate(proj_filt, suppress_errors=True)
    print("Bottom-up complete.")


def run_tracking(proj_filt, tracking_method_name):
    print(f"\n=== Step 4: Running tracking ({tracking_method_name}) ===")
    tracking_method = (
        TrackingBboxMethodLookup & f'tracking_method_name="{tracking_method_name}"'
    ).fetch1("tracking_method")

    tracking_keys = (Video & proj_filt).fetch("KEY")
    for key in tracking_keys:
        key["tracking_method"] = tracking_method
        TrackingBboxMethod.insert1(key, skip_duplicates=True)

    TrackingBbox.populate(tracking_keys, suppress_errors=True)
    n_tracked = len(TrackingBbox & proj_filt)
    print(f"Tracking complete. {n_tracked} TrackingBbox entries.")


def run_annotation(proj_filt, auto_annotate=False):
    print("\n=== Step 5: Annotation ===")
    video_keys = (Video & proj_filt).fetch("KEY")

    # Diagnostics
    n_tracking = len(TrackingBbox & proj_filt)
    n_valid = len(PersonBboxValid & proj_filt)
    print(f"  DB state: {len(video_keys)} videos, {n_tracking} TrackingBbox entries, {n_valid} PersonBboxValid entries")
    if n_tracking > 0:
        num_tracks = (TrackingBbox & proj_filt).fetch("num_tracks")
        print(f"  Track counts per video: {list(num_tracks)}")

    # First pass: auto-annotate single-track videos
    for key in video_keys:
        annotate_single_person(key)

    # Second pass: auto-annotate multi-track videos if requested
    if auto_annotate:
        print("Auto-annotating multi-track videos (selecting dominant track)...")
        annotate_dominant_person(proj_filt)

    n_valid_after = len(PersonBboxValid & proj_filt)
    print(f"  PersonBboxValid entries after annotation: {n_valid_after}")

    PersonBbox.populate(proj_filt, suppress_errors=True)

    n_annotated = len(PersonBbox & proj_filt)
    n_videos = len(Video & proj_filt)
    print(f"PersonBbox populated for {n_annotated}/{n_videos} videos.")

    if n_annotated < n_videos:
        print("Some videos need manual annotation via the annotations GUI.")
        print(
            "Annotate them, then re-run this script (bottom-up/tracking will be skipped)."
        )

    DetectedFrames.populate(proj_filt, suppress_errors=True)
    return n_annotated > 0


def run_top_down(proj_filt, top_down_method_name):
    print(
        f"\n=== Step 6: Running top-down pose estimation ({top_down_method_name}) ==="
    )

    # BottomUpBridgingPerson matches bridging detections to tracked person bboxes.
    # Required by Bridging_* top-down methods.
    BottomUpBridgingPerson.populate(proj_filt, suppress_errors=True)
    n_bridging_person = len(BottomUpBridgingPerson & proj_filt)
    print(f"  BottomUpBridgingPerson: {n_bridging_person} entries")

    top_down_method = (
        TopDownMethodLookup & f'top_down_method_name="{top_down_method_name}"'
    ).fetch1("top_down_method")

    top_down_keys = (PersonBbox & proj_filt).fetch("KEY")
    for td in top_down_keys:
        td["top_down_method"] = top_down_method
        TopDownMethod.insert1(td, skip_duplicates=True)

    TopDownPerson.populate(proj_filt, suppress_errors=True)
    n_top_down = len(TopDownPerson & proj_filt)
    print(f"Top-down complete. {n_top_down} TopDownPerson entries.")


def run_lifting(proj_filt, lifting_method_name):
    print(f"\n=== Step 7: Running lifting ({lifting_method_name}) ===")
    lifting_method = (
        LiftingMethodLookup & f'lifting_method_name="{lifting_method_name}"'
    ).fetch1("lifting_method")

    lifting_keys = (TopDownPerson & proj_filt).fetch("KEY")
    for L in lifting_keys:
        L["lifting_method"] = lifting_method
        LiftingMethod.insert1(L, skip_duplicates=True)

    LiftingPerson.populate(proj_filt, suppress_errors=True)
    print("Lifting complete.")


def save_results(proj_filt, output_dir):
    print(f"\n=== Step 8: Saving results to {output_dir} ===")
    os.makedirs(output_dir, exist_ok=True)

    results = (LiftingPerson & proj_filt).fetch(as_dict=True)

    if not results:
        print("No lifting results found. Check if all upstream steps completed.")
        return

    for r in results:
        video_info = (Video & r).fetch1()
        filename = os.path.splitext(video_info["filename"])[0]
        out_path = os.path.join(output_dir, f"{filename}_keypoints3d.npy")
        kp = r.get("keypoints_3d", r.get("keypoints"))
        if kp is None:
            print(f"Skipping {filename}: no keypoint data found (keys: {list(r.keys())})")
            continue
        np.save(out_path, kp)
        print(f"Saved: {out_path} (shape: {kp.shape})")

    print(f"\nDone! Saved {len(results)} result(s) to {output_dir}")


def main():
    args = parse_args()

    proj_filt = import_videos(args.video_dir, args.project)
    run_video_info(proj_filt)

    if not args.skip_bottom_up:
        run_bottom_up(proj_filt)

    run_tracking(proj_filt, args.tracking_method)

    has_annotations = run_annotation(proj_filt, auto_annotate=args.auto_annotate)
    if not has_annotations:
        print("\nStopping: no annotated videos to process further.")
        print("Use the annotations GUI, then re-run with --skip_bottom_up.")
        return

    run_top_down(proj_filt, args.top_down_method)
    run_lifting(proj_filt, args.lifting_method)

    output_dir = args.output_dir or os.path.join(args.video_dir, "output")
    save_results(proj_filt, output_dir)


if __name__ == "__main__":
    main()
