import numpy as np
from pose_pipeline import *


def annotate_single_person(filt, subject_id=0, confirm=False):

    keys = ((TrackingBbox & filt & "num_tracks=1") - PersonBboxValid).fetch("KEY")

    if confirm:
        print(f"Found {len(keys)} videos that can be auto-annotated with only one person present. Type Yes to confirm.")
        response = input()
        if response[0].upper() != "Y":
            print("Aborting")
            return

    for k in keys:
        tracks = (TrackingBbox & k).fetch1("tracks")
        track_id = np.unique([[t["track_id"] for t in t2] for t2 in tracks if len(t2) > 0])
        assert len(track_id) == 1, "Found two tracks, should not have"
        k.update({"video_subject_id": subject_id, "keep_tracks": track_id})
        PersonBboxValid.insert1(k)


def annotate_dominant_person(filt, subject_id=0):
    """Auto-annotate by selecting the track with the most frames.

    For videos with multiple detected tracks, this picks the dominant
    track (most frame appearances). Ties are broken by largest average
    bounding box area. Works for any number of tracks including 1.
    """

    keys = ((TrackingBbox & filt) - PersonBboxValid).fetch("KEY")

    for k in keys:
        tracks = (TrackingBbox & k).fetch1("tracks")

        # count frames and average bbox area per track_id
        track_stats = {}
        for frame in tracks:
            for t in frame:
                tid = t["track_id"]
                bbox = t["tlbr"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if tid not in track_stats:
                    track_stats[tid] = {"count": 0, "total_area": 0.0}
                track_stats[tid]["count"] += 1
                track_stats[tid]["total_area"] += area

        if not track_stats:
            continue

        # pick track with most frames, break ties by average area
        best_tid = max(
            track_stats,
            key=lambda tid: (
                track_stats[tid]["count"],
                track_stats[tid]["total_area"] / track_stats[tid]["count"],
            ),
        )

        k.update({"video_subject_id": subject_id, "keep_tracks": np.array([best_tid])})
        PersonBboxValid.insert1(k)
        print(f"  Auto-annotated {k.get('filename', k)}: selected track {best_tid} "
              f"({track_stats[best_tid]['count']} frames, "
              f"{len(track_stats)} total tracks)")
