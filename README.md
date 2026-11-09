# Tennis Video Reframing · Build Track

The system follows the near-side player in each landscape clip and exports 1080×1920 video with the original audio. The first-frame box in `targets.json` identifies the player. MoveNet estimates poses in a search region around the player. The midpoint between the shoulder and hip centers guides the crop, with checks on position, scale, and pose geometry. A central dead zone lets the player move without shifting the crop. The crop responds faster near the edges and limits its speed within each shot. It holds the framing through brief detection failures. At a detected cut or after sustained tracking loss, the system searches the full frame and resets the crop once appearance and adjacent frames confirm the target. The pose model is MoveNet SinglePose Lightning v4, licensed under Apache 2.0. [Model card](https://www.kaggle.com/models/google/movenet/tensorFlow2/singlepose-lightning/4)

## Reproduce the outputs

Place the two supplied source clips in `clips/` with their original filenames. Use Python 3.10–3.12. The first model download requires internet access. The submitted videos are in `outputs/`.

```bash
python -m pip install -r requirements.txt
python track.py --output out/reproduced
```

## Results

We annotated player boxes independently on the source clips. After excluding the initialization frames, we evaluated 22 uniformly sampled frames and nine diagnostic frames as separate groups. Center error is the horizontal distance between the crop center and the annotated box center, divided by crop width. Frame-to-frame movement is measured across the full clip in source pixels. The improved crop fully retained all 31 sampled player boxes. [Evaluation data](evaluation/logs/metrics.json)

| Version and clip | Mean center error, regular frames | Mean center error, diagnostic frames | Target center outside crop, diagnostic frames | Maximum frame-to-frame pan | CPU processing and export time |
|---|---:|---:|---:|---:|---:|
| Baseline (outdoor) | 1.72% | 26.12% | 2/5 | 223 px | 44.7 s |
| Baseline (indoor) | 3.25% | 39.31% | 2/4 | 428 px | 119.0 s |
| Improved (outdoor) | 5.69% | 4.35% | 0/5 | 4 px | 49.0 s |
| Improved (indoor) | 9.77% | 10.53% | 0/4 | 21 px | 138.4 s |

At 13.11 seconds in the indoor clip (zero-based frame 393), the baseline has a center error of **2.55%** after a **423 px** jump from the preceding frame. The improved crop holds its position with a center error of **9.46%**. Both crops fully contain the player in this frame. Center error measures framing on a single frame, so the evaluation also reports player-box retention and frame-to-frame crop movement. [Frame 393 data](evaluation/case_frame_393.json)

Both versions ran on an Intel Core i7-13650HX with 16 TensorFlow CPU inference threads. Table times cover reading each clip through export. Model loading was timed separately. For the two clips totaling 47.6 seconds, the baseline took **185.4 s** (including **21.6 s** to load the model) and the improved version took **202.9 s** (including **15.4 s** to load the model). Processing and exporting one minute of footage took about **3 min 27 s** and **3 min 56 s**, respectively. The baseline script writes silent MP4V video. The improved version encodes H.264 and includes the original audio. [Baseline timings](evaluation/timing_baseline.json) · [Improved timings](evaluation/timing.json)

To recompute the evaluation from the included frame logs, run `python evaluate.py --baseline-dir evaluation/logs --output-dir evaluation/logs` after placing the source clips in `clips/`. The next evaluation step is to annotate more clips with cuts and measure target recovery rate and frames to recovery. AI assistance was used for implementation and reference-frame annotation.
