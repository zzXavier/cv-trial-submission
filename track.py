"""CPU single-player reframing. Run `python track.py` for all clips.

Use --replay --mode smooth/direct to compare cameras on the saved target trajectory.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import imageio_ffmpeg
import numpy as np

from crop_control import CropController
from tracking import map_pose
from shot_tracking import ShotTracker, validate_anchors

MODEL_URL = 'https://tfhub.dev/google/movenet/singlepose/lightning/4'


def load_model():
    # TF 2.18 Windows file I/O fails for caches under this Chinese workspace path.
    os.environ.setdefault('TFHUB_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'cv-trial-movenet-tfhub'))
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', str(min(16, os.cpu_count() or 4)))
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    import tensorflow as tf
    import tensorflow_hub as hub
    model = hub.load(MODEL_URL).signatures['serving_default']

    def infer(frame, roi):
        left, top, side = roi
        scale = 192 / side
        transform = np.array([[scale, 0, -left * scale],
                              [0, scale, -top * scale]], dtype=np.float32)
        patch = cv2.warpAffine(frame, transform, (192, 192), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        points = model(tf.convert_to_tensor(rgb[None], dtype=tf.int32))['output_0'].numpy()[0, 0]
        return map_pose(points, roi)

    return infer


def write_json(path, data):
    with path.open('w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write('\n')


def validate_replay(saved, metadata):
    if any(saved.get(key) != value for key, value in metadata.items()):
        raise ValueError('Saved trajectory format, source, target or anchors differ; rerun without --replay')


def build_crops(trajectory, frame_size, fps, initial_box, mode):
    width, height = frame_size
    camera = CropController(width, height, fps, initial_box[0] + initial_box[2] / 2, mode)
    crop_h = min(height, round(camera.crop_w * 16 / 9))
    crop_y = int(np.clip(initial_box[1] + initial_box[3] / 2 - crop_h / 2, 0, height - crop_h))
    rows = []
    for index, state in enumerate(trajectory):
        if state['frame_index'] != index:
            raise ValueError('Saved trajectory has missing or reordered frames')
        if state.get('camera_reset'):
            reset_box = state.get('reset_box') or state.get('target_box')
            center = (state['reset_box'][0] + state['reset_box'][2] / 2
                      if state.get('reset_box') else state['target_x'])
            if center is None:
                raise ValueError('Camera reset requires a confirmed target or explicit anchor')
            camera = CropController(width, height, fps, center, mode)
            if reset_box:
                crop_y = int(np.clip(reset_box[1] + reset_box[3] / 2 - crop_h / 2, 0, height - crop_h))
        left = camera.update(state['target_x'], state['target_box'])
        rows.append({**state, 'crop_x': left, 'crop_y': crop_y,
                     'crop_w': camera.crop_w, 'crop_h': crop_h})
    return rows


def process_clip(source, target, args, infer):
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video {source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        cap.release()
        raise ValueError(f'Invalid dimensions or frame rate in {source}')
    if target.get('frame_index') != 0:
        cap.release()
        raise ValueError('Only frame-0 target annotations are supported')
    box = [target[key] for key in ('x', 'y', 'w', 'h')]
    anchor_rows = args.anchor_map.get(source.stem, [])
    anchors = validate_anchors(anchor_rows, (width, height), expected_frames)
    digest = hashlib.sha256()
    with source.open('rb') as original:
        for chunk in iter(lambda: original.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata = {'schema_version': 2, 'source': source.name, 'source_size': [width, height],
                'fps': fps, 'source_sha256': digest.hexdigest(), 'initial_box': box,
                'anchors': anchor_rows}
    trajectory_path = args.output / f'{source.stem}_tracking.json'
    inference_s = 0.0
    processing_started = time.perf_counter()
    if args.replay:
        saved = json.loads(trajectory_path.read_text(encoding='utf-8'))
        validate_replay(saved, metadata)
        trajectory = saved['frames']
    else:
        def counted_infer(frame, roi):
            nonlocal inference_s
            inference_started = time.perf_counter()
            pose = infer(frame, roi)
            inference_s += time.perf_counter() - inference_started
            return pose

        tracker = ShotTracker(box, (width, height), fps, counted_infer, anchors=anchors)
        trajectory = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                trajectory.append(tracker.update(frame, len(trajectory)))
                if len(trajectory) % 300 == 0:
                    print(f'{source.name}: tracking {len(trajectory)}/{expected_frames}', flush=True)
        finally:
            cap.release()
        # Confirmation may revise earlier observations. Render only once the
        # trajectory is complete so the first confirmed cut frame is correct.
        cap = cv2.VideoCapture(str(source))
    if not trajectory or (expected_frames > 0 and len(trajectory) != expected_frames):
        cap.release()
        raise RuntimeError(f'Trajectory has {len(trajectory)} of {expected_frames} frames')
    logs = build_crops(trajectory, (width, height), fps, box, args.mode)

    output_path = args.output / f'{source.stem}_{args.mode}.mp4'
    with tempfile.TemporaryDirectory(prefix='.render-', dir=args.output) as temporary:
        silent = Path(temporary) / 'silent.mp4'
        final = Path(temporary) / 'final.mp4'
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1080, 1920))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f'Cannot create video writer in {args.output}')
        try:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index >= len(logs):
                    raise ValueError('Trajectory is shorter than the source')
                row = logs[index]
                x, y, w, h = (row[key] for key in ('crop_x', 'crop_y', 'crop_w', 'crop_h'))
                crop = frame[y:y + h, x:x + w]
                writer.write(cv2.resize(crop, (1080, 1920), interpolation=cv2.INTER_LINEAR))
                index += 1
                if index % 300 == 0:
                    print(f'{source.name}: {index}/{expected_frames} frames', flush=True)
        finally:
            cap.release()
            writer.release()
        processing_s = time.perf_counter() - processing_started
        if index != len(logs):
            raise RuntimeError(f'Decode stopped at {index} of {len(logs)} frames')

        encoding_started = time.perf_counter()
        # H.264 is broadly playable; the original audio is copied without re-encoding.
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                   '-y', '-i', str(silent), '-i', str(source), '-map', '0:v:0',
                   '-map', '1:a?', '-c:v', 'libx264', '-threads', '4', '-preset', 'veryfast',
                   '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
                   '-movflags', '+faststart', str(final)]
        subprocess.run(command, check=True, capture_output=True)
        encoding_s = time.perf_counter() - encoding_started
        check = cv2.VideoCapture(str(final))
        readable = check.isOpened() and check.read()[0]
        encoded_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        check.release()
        if not readable or encoded_frames != len(logs):
            raise RuntimeError('Final video validation failed')
        final.replace(output_path)

    write_json(output_path.with_suffix('.json'), logs)
    if not args.replay:
        write_json(trajectory_path, {**metadata, 'frames': trajectory})
    result = {'source': source.name, 'output': str(output_path), 'mode': args.mode,
              'frames': len(logs), 'video_s': len(logs) / fps,
              'processing_s': processing_s, 'inference_s': inference_s,
              'encoding_s': encoding_s, 'clip_total_s': time.perf_counter() - started,
              'accepted_frames': sum(r['status'] == 'tracking' for r in logs),
              'unresolved_frames': sum(r['target_x'] is None for r in logs),
              'camera_resets': sum(bool(r.get('camera_reset')) for r in logs[1:]),
              'replay': args.replay}
    print(json.dumps(result), flush=True)
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='*', type=Path)
    parser.add_argument('--targets', type=Path, default=Path('targets.json'))
    parser.add_argument('--output', type=Path, default=Path('out/improved'))
    parser.add_argument('--mode', choices=('direct', 'smooth', 'deadzone'), default='deadzone')
    parser.add_argument('--replay', action='store_true', help='reuse saved target observations')
    parser.add_argument('--anchors', type=Path, help='optional per-clip lists of later frame target boxes')
    args = parser.parse_args()
    sources = args.sources or sorted(Path('clips').glob('*.mp4'))
    if not sources:
        parser.error('No videos supplied or found under clips/')
    targets = json.loads(args.targets.read_text(encoding='utf-8-sig'))
    args.anchor_map = (json.loads(args.anchors.read_text(encoding='utf-8-sig')) if args.anchors else {})
    if not isinstance(args.anchor_map, dict):
        parser.error('--anchors must contain an object keyed by clip stem')
    for source in sources:
        if source.stem not in targets:
            parser.error(f'Missing first-frame target for {source.name}')
    args.output.mkdir(parents=True, exist_ok=True)
    model_started = time.perf_counter()
    infer = None if args.replay else load_model()
    model_s = time.perf_counter() - model_started
    results = [process_clip(source, targets[source.stem], args, infer) for source in sources]
    write_json(args.output / f'run_{args.mode}.json', {
        'model_load_s': model_s, 'command_total_s': time.perf_counter() - started,
        'intra_op_threads': os.environ.get('TF_NUM_INTRAOP_THREADS'), 'clips': results})


if __name__ == '__main__':
    main()
