import cv2
import numpy as np
from collections import deque
from datetime import datetime
import time

from mp_face import create_face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
ZONES = ['CENTER', 'LEFT', 'RIGHT', 'DOWN', 'UP']

CONFIG = {
    'calibration_seconds': 15,
    'min_calibration_samples': 30,

    'head_yaw_threshold': 28.0,
    'head_pitch_down_threshold': 18.0,
    'head_pitch_up_threshold': 15.0,
    'iris_h_threshold': 0.16,
    'iris_v_threshold': 0.16,
    'iris_gain': 3.0,
    'forward_band_yaw': 12.0,

    'ema_pose': 0.12,
    'ema_iris': 0.25,
    'zone_hold_frames': 6,

    'dwell_alert_secs': 2.0,
    'no_face_alert_secs': 3.0,
    'second_face_alert_secs': 1.0,

    'saccade_margin': 0.12,
    'saccade_window_secs': 4.0,
    'saccade_min_count': 3,

    'blink_ear_threshold': 0.2,
    'blink_window_secs': 90.0,
    'blink_rate_multiplier': 1.8,

    'max_events': 200,
}

EVENT_WEIGHTS = {
    'side_look': 9,
    'look_down': 9,
    'look_up': 6,
    'reading': 7,
    'no_face': 8,
    'second_face': 40,
    'stress': 1,
}

EVENT_NAMES = {
    'side_look': 'Looking to the side (possible monitor/notes)',
    'look_down': 'Looking down toward desk/phone',
    'look_up': 'Looking up/away from screen',
    'reading': 'Rapid eye scanning while head forward (possible script/autocue)',
    'no_face': 'Face lost / person out of frame',
    'second_face': 'Second person visible in frame',
    'stress': 'Elevated blink rate (soft)',
}

face_mesh = create_face_mesh(num_faces=2, refine=True)


def pt2f(lm, image_w, image_h):
    return np.array([lm.x * image_w, lm.y * image_h], dtype=np.float64)


def estimate_head_pose(landmarks, image_w, image_h):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
        (0.0, -330.0, -65.0),
    ], dtype=np.float64)
    index_points = [1, 33, 263, 61, 291, 152]
    image_points = np.array(
        [pt2f(landmarks[i], image_w, image_h) for i in index_points], dtype=np.float64)

    camera_matrix = np.array([
        [image_w, 0, image_w / 2],
        [0, image_w, image_h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    projection = np.hstack((rotation_matrix, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection)
    pitch, yaw, roll = euler.flatten()[:3]
    return pitch, yaw, roll


def iris_offsets(landmarks, image_w, image_h):
    def single(iris_idx, eye_idx):
        iris = landmarks[iris_idx]
        ix, iy = iris.x * image_w, iris.y * image_h
        xs, ys = [], []
        for i in eye_idx:
            p = pt2f(landmarks[i], image_w, image_h)
            xs.append(p[0])
            ys.append(p[1])
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        width = float(np.max(xs) - np.min(xs))
        height = float(np.max(ys) - np.min(ys))
        if width <= 1e-6 or height <= 1e-6:
            return 0.0, 0.0
        return (ix - cx) / width, (iy - cy) / height

    lh, lv = single(468, LEFT_EYE)
    rh, rv = single(473, RIGHT_EYE)
    return (lh + rh) / 2.0, (lv + rv) / 2.0


def eye_aspect_ratio(landmarks, eye_points, image_w, image_h):
    p = [pt2f(landmarks[i], image_w, image_h) for i in eye_points]
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    if c <= 1e-6:
        return 1.0
    return (a + b) / (2.0 * c)


def classify_gaze(yaw, pitch, h_dev, v_dev):
    eff_h = yaw + h_dev * CONFIG['iris_gain']
    eff_v = pitch + v_dev * CONFIG['iris_gain'] * 0.7
    if eff_h > CONFIG['head_yaw_threshold']:
        return 'RIGHT'
    if eff_h < -CONFIG['head_yaw_threshold']:
        return 'LEFT'
    if eff_v > CONFIG['head_pitch_down_threshold']:
        return 'DOWN'
    if eff_v < -CONFIG['head_pitch_up_threshold']:
        return 'UP'
    return 'CENTER'


def adaptive_ema(old, new, alpha):
    return alpha * new + (1.0 - alpha) * old if old is not None else new


class IntegritySession:
    def __init__(self, baseline):
        self.baseline = baseline
        self.start_time = time.monotonic()
        self.wall_start = datetime.now()

        self.yaw = baseline['yaw']
        self.pitch = baseline['pitch']
        self.roll = baseline['roll']
        self.h_dev = 0.0
        self.v_dev = 0.0
        self.prev_h = baseline['h']

        self.zone = 'CENTER'
        self.zone_since = time.monotonic()
        self.zone_votes = 0
        self.zone_times = {z: 0.0 for z in ZONES}

        self.had_face = False
        self.no_face_since = None
        self.second_face_active = False
        self.second_face_since = None
        self.second_face_logged = False

        self.blink_closed = False
        self.blink_count = 0
        self.blink_times = deque()
        self.stressed = False

        self.reading = False
        self.reading_since = None
        self.saccades = deque()

        self.events = []

    def _log_event(self, etype, duration, timestamp=None):
        if len(self.events) >= CONFIG['max_events']:
            return
        ts = timestamp if timestamp else datetime.now().strftime('%H:%M:%S')
        severity = min(10, int(100 * EVENT_WEIGHTS[etype] /
                               max(EVENT_WEIGHTS.values())))
        self.events.append({
            'time': timestamp if timestamp is not None else time.monotonic(),
            'ts': ts,
            'type': etype,
            'duration': duration,
            'name': EVENT_NAMES[etype],
            'severity': severity,
        })

    def update(self, yaw, pitch, roll, h, v, dt, blink_closed,
               faces_present, num_faces):
        self.yaw = adaptive_ema(self.yaw, yaw, CONFIG['ema_pose'])
        self.pitch = adaptive_ema(self.pitch, pitch, CONFIG['ema_pose'])
        self.roll = adaptive_ema(self.roll, roll, CONFIG['ema_pose'])
        self.h_dev = adaptive_ema(self.h_dev, h - self.baseline['h'],
                                  CONFIG['ema_iris'])
        self.v_dev = adaptive_ema(self.v_dev, v - self.baseline['v'],
                                  CONFIG['ema_iris'])

        self.zone_times[self.zone] += dt

        self._update_blinks(blink_closed)
        self._update_second_face(num_faces)
        self._update_no_face(faces_present)

        pending = classify_gaze(self.yaw, self.pitch, self.h_dev, self.v_dev)
        self._update_zone(pending)

        if faces_present and not blink_closed:
            self._update_reading()

    def _update_zone(self, pending):
        if pending == self.zone:
            self.zone_votes = 0
        else:
            self.zone_votes += 1
            if self.zone_votes >= CONFIG['zone_hold_frames']:
                self._change_zone(pending)

    def _change_zone(self, new_zone):
        now = time.monotonic()
        dwell = now - self.zone_since
        if self.zone != 'CENTER' and dwell >= CONFIG['dwell_alert_secs']:
            if self.zone in ('LEFT', 'RIGHT'):
                self._log_event('side_look', dwell)
            elif self.zone == 'DOWN':
                self._log_event('look_down', dwell)
            elif self.zone == 'UP':
                self._log_event('look_up', dwell)
        self.zone = new_zone
        self.zone_since = now
        self.zone_votes = 0

    def _update_blinks(self, blink_closed):
        if blink_closed and not self.blink_closed:
            self.blink_closed = True
        elif not blink_closed and self.blink_closed:
            self.blink_closed = False
            self.blink_count += 1
            self.blink_times.append(time.monotonic())
        while self.blink_times and \
              self.blink_times[0] < time.monotonic() - CONFIG['blink_window_secs']:
            self.blink_times.popleft()

        rate = self._blink_rate()
        base = self.baseline['blink_rate']
        stressed = rate > base * CONFIG['blink_rate_multiplier'] and base > 0
        if stressed and not self.stressed:
            self._log_event('stress', 1.0)
        self.stressed = stressed

    def _blink_rate(self):
        window = min(CONFIG['blink_window_secs'],
                     max(time.monotonic() - self.start_time, 1e-6))
        return (len(self.blink_times) / window) * 60.0

    def _update_second_face(self, num_faces):
        now = time.monotonic()
        if num_faces > 1:
            if not self.second_face_active:
                self.second_face_active = True
                self.second_face_since = now
                self.second_face_logged = False
            elif not self.second_face_logged and \
                    now - self.second_face_since >= CONFIG['second_face_alert_secs']:
                self._log_event('second_face', 1.0)
                self.second_face_logged = True
        else:
            self.second_face_active = False

    def _update_no_face(self, faces_present):
        if faces_present:
            if self.no_face_since is not None:
                gap = time.monotonic() - self.no_face_since
                if self.had_face and gap >= CONFIG['no_face_alert_secs']:
                    self._log_event('no_face', gap)
                self.no_face_since = None
            self.had_face = True
        else:
            if self.no_face_since is None:
                self.no_face_since = time.monotonic()

    def _update_reading(self):
        now = time.monotonic()
        hvel = self.h_dev - self.prev_h
        self.prev_h = self.h_dev
        margin = CONFIG['saccade_margin']
        if abs(hvel) > margin:
            sign = 1 if hvel > 0 else -1
            if not self.saccades or self.saccades[-1][1] != sign:
                self.saccades.append((now, sign))
        while self.saccades and \
              self.saccades[0][0] < now - CONFIG['saccade_window_secs']:
            self.saccades.popleft()

        count = len(self.saccades)
        if count >= CONFIG['saccade_min_count'] and not self.reading:
            self.reading = True
            self.reading_since = now
            self._log_event('reading', 0.0)
        elif self.reading and count < CONFIG['saccade_min_count'] - 1:
            self._end_reading(now)

    def _end_reading(self, now):
        if self.reading and self.reading_since is not None:
            if self.events and self.events[-1]['type'] == 'reading':
                self.events[-1]['duration'] = now - self.reading_since
        self.reading = False
        self.reading_since = None

    def finish(self):
        now = time.monotonic()
        dwell = now - self.zone_since
        if self.zone != 'CENTER' and dwell >= CONFIG['dwell_alert_secs']:
            if self.zone in ('LEFT', 'RIGHT'):
                self._log_event('side_look', dwell)
            elif self.zone == 'DOWN':
                self._log_event('look_down', dwell)
            elif self.zone == 'UP':
                self._log_event('look_up', dwell)
        if self.no_face_since is not None and self.had_face:
            gap = now - self.no_face_since
            if gap >= CONFIG['no_face_alert_secs']:
                self._log_event('no_face', gap)
        if self.reading:
            self._end_reading(now)

        duration = now - self.start_time
        penalty = sum(EVENT_WEIGHTS[e['type']] * max(e['duration'], 0.5)
                      for e in self.events)
        score = max(0.0, min(100.0, 100.0 - penalty))
        return self._build_report(score, duration)

    def _build_report(self, score, duration):
        lines = ['=' * 56, '  INTERVIEW INTEGRITY REPORT',
                 '=' * 56,
                 f"  Date/Time      : {self.wall_start.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"  Duration       : {self._fmt_dur(duration)}",
                 '',
                 f"  Integrity score: {score:.1f} / 100",
                 f"  Verdict        : {verdict(score)}",
                 '',
                 '  Time per gaze zone:']
        for z in ZONES:
            pct = (self.zone_times[z] / max(duration, 1e-6)) * 100
            lines.append(f"    {z:<8} {self._fmt_dur(self.zone_times[z])}"
                         f"  ({pct:.1f}%)")
        lines += ['',
                  f"  Blinks          : {self.blink_count}",
                  f"  Avg blink rate  : "
                  f"{(self.blink_count / max(duration / 60.0, 1e-6)):.1f} / min"
                  f"  (baseline {self.baseline['blink_rate']:.1f} / min)",
                  '',
                  f"  Flagged events  : {len(self.events)}"]
        if not self.events:
            lines += ['    No suspicious behavior recorded.']
        for i, e in enumerate(self.events, 1):
            lines.append(
                f"    [{i:>3}] {e['ts']}  {self._fmt_dur(max(e['duration'], 0.0))}"
                f"  \u2022 {e['name']}  (sev {e['severity']}/10)")
        lines += ['',
                  '  Note: single-webcam face-movement analysis is approximate;',
                  '  use as a screening tool, not as proof of misconduct.',
                  '=' * 56]
        return '\n'.join(lines)

    @staticmethod
    def _fmt_dur(secs):
        secs = max(int(secs), 0)
        return f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"


def verdict(score):
    if score >= 80:
        return 'LOW RISK'
    if score >= 60:
        return 'MODERATE RISK'
    return 'HIGH RISK'


def run_calibration(cap):
    cal_start = time.monotonic()
    samples = []
    blinks = 0
    closed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_mesh.process(frame_rgb)
        image_h, image_w = frame.shape[:2]

        if faces:
            lm = faces[0].landmark
            pose = estimate_head_pose(lm, image_w, image_h)
            h, v = iris_offsets(lm, image_w, image_h)
            left_ear = eye_aspect_ratio(lm, LEFT_EYE, image_w, image_h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, image_w, image_h)
            if pose is not None:
                samples.append((pose[0], pose[1], pose[2], h, v))
            ear = (left_ear + right_ear) / 2.0
            if ear < CONFIG['blink_ear_threshold']:
                if not closed:
                    closed = True
            else:
                if closed:
                    blinks += 1
                    closed = False

        elapsed = time.monotonic() - cal_start
        progress = min(1.0, elapsed / CONFIG['calibration_seconds'])
        bar_w = 300
        cv2.rectangle(frame, (10, image_h - 40),
                      (10 + bar_w, image_h - 10), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, image_h - 40),
                      (10 + int(bar_w * progress), image_h - 10),
                      (0, 200, 0), -1)
        cv2.putText(frame,
                    'CALIBRATING - sit still and look straight at the camera',
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Samples: {len(samples)}   ESC to skip',
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow('Interview Integrity (Calibration)', frame)

        if progress >= 1.0 and len(samples) >= CONFIG['min_calibration_samples']:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if samples:
        arr = np.array(samples)
        base = {
            'yaw': float(np.mean(arr[:, 1])),
            'pitch': float(np.mean(arr[:, 0])),
            'roll': float(np.mean(arr[:, 2])),
            'h': float(np.mean(arr[:, 3])),
            'v': float(np.mean(arr[:, 4])),
            'blink_rate': (blinks / max(elapsed / 60.0, 1e-6)),
        }
    else:
        base = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                'h': 0.0, 'v': 0.0, 'blink_rate': 0.0}
    return base


def draw_overlay(frame, session, fps, num_faces):
    image_h, image_w = frame.shape[:2]

    if session.zone != 'CENTER':
        in_dwell = time.monotonic() - session.zone_since
        status = 'FLAGGED' if in_dwell >= CONFIG['dwell_alert_secs'] else 'SUSPICIOUS'
    else:
        status = 'OK'
    color = (0, 200, 0) if status == 'OK' else (
        (0, 200, 255) if status == 'SUSPICIOUS' else (0, 60, 255))

    rows = [
        f"INTEGRITY: {status}",
        f"Zone: {session.zone}",
        f"Yaw {session.yaw:+.1f} | Pitch {session.pitch:+.1f}",
        f"H {session.h_dev:+.2f} | V {session.v_dev:+.2f}",
        f"Blinks: {session.blink_count}",
        f"Faces: {num_faces}",
    ]
    for i, text in enumerate(rows):
        cv2.putText(frame, text, (15, 40 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.putText(frame, f"FPS {fps:.0f}",
                (image_w - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (200, 200, 0), 2)
    elapsed = time.monotonic() - session.start_time
    cv2.putText(frame, session._fmt_dur(elapsed),
                (image_w - 110, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (200, 200, 200), 2)

    if session.events:
        last = session.events[-1]
        cv2.putText(frame, f"LAST FLAG: {last['name']}",
                    (15, image_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 60, 255), 2)


def run_session(cap, baseline):
    session = IntegritySession(baseline)
    prev = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break

        now = time.monotonic()
        dt = max(now - prev, 1e-6)
        prev = now

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_h, image_w = frame.shape[:2]
        faces = face_mesh.process(frame_rgb)

        if faces:
            face_mesh.draw(frame, faces)
            lm = faces[0].landmark
            pose = estimate_head_pose(lm, image_w, image_h)
            h, v = iris_offsets(lm, image_w, image_h)
            left_ear = eye_aspect_ratio(lm, LEFT_EYE, image_w, image_h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, image_w, image_h)
            blink_closed = (left_ear + right_ear) / 2.0 < \
                CONFIG['blink_ear_threshold']

            if pose is None:
                pose = (baseline['pitch'], baseline['yaw'], baseline['roll'])
            session.update(pose[0], pose[1], pose[2], h, v, dt,
                           blink_closed, True, len(faces))
        else:
            session.update(baseline['pitch'], baseline['yaw'], baseline['roll'],
                           baseline['h'], baseline['v'], dt,
                           False, False, 0)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 1.0 / max(dt, 1e-6)
        draw_overlay(frame, session, fps, len(faces) if faces else 0)

        cv2.imshow('Interview Integrity', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    report = session.finish()
    return report


def write_report(report):
    path = 'interview_report.txt'
    with open(path, 'w') as f:
        f.write(report + '\n')
    return path


def selftest():
    print('Running selftest on pure functions...')
    image_w, image_h = 640, 480

    fake = []
    for i in range(478):
        fake.append(type('L', (), {'x': 0.5, 'y': 0.5, 'z': 0.0})())
    fake[1] = type('L', (), {'x': 0.5, 'y': 0.4, 'z': 0.0})()
    fake[152] = type('L', (), {'x': 0.5, 'y': 0.85, 'z': 0.0})()
    fake[33] = type('L', (), {'x': 0.35, 'y': 0.42, 'z': 0.0})()
    fake[263] = type('L', (), {'x': 0.65, 'y': 0.42, 'z': 0.0})()
    fake[61] = type('L', (), {'x': 0.43, 'y': 0.62, 'z': 0.0})()
    fake[291] = type('L', (), {'x': 0.57, 'y': 0.62, 'z': 0.0})()
    for i in LEFT_EYE + RIGHT_EYE + [468, 473]:
        fake[i] = type('L', (), {'x': 0.5, 'y': 0.42, 'z': 0.0})()

    pose = estimate_head_pose(fake, image_w, image_h)
    assert pose is not None, 'head pose failed'
    h, v = iris_offsets(fake, image_w, image_h)
    assert abs(h) < 0.5 and abs(v) < 0.5, 'iris offsets out of range'
    ear = eye_aspect_ratio(fake, LEFT_EYE, image_w, image_h)
    assert 0 < ear < 1.5, f'bad EAR {ear}'

    assert classify_gaze(0, 0, 0, 0) == 'CENTER'
    assert classify_gaze(40, 0, 0, 0) == 'RIGHT'
    assert classify_gaze(-40, 0, 0, 0) == 'LEFT'
    assert classify_gaze(0, 40, 0, 0) == 'DOWN'
    assert classify_gaze(0, -40, 0, 0) == 'UP'
    assert verdict(90) == 'LOW RISK'
    assert verdict(65) == 'MODERATE RISK'
    print('selftest: OK')


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open webcam. Check camera permissions or index.')
        return 1

    print('Starting calibration. Sit still and look straight at the camera.')
    baseline = run_calibration(cap)
    print(f'Calibration complete. Baseline yaw {baseline["yaw"]:.1f}, '
          f'pitch {baseline["pitch"]:.1f}, iris h {baseline["h"]:.3f}, '
          f'blink rate {baseline["blink_rate"]:.1f}/min')
    cv2.destroyAllWindows()

    print('Interview tracking started. Press q or ESC to finish.')
    report = run_session(cap, baseline)
    cap.release()
    cv2.destroyAllWindows()

    print('\n' + report + '\n')
    path = write_report(report)
    print(f'Report written to {path}')
    return 0


if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        selftest()
    else:
        sys.exit(main())