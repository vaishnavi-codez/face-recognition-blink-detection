from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import cv2
import numpy as np


Point = tuple[float, float]


class ChallengeAction(str, Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SMILE = "smile"
    NOD = "nod"


@dataclass(frozen=True)
class AntiSpoofThresholds:
    min_texture_laplacian_var: float = 45.0
    min_lbp_entropy: float = 3.2
    max_specular_uniformity: float = 0.58
    min_specular_component_count: int = 3
    max_flat_depth_score: float = 0.82
    min_face_background_flow_ratio: float = 1.65
    max_screen_grid_energy_ratio: float = 0.22
    min_model_live_score: float = 0.72
    min_passed_checks: int = 5


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    score: float
    reason: str


@dataclass(frozen=True)
class AntiSpoofDecision:
    live: bool
    confidence: float
    checks: tuple[CheckResult, ...]


@dataclass
class ChallengeState:
    action: ChallengeAction
    baseline_landmarks: dict[str, Point]
    frames_seen: int = 0
    satisfied: bool = False


@dataclass
class ActiveLivenessChallenge:
    actions: Sequence[ChallengeAction] = (
        ChallengeAction.TURN_LEFT,
        ChallengeAction.TURN_RIGHT,
        ChallengeAction.SMILE,
        ChallengeAction.NOD,
    )
    rng: random.Random = field(default_factory=random.Random)
    state: ChallengeState | None = None

    def start(self, landmarks: dict[str, Point]) -> ChallengeState:
        self.state = ChallengeState(self.rng.choice(tuple(self.actions)), landmarks)
        return self.state

    def update(self, landmarks: dict[str, Point]) -> bool:
        if self.state is None:
            self.start(landmarks)
            return False

        self.state.frames_seen += 1
        base = self.state.baseline_landmarks
        action = self.state.action

        if action == ChallengeAction.TURN_LEFT:
            self.state.satisfied = yaw_delta(base, landmarks) < -0.08
        elif action == ChallengeAction.TURN_RIGHT:
            self.state.satisfied = yaw_delta(base, landmarks) > 0.08
        elif action == ChallengeAction.SMILE:
            self.state.satisfied = smile_delta(base, landmarks) > 0.10
        elif action == ChallengeAction.NOD:
            self.state.satisfied = nod_delta(base, landmarks) > 0.08

        return self.state.satisfied


def analyze_frame(
    frame_bgr: np.ndarray,
    face_box: tuple[int, int, int, int],
    landmarks: dict[str, Point] | None = None,
    previous_frame_bgr: np.ndarray | None = None,
    model_live_score_fn: Callable[[np.ndarray], float] | None = None,
    thresholds: AntiSpoofThresholds = AntiSpoofThresholds(),
) -> AntiSpoofDecision:
    """Run passive anti-spoofing checks for one frame.

    face_box is (x, y, width, height). landmarks should contain normalized or
    pixel points for: left_eye, right_eye, nose_tip, mouth_left, mouth_right,
    chin, and optionally forehead/left_cheek/right_cheek.
    """

    face = crop_box(frame_bgr, face_box)
    checks = [
        texture_check(face, thresholds),
        reflection_check(face, thresholds),
        screen_artifact_check(face, thresholds),
    ]

    if landmarks:
        checks.append(depth_cue_check(landmarks, thresholds))

    if previous_frame_bgr is not None:
        checks.append(background_motion_check(previous_frame_bgr, frame_bgr, face_box, thresholds))

    if model_live_score_fn is not None:
        checks.append(model_check(face, model_live_score_fn, thresholds))

    passed_count = sum(check.passed for check in checks)
    confidence = passed_count / max(len(checks), 1)
    live = passed_count >= min(thresholds.min_passed_checks, len(checks))
    return AntiSpoofDecision(live=live, confidence=confidence, checks=tuple(checks))


def texture_check(face_bgr: np.ndarray, thresholds: AntiSpoofThresholds) -> CheckResult:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    entropy = local_binary_pattern_entropy(gray)
    score = min(lap_var / thresholds.min_texture_laplacian_var, entropy / thresholds.min_lbp_entropy)
    passed = lap_var >= thresholds.min_texture_laplacian_var and entropy >= thresholds.min_lbp_entropy
    reason = f"laplacian_var={lap_var:.2f}, lbp_entropy={entropy:.2f}"
    return CheckResult("texture", passed, float(score), reason)


def reflection_check(face_bgr: np.ndarray, thresholds: AntiSpoofThresholds) -> CheckResult:
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    specular = ((value > np.percentile(value, 92)) & (saturation < np.percentile(saturation, 45))).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(specular, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.array([], dtype=np.int32)
    significant = areas[areas >= max(4, face_bgr.size // 45000)]
    glare_area = float(significant.sum()) / max(specular.size, 1)
    uniformity = glare_area / max(len(significant), 1)
    passed = len(significant) >= thresholds.min_specular_component_count and uniformity <= thresholds.max_specular_uniformity
    score = len(significant) / max(thresholds.min_specular_component_count, 1) - uniformity
    reason = f"specular_components={len(significant)}, uniformity={uniformity:.3f}"
    return CheckResult("reflection", passed, float(score), reason)


def depth_cue_check(landmarks: dict[str, Point], thresholds: AntiSpoofThresholds) -> CheckResult:
    left_eye = np.array(landmarks["left_eye"], dtype=np.float32)
    right_eye = np.array(landmarks["right_eye"], dtype=np.float32)
    nose = np.array(landmarks["nose_tip"], dtype=np.float32)
    chin = np.array(landmarks["chin"], dtype=np.float32)
    mouth_left = np.array(landmarks["mouth_left"], dtype=np.float32)
    mouth_right = np.array(landmarks["mouth_right"], dtype=np.float32)

    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (mouth_left + mouth_right) / 2.0
    face_height = max(float(np.linalg.norm(chin - eye_mid)), 1.0)
    eye_width = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)

    nose_offset = abs(float(nose[0] - eye_mid[0])) / eye_width
    mouth_offset = abs(float(mouth_mid[0] - eye_mid[0])) / eye_width
    vertical_structure = abs(float(nose[1] - eye_mid[1])) / face_height
    asymmetry = abs(nose_offset - mouth_offset)
    flat_score = max(0.0, 1.0 - (vertical_structure * 0.9 + asymmetry * 1.8 + nose_offset * 0.7))

    passed = flat_score <= thresholds.max_flat_depth_score
    reason = f"flat_depth_score={flat_score:.3f}"
    return CheckResult("depth_cues", passed, 1.0 - flat_score, reason)


def background_motion_check(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    face_box: tuple[int, int, int, int],
    thresholds: AntiSpoofThresholds,
) -> CheckResult:
    prev_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)

    x, y, w, h = clamp_box(face_box, current_bgr.shape[1], current_bgr.shape[0])
    face_mask = np.zeros(mag.shape, dtype=bool)
    face_mask[y : y + h, x : x + w] = True

    face_motion = float(np.median(mag[face_mask])) if np.any(face_mask) else 0.0
    background_motion = float(np.median(mag[~face_mask])) if np.any(~face_mask) else 0.0
    ratio = face_motion / max(background_motion, 1e-3)
    passed = ratio >= thresholds.min_face_background_flow_ratio
    reason = f"face_motion={face_motion:.3f}, background_motion={background_motion:.3f}, ratio={ratio:.2f}"
    return CheckResult("background_motion", passed, ratio, reason)


def screen_artifact_check(face_bgr: np.ndarray, thresholds: AntiSpoofThresholds) -> CheckResult:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    magnitude = np.log1p(np.abs(spectrum))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high_freq = radius > 55
    axis_grid = ((np.abs(xx - cx) < 3) | (np.abs(yy - cy) < 3)) & high_freq

    grid_energy = float(magnitude[axis_grid].mean())
    high_energy = float(magnitude[high_freq].mean())
    ratio = max(0.0, (grid_energy - high_energy) / max(high_energy, 1e-6))
    passed = ratio <= thresholds.max_screen_grid_energy_ratio
    reason = f"screen_grid_excess_ratio={ratio:.3f}"
    return CheckResult("screen_artifacts", passed, 1.0 - ratio, reason)


def model_check(
    face_bgr: np.ndarray,
    model_live_score_fn: Callable[[np.ndarray], float],
    thresholds: AntiSpoofThresholds,
) -> CheckResult:
    live_score = float(model_live_score_fn(face_bgr))
    passed = live_score >= thresholds.min_model_live_score
    reason = f"model_live_score={live_score:.3f}"
    return CheckResult("anti_spoofing_model", passed, live_score, reason)


def local_binary_pattern_entropy(gray: np.ndarray) -> float:
    center = gray[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    neighbors = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    for bit, neighbor in enumerate(neighbors):
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    hist = np.bincount(codes.ravel(), minlength=256).astype(np.float64)
    probabilities = hist / max(hist.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def yaw_delta(baseline: dict[str, Point], current: dict[str, Point]) -> float:
    return normalized_x(current["nose_tip"], current) - normalized_x(baseline["nose_tip"], baseline)


def smile_delta(baseline: dict[str, Point], current: dict[str, Point]) -> float:
    return mouth_width(current) / max(eye_width(current), 1e-6) - mouth_width(baseline) / max(eye_width(baseline), 1e-6)


def nod_delta(baseline: dict[str, Point], current: dict[str, Point]) -> float:
    return abs(normalized_y(current["nose_tip"], current) - normalized_y(baseline["nose_tip"], baseline))


def normalized_x(point: Point, landmarks: dict[str, Point]) -> float:
    left_eye = np.array(landmarks["left_eye"], dtype=np.float32)
    right_eye = np.array(landmarks["right_eye"], dtype=np.float32)
    eye_mid = (left_eye + right_eye) / 2.0
    return float((point[0] - eye_mid[0]) / max(np.linalg.norm(right_eye - left_eye), 1e-6))


def normalized_y(point: Point, landmarks: dict[str, Point]) -> float:
    left_eye = np.array(landmarks["left_eye"], dtype=np.float32)
    right_eye = np.array(landmarks["right_eye"], dtype=np.float32)
    eye_mid = (left_eye + right_eye) / 2.0
    return float((point[1] - eye_mid[1]) / max(np.linalg.norm(right_eye - left_eye), 1e-6))


def mouth_width(landmarks: dict[str, Point]) -> float:
    return float(np.linalg.norm(np.array(landmarks["mouth_right"]) - np.array(landmarks["mouth_left"])))


def eye_width(landmarks: dict[str, Point]) -> float:
    return float(np.linalg.norm(np.array(landmarks["right_eye"]) - np.array(landmarks["left_eye"])))


def crop_box(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = clamp_box(box, frame_bgr.shape[1], frame_bgr.shape[0])
    return frame_bgr[y : y + h, x : x + w]


def clamp_box(box: tuple[int, int, int, int], max_w: int, max_h: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(int(x), max_w - 1))
    y = max(0, min(int(y), max_h - 1))
    w = max(1, min(int(w), max_w - x))
    h = max(1, min(int(h), max_h - y))
    return x, y, w, h


def summarize(decision: AntiSpoofDecision) -> str:
    status = "LIVE" if decision.live else "SPOOF"
    lines = [f"{status} confidence={decision.confidence:.2f}"]
    lines.extend(
        f"- {check.name}: {'pass' if check.passed else 'fail'} score={check.score:.3f} ({check.reason})"
        for check in decision.checks
    )
    return "\n".join(lines)
