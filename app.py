"""
Geophone Intrusion Detection Dashboard - Demo Backend
------------------------------------------------------
가짜 지오폰 신호를 생성해 대시보드를 시연합니다.
peak 누적 개수를 기준으로 STABLE -> SUSPICIOUS -> WALK -> INTRUSION 4단계로 판정합니다.

실행:
    pip install -r requirements.txt
    python app.py
브라우저: http://localhost:8000
"""

import asyncio
import json
import random
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.signal import find_peaks, peak_prominences, peak_widths


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(schedule_monitor())
    asyncio.create_task(daily_reset_monitor())
    asyncio.create_task(fake_signal_loop())
    yield


app = FastAPI(title="Geophone Intrusion Detection Dashboard", lifespan=lifespan)

# ---------------------------------------------------------------
# 학습된 RandomForest 모델 로드 (har_geophone_rf_model_v1.pkl)
# ---------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model" / "har_geophone_rf_model_v1_fixed.pkl"
try:
    ML_MODEL = joblib.load(MODEL_PATH)
    FEATURE_COLS = list(ML_MODEL.feature_names_in_)
    MODEL_LOADED = True
    print(f"[모델 로드 성공] {MODEL_PATH.name} | features={FEATURE_COLS}")
except Exception as e:
    ML_MODEL = None
    FEATURE_COLS = []
    MODEL_LOADED = False
    print(f"[경고] 모델을 불러오지 못했습니다 ({e}). model/{MODEL_PATH.name} 파일 위치를 확인하세요.")

# 모델/알고리즘에 이미 적용돼 있다고 가정한 고정 threshold (감도 UI는 제거)
LOW_THRESHOLD = 3.0
WALK_WINDOW_SEC = 2.0


class SystemState:
    def __init__(self):
        self.mode = "DISARMED"  # "ARMED" | "DISARMED"
        self.allowed_start = dtime(8, 0)
        self.allowed_end = dtime(19, 0)
        self.today_walk_count = 0
        self.today_intrusion_count = 0
        self.today_date = datetime.now().date()
        self.events = deque(maxlen=50)
        self.connections: list[WebSocket] = []
        self.schedule_armed = not self.is_within_allowed_time()
        # 이벤트 로그 중복 방지용 상태. 연결(탭)마다 따로 두지 않고 여기 하나로 공유해서
        # 탭을 여러 개 열어도 이벤트/통계가 중복으로 쌓이지 않도록 함.
        self.last_logged_state = "STABLE"
        # 실제 Geophone 하드웨어가 /ws/ingest로 붙어있는지 여부 (붙어있으면 가짜 신호 생성 중단)
        self.hardware_connected = False
        self.ingest_connections = 0

    def is_within_allowed_time(self) -> bool:
        now = datetime.now().time()
        if self.allowed_start <= self.allowed_end:
            return self.allowed_start <= now <= self.allowed_end
        return now >= self.allowed_start or now <= self.allowed_end


state = SystemState()


async def broadcast_json(payload: dict):
    dead = []
    for ws in state.connections:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state.connections:
            state.connections.remove(ws)


async def set_mode(new_mode: str, source: str):
    """모드를 바꾸고, 연결된 모든 클라이언트에게 즉시 알림 (버튼 클릭이든 자동 스케줄이든 동일하게 처리)"""
    if new_mode == state.mode:
        return
    state.mode = new_mode
    await broadcast_json({"type": "mode_change", "mode": new_mode, "source": source})


class AllowedTimeUpdate(BaseModel):
    start: str
    end: str


@app.post("/api/arm")
async def arm_system():
    await set_mode("ARMED", "manual")
    return {"mode": state.mode}


@app.post("/api/disarm")
async def disarm_system():
    await set_mode("DISARMED", "manual")
    return {"mode": state.mode}


@app.post("/api/recalibrate")
async def recalibrate():
    """언제든 수동으로 영점 재측정을 다시 시작"""
    processor.start_calibration()
    return {"calibrating": True}


@app.post("/api/allowed-time")
async def set_allowed_time(payload: AllowedTimeUpdate):
    h1, m1 = map(int, payload.start.split(":"))
    h2, m2 = map(int, payload.end.split(":"))
    state.allowed_start = dtime(h1, m1)
    state.allowed_end = dtime(h2, m2)

    # 새 허용시간 기준으로 지금 당장 어떤 모드여야 하는지 즉시 재계산해서 반영
    should_be_armed = not state.is_within_allowed_time()
    state.schedule_armed = should_be_armed
    await set_mode("ARMED" if should_be_armed else "DISARMED", "schedule")

    return {"start": payload.start, "end": payload.end, "mode": state.mode}


@app.get("/api/status")
async def get_status():
    return {
        "mode": state.mode,
        "allowed_start": state.allowed_start.strftime("%H:%M"),
        "allowed_end": state.allowed_end.strftime("%H:%M"),
        "today_walk_count": state.today_walk_count,
        "today_intrusion_count": state.today_intrusion_count,
        "hardware_connected": state.hardware_connected,
    }


# ---------------------------------------------------------------
# 가짜 신호 생성기 (나중에 실제 ADC 읽기로 교체할 부분)
# ---------------------------------------------------------------
FS = 50
NOISE_LEVEL = 0.00005
_walk_burst_remaining = 0
_walk_burst_phase = 0.0


def generate_fake_signal() -> float:
    global _walk_burst_remaining, _walk_burst_phase
    base_noise = random.gauss(0, NOISE_LEVEL)

    if _walk_burst_remaining <= 0 and random.random() < 0.002:
        _walk_burst_remaining = FS * random.randint(3, 6)
        _walk_burst_phase = 0.0

    if _walk_burst_remaining > 0:
        _walk_burst_remaining -= 1
        _walk_burst_phase += 1
        step_freq = 1.8
        envelope = 0.0008 * (0.6 + 0.4 * random.random())
        walk_signal = envelope * np.sin(2 * np.pi * step_freq * _walk_burst_phase / FS)
        walk_signal *= 1 + 0.3 * random.random()
        return base_noise + walk_signal

    return base_noise


class StreamProcessor:
    """
    보행 감지 알고리즘 PDF의 배치(offline) 로직을 실시간 스트리밍용으로 이식.
    #1 캘리브레이션 -> #2 threshold 기반 1차 peak 탐지 -> #3 여진 제거
    -> #4 11개 feature 추출 -> #5 모델 예측 -> #6 다수결 필터
    """

    # peak 후보의 하강 구간까지 다 채워질 시간을 잠깐 기다렸다가 확정 (find_peaks가 온전한 모양을 보게 하기 위함)
    DETECT_LAG_SEC = 0.3
    MIN_DISTANCE_SEC = 0.3
    MIN_PROMINENCE = 4.0
    MIN_WIDTH_SEC = 0.05
    MAX_WIDTH_SEC = 0.50
    VOTING_WINDOW = 3
    VOTING_WALK_THRESHOLD = 2

    # 여진(echo) 제거: 최근 2초 내 더 큰 peak 대비 이 비율 미만 크기면 잔진동으로 간주
    ECHO_TIME_SEC = 2.0
    ECHO_RATIO = 0.3

    # 최종 walk 판정: 확률 기반 + 연속성 체크 (참고 코드와 동일 기준)
    WALK_SCORE_THRESHOLD = 0.75
    GAP_RESET_SEC = 1.0

    # 자동 영점 캘리브레이션
    CALIBRATION_SEC = 5
    NOISE_VAR_THRESHOLD = 0.0002
    CAL_Z_THRESHOLD = 5.0

    def __init__(self):
        self.win_sec = 0.15
        self.buffer = deque(maxlen=5000)      # (t, voltage) - RMS envelope 계산용
        self.z_history = deque(maxlen=5000)    # (t, z) - peak 탐지용 (약 8초 분량 유지)
        self.t = 0.0
        self._last_t_wall = None  # 실제 하드웨어가 보내주는 타임스탬프 추적용
        # 데모용 초기 캘리브레이션 값 (실제로는 quiet_base 파일로 계산해야 함 - PDF #1 단계)
        self.global_med = 0.00005
        self.global_mad = 0.0000077

        self.processed_peak_times: set[float] = set()
        self.confirmed_peaks: list[dict] = []      # feature 계산용 최근 peak 컨텍스트
        self.recent_votes = deque(maxlen=self.VOTING_WINDOW)

        # 화면 상태 판정용: 최근 2초 이내에 있었던 "모델이 확정한 walk peak" / "noise로 분류된 peak"
        self.recent_walk_times: list[float] = []
        self.recent_noise_times: list[float] = []

        # 자동 영점 캘리브레이션 상태
        self.calibrating = False
        self.calibrated = False
        self._cal_chunk: list[float] = []
        self._cal_chunk_start_t: Optional[float] = None
        self._cal_quiet_samples: list[float] = []
        self.calibration_seconds_done = 0

    def start_calibration(self):
        """하드웨어가 새로 연결되거나 수동 재보정 요청 시 호출. 5초간 조용함을 확인해서 기준값을 새로 계산."""
        self.calibrating = True
        self.calibrated = False
        self._cal_chunk = []
        self._cal_chunk_start_t = None
        self._cal_quiet_samples = []
        self.calibration_seconds_done = 0

    def _current_fs(self) -> float:
        """
        최근 버퍼의 실제 타임스탬프 간격으로 유효 샘플링레이트를 추정.
        가짜 신호(50Hz)든 실제 하드웨어(예: ~68.5Hz)든 코드 수정 없이 자동으로 맞춰 씀.
        """
        if len(self.z_history) < 10:
            return FS
        times = np.array([p[0] for p in list(self.z_history)[-50:]])
        dts = np.diff(times)
        dts = dts[dts > 0]
        if len(dts) == 0:
            return FS
        return float(1.0 / np.median(dts))

    def add_sample(self, voltage: float, t_wall: Optional[float] = None):
        # 실제 하드웨어가 타임스탬프(t_wall)를 같이 보내주면 그걸로 실제 경과시간을 반영.
        # (가짜 신호처럼 t_wall이 없으면 FS 기준 명목상 간격만큼 흘려보냄)
        if t_wall is not None and self._last_t_wall is not None:
            dt = t_wall - self._last_t_wall
            self.t += dt if dt > 0 else 1.0 / FS
        else:
            self.t += 1.0 / FS
        self._last_t_wall = t_wall

        if self.calibrating:
            return self._process_calibration_sample(voltage)

        self.buffer.append((self.t, voltage))
        # RMS envelope 계산에 필요한 win_sec 구간만 남기고 오래된 원신호는 정리
        while self.buffer and self.t - self.buffer[0][0] > 2.0:
            self.buffer.popleft()

        win_start = self.t - self.win_sec
        recent = [v for (tt, v) in self.buffer if tt >= win_start]
        if len(recent) < 3:
            return None
        env = float(np.sqrt(np.mean(np.square(recent))))
        z = (env - self.global_med) / (1.4826 * self.global_mad + 1e-9)
        self.z_history.append((self.t, z))
        while self.z_history and self.t - self.z_history[0][0] > 8.0:
            self.z_history.popleft()

        # 2초 지난 기록은 매 tick마다 정리 -> 아무 일 없으면 자동으로 STABLE 복귀
        self.recent_walk_times = [pt for pt in self.recent_walk_times if self.t - pt < WALK_WINDOW_SEC]
        self.recent_noise_times = [pt for pt in self.recent_noise_times if self.t - pt < WALK_WINDOW_SEC]

        # 마지막 처리된 peak 이후 GAP_RESET_SEC 이상 아무것도 없으면 voting queue 리셋 (참고 코드와 동일)
        if self.confirmed_peaks and (self.t - self.confirmed_peaks[-1]["time"]) > self.GAP_RESET_SEC and self.recent_votes:
            self.recent_votes.clear()

        peak_result = self._detect_and_classify_peak()

        return {
            "t": self.t, "voltage": voltage, "z": z, "threshold": LOW_THRESHOLD,
            "peak": peak_result is not None,
            "peak_type": peak_result["label"] if peak_result else None,
            "peak_height": peak_result["height"] if peak_result else None,
            "peak_lag_samples": peak_result["lag_samples"] if peak_result else None,
        }

    def _process_calibration_sample(self, voltage: float) -> dict:
        """5초간 조용함을 확인하며 global_med/global_mad를 새로 계산 (참고 코드의 캘리브레이션 단계와 동일)."""
        if self._cal_chunk_start_t is None:
            self._cal_chunk_start_t = self.t
        self._cal_chunk.append(voltage)

        if self.t - self._cal_chunk_start_t >= 1.0:
            arr = np.array(self._cal_chunk)
            cur_var = float(np.var(arr))

            win = max(1, int(FS * self.win_sec))
            kernel = np.ones(win) / win
            x_centered = arr - np.median(arr)
            env = np.sqrt(np.convolve(x_centered ** 2, kernel, mode="same"))
            med = float(np.median(env))
            mad = float(np.median(np.abs(env - med))) or 1e-6
            cur_cal_z = float(np.max((env - med) / (1.4826 * mad))) if len(env) else 0.0

            if cur_var < self.NOISE_VAR_THRESHOLD and cur_cal_z < self.CAL_Z_THRESHOLD:
                self.calibration_seconds_done += 1
                self._cal_quiet_samples.extend(self._cal_chunk)
            else:
                # 진동이 감지되면 처음부터 다시
                self.calibration_seconds_done = 0
                self._cal_quiet_samples = []

            self._cal_chunk = []
            self._cal_chunk_start_t = self.t

            if self.calibration_seconds_done >= self.CALIBRATION_SEC:
                arr_all = np.array(self._cal_quiet_samples)
                x_centered = arr_all - np.median(arr_all)
                env_all = np.sqrt(np.convolve(x_centered ** 2, kernel, mode="same"))
                self.global_med = float(np.median(env_all))
                self.global_mad = float(np.median(np.abs(env_all - self.global_med))) or 1e-6
                self.calibrating = False
                self.calibrated = True

        return {"calibrating": True, "progress": f"{self.calibration_seconds_done}/{self.CALIBRATION_SEC}"}

    def _detect_and_classify_peak(self) -> Optional[dict]:
        if not MODEL_LOADED or len(self.z_history) < 10:
            return None

        fs = self._current_fs()  # 실제 들어오는 속도에 맞춰 매번 재추정
        times = np.array([p[0] for p in self.z_history])
        zvals = np.array([p[1] for p in self.z_history])

        raw_peaks, _ = find_peaks(
            zvals, height=LOW_THRESHOLD,
            distance=max(1, int(fs * self.MIN_DISTANCE_SEC)),
            prominence=self.MIN_PROMINENCE,
            width=[max(1, int(fs * self.MIN_WIDTH_SEC)), max(2, int(fs * self.MAX_WIDTH_SEC))],
        )
        if len(raw_peaks) == 0:
            return None

        latest_idx = raw_peaks[-1]
        peak_time = float(times[latest_idx])
        peak_height = float(zvals[latest_idx])

        # 하강 구간이 아직 덜 채워졌으면(방금 막 생긴 peak) 다음 tick에 다시 확인
        if self.t - peak_time < self.DETECT_LAG_SEC:
            return None
        if peak_time in self.processed_peak_times:
            return None
        self.processed_peak_times.add(peak_time)

        # ---- PDF #3 단계와 동일한 여진(echo) 제거: 최근 2초 내 더 큰 peak 대비 30% 미만 크기면 무시 ----
        for prev in reversed(self.confirmed_peaks):
            dt = peak_time - prev["time"]
            if dt > self.ECHO_TIME_SEC:
                break
            if peak_height < prev["height"] * self.ECHO_RATIO:
                return None  # 여진으로 판단 -> confirmed_peaks에도 추가하지 않고 그냥 무시

        # ---- PDF #4 단계와 동일한 11개 feature 추출 ----
        prominence = float(peak_prominences(zvals, [latest_idx])[0][0])
        width_sec = float(peak_widths(zvals, [latest_idx], rel_height=0.5)[0][0] / fs)

        half_win = int(fs * 0.25)
        start_idx = max(0, latest_idx - half_win)
        end_idx = min(len(zvals), latest_idx + half_win + 1)
        local_z = zvals[start_idx:end_idx]

        local_rms = float(np.sqrt(np.mean(local_z ** 2)))
        local_energy = float(np.sum(local_z ** 2))
        area_above_thresh = float(np.sum(np.maximum(local_z - LOW_THRESHOLD, 0)) / fs)

        if not self.confirmed_peaks:
            is_first_peak, interval_prev = 1, 0.5386
            height_ratio_prev, prominence_ratio_prev = 1.0, 1.0
        else:
            prev = self.confirmed_peaks[-1]
            interval_prev = peak_time - prev["time"]
            if interval_prev > 3.0:
                is_first_peak, interval_prev = 1, 0.5386
                height_ratio_prev, prominence_ratio_prev = 1.0, 1.0
            else:
                is_first_peak = 0
                height_ratio_prev = peak_height / prev["height"] if prev["height"] > 0 else 1.0
                prominence_ratio_prev = prominence / prev["prominence"] if prev["prominence"] > 0 else 1.0

        recent_peak_count_2s = sum(1 for p in self.confirmed_peaks if 0 < (peak_time - p["time"]) <= 2.0)

        features = {
            "peak_height": peak_height,
            "peak_prominence": prominence,
            "peak_width_sec": width_sec,
            "local_rms": local_rms,
            "local_energy": local_energy,
            "area_above_threshold": area_above_thresh,
            "interval_prev": interval_prev,
            "recent_peak_count_2s": recent_peak_count_2s,
            "height_ratio_prev": height_ratio_prev,
            "prominence_ratio_prev": prominence_ratio_prev,
            "is_first_peak": is_first_peak,
        }
        X = pd.DataFrame([features], columns=FEATURE_COLS)
        pred_binary = int(ML_MODEL.predict(X)[0])  # 1=walk, 0=noise (모델 학습 시 라벨과 동일)
        if hasattr(ML_MODEL, "predict_proba"):
            walk_score = float(ML_MODEL.predict_proba(X)[0][1])
        else:
            walk_score = float(pred_binary)

        self.confirmed_peaks.append({"time": peak_time, "height": peak_height, "prominence": prominence})
        self.confirmed_peaks = [p for p in self.confirmed_peaks if peak_time - p["time"] < 5.0]

        # ---- PDF #6 단계 다수결 필터 + 확률(walk_score) + 연속성(continuity) 체크 ----
        # (peak 1개만으로는 절대 최종 walk로 인정하지 않음 - 최소 2표 이상 필요)
        self.recent_votes.append(pred_binary)
        walk_votes = sum(self.recent_votes)
        continuity_ok = (
            is_first_peak == 0
            and interval_prev <= self.GAP_RESET_SEC
            and recent_peak_count_2s >= 1
        )
        final_walk = (
            len(self.recent_votes) >= 2
            and walk_votes >= self.VOTING_WALK_THRESHOLD
            and pred_binary == 1
            and walk_score >= self.WALK_SCORE_THRESHOLD
            and continuity_ok
        )

        if final_walk:
            self.recent_walk_times.append(peak_time)
        else:
            self.recent_noise_times.append(peak_time)

        return {
            "label": "walk" if final_walk else "noise",
            "time": peak_time,
            "height": peak_height,  # 실제 peak 지점의 z값 (지금 이 순간의 z가 아님)
            "lag_samples": max(0, round((self.t - peak_time) * fs)),  # 몇 틱 전에 실제로 일어났는지
        }


processor = StreamProcessor()


def compute_current_state():
    """
    모델이 최종 확정한 peak 라벨(walk/noise) 기반 판정. ARMED와 DISARMED는 서로 다른 상태 집합을 가짐.

    ARMED(외출 모드): 침입 감지가 활성화된 상태이므로 확정된 보행이 곧 침입.
        - 최근 2초 내 확정 walk peak 없음: STABLE (noise peak만 있으면 SUSPICIOUS)
        - 최근 2초 내 noise peak만 있음: SUSPICIOUS - 애매함, 조금 더 지켜봄
        - 최근 2초 내 확정 walk peak 있음: INTRUSION

    DISARMED(복귀 모드): 침입 감지를 하지 않는 상태이므로 보행은 그냥 보행.
        - 확정 walk peak 없음: STABLE
        - 확정 walk peak 있음: WALK - 정상적인 보행 (침입 아님)
    """
    now = processor.t
    has_recent_walk = any(now - pt <= WALK_WINDOW_SEC for pt in processor.recent_walk_times)
    has_recent_noise = any(now - pt <= WALK_WINDOW_SEC for pt in processor.recent_noise_times)

    if state.mode == "ARMED":
        if has_recent_walk:
            return "INTRUSION", "HIGH"
        if has_recent_noise:
            return "SUSPICIOUS", "MEDIUM"
        return "STABLE", "LOW"
    else:  # DISARMED
        if has_recent_walk:
            return "WALK", "LOW"
        return "STABLE", "LOW"


async def handle_new_sample(voltage: float, t_wall: Optional[float] = None):
    """
    전압 샘플 1개를 처리해서 모든 프론트엔드 탭에 브로드캐스트.
    가짜 신호 루프든, 실제 하드웨어(/ws/ingest)든 결국 이 함수 하나로 들어옴.
    """
    result = processor.add_sample(voltage, t_wall)
    if result is None:
        return

    if result.get("calibrating"):
        await broadcast_json({"type": "calibration", "progress": result["progress"]})
        return

    current_state, risk = compute_current_state()

    event = None
    if current_state != state.last_logged_state and current_state != "STABLE":
        event = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "label": current_state,
            "risk": risk,
            "z_score": round(result["z"], 1),
        }
        state.events.appendleft(event)
        if current_state == "WALK":
            state.today_walk_count += 1
        elif current_state == "INTRUSION":
            state.today_intrusion_count += 1
    state.last_logged_state = current_state

    payload = {
        "type": "sample",
        "t": round(result["t"], 3),
        "voltage": voltage,
        "z": round(result["z"], 2),
        "threshold": result["threshold"],
        "peak": result["peak"],
        "peak_type": result["peak_type"],
        "peak_height": result["peak_height"],
        "peak_lag_samples": result["peak_lag_samples"],
        "mode": state.mode,
        "allowed": state.is_within_allowed_time(),
        "current_state": current_state,
        "risk": risk,
        "event": event,
        "hardware_connected": state.hardware_connected,
        "stats": {
            "walk": state.today_walk_count,
            "intrusion": state.today_intrusion_count,
        },
    }
    await broadcast_json(payload)


@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    """프론트엔드(브라우저) 전용. 데이터를 직접 만들지 않고 handle_new_sample()이 뿌려주는 걸 받기만 함."""
    await websocket.accept()
    state.connections.append(websocket)
    try:
        while True:
            # 브라우저 쪽에서 별다른 메시지를 보내진 않지만, 연결 끊김 감지를 위해 대기
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in state.connections:
            state.connections.remove(websocket)


@app.websocket("/ws/ingest")
async def ingest_endpoint(websocket: WebSocket):
    """
    실제 Geophone/ADC 쪽(라즈베리파이 등)에서 붙는 엔드포인트.
    { "voltage": 0.000123 } 형태의 JSON을 계속 보내주면 됨.
    이 연결이 살아있는 동안은 가짜 신호 생성기가 자동으로 멈춤.
    """
    await websocket.accept()
    was_connected = state.hardware_connected
    state.hardware_connected = True
    state.ingest_connections += 1
    if not was_connected:
        processor.start_calibration()
        print("[하드웨어 연결됨] /ws/ingest 클라이언트 접속 - 가짜 신호 생성 중단, 자동 영점 캘리브레이션 시작")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                voltage = float(msg["voltage"])
                t_wall = float(msg["t_wall"]) if "t_wall" in msg and msg["t_wall"] is not None else None
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            await handle_new_sample(voltage, t_wall)
    except WebSocketDisconnect:
        pass
    finally:
        state.ingest_connections = max(0, state.ingest_connections - 1)
        if state.ingest_connections == 0:
            state.hardware_connected = False
            print("[하드웨어 연결 끊김] /ws/ingest 클라이언트 종료 - 가짜 신호 생성 재개")


async def fake_signal_loop():
    """실제 하드웨어가 안 붙어있을 때만 동작하는 데모용 신호 생성 루프."""
    while True:
        if not state.hardware_connected:
            voltage = generate_fake_signal()
            await handle_new_sample(voltage)
        await asyncio.sleep(1.0 / FS)


async def schedule_monitor():
    """허용 시간 경계를 넘을 때마다 자동으로 ARMED/DISARMED 전환 + 모든 클라이언트에 알림"""
    while True:
        should_be_armed = not state.is_within_allowed_time()
        if should_be_armed != state.schedule_armed:
            state.schedule_armed = should_be_armed
            await set_mode("ARMED" if should_be_armed else "DISARMED", "schedule")
        await asyncio.sleep(1)


async def daily_reset_monitor():
    """날짜가 바뀌면(자정) 오늘 통계/이벤트 로그를 초기화하고 클라이언트에도 알림"""
    while True:
        today = datetime.now().date()
        if today != state.today_date:
            state.today_date = today
            state.today_walk_count = 0
            state.today_intrusion_count = 0
            state.events.clear()
            state.last_logged_state = "STABLE"
            await broadcast_json({"type": "daily_reset"})
        await asyncio.sleep(30)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)