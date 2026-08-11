"""
Geophone Sensor Client (Raspberry Pi 4 Model B에서 실행)
---------------------------------------------------------
ADS1115(I2C ADC)로 Geophone 전압을 읽어서, 다른 건물에 있는 백엔드 서버로
Tailscale을 통해 실시간 전송합니다.
"""

import asyncio
import json
import time

import websockets

SERVER_HOST = "100.112.45.74"      # 랩실 컴퓨터의 Tailscale IP
SERVER_PORT = 8000
TARGET_FS = 100                # 원래 학습 데이터를 만들 때의 샘플링레이트 (참고용 목표치)
ADS_GAIN = 16                   # 측정범위 ±0.256V - 학습 데이터의 전압 양자화 단위(7.8125uV)와 정확히 일치

WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws/ingest"

_ads_channel = None


def read_voltage() -> float:
    global _ads_channel
    if _ads_channel is None:
        import board
        import busio
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn

        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)
        ads.gain = ADS_GAIN
        _ads_channel = AnalogIn(ads, 0, 1) # A0 핀 사용

    return _ads_channel.voltage


async def main():
    print(f"백엔드로 연결 시도: {WS_URL}")
    async with websockets.connect(WS_URL) as ws:
        print(f"연결 성공! GAIN={ADS_GAIN}, 목표 FS={TARGET_FS}Hz로 전송 시작합니다. (Ctrl+C로 종료)")
        interval = 1.0 / TARGET_FS
        next_tick = time.perf_counter()
        count = 0
        while True:
            voltage = read_voltage()
            t_wall = time.time()  # 백엔드가 실제 경과시간을 계산할 수 있도록 타임스탬프도 같이 전송
            await ws.send(json.dumps({"voltage": voltage, "t_wall": t_wall}))

            count += 1
            if count % (int(TARGET_FS) * 5) == 0:  # 5초마다 상태 출력
                print(f"전송 중... 최근 voltage={voltage:.8f}")

            next_tick += interval
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()  # 밀렸으면 리셋 (ADC/I2C가 목표 속도보다 느릴 수 있음)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n종료합니다.")