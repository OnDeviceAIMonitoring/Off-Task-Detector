# Off-Task Detector

웹캠을 통해 사용자가 딴 짓(Off-Task)을 하는지 실시간으로 인지하는 알고리즘입니다. 본 프로젝트는 임베디드 환경(라즈베리 파이 5)에서 동작하도록 최적화되어 있으며, 객체 탐지와 포즈 추정 모델을 결합하여 사용자의 집중 상태를 효과적으로 판단합니다.

## 주요 기능

- **실시간 Off-Task 감지**: 웹캠 영상을 분석하여 사용자가 딴 짓(예: 휴대폰 사용, 자세 흐트러짐, 손이 책상 위에 없는 경우 등)을 하는지 감지합니다.
- **경량화된 모델 사용**: YOLO26n 객체 탐지 모델을 int8로 양자화하여 임베디드 환경에서도 빠른 추론이 가능합니다.
- **포즈 추정**: MediaPipe 기반의 포즈 및 얼굴 랜드마크 추정으로 사용자의 자세와 표정(웃음, 말하기 등)을 분석합니다.
- **다양한 상태 시각화**: 탐지 결과를 실시간으로 화면에 오버레이하여 시각적으로 확인할 수 있습니다.

## 사용 모델 및 환경

- **Object Detection**: YOLO26n (int8 양자화, TFLite/ONNX 지원)
- **Pose Estimation**: MediaPipe Holistic
- **추론 환경**: Raspberry Pi 5 (임베디드 환경 최적화)
- **프레임워크**: Python 3.x, OpenCV, NumPy 등

## 설치 방법

1. Python 3.x 환경을 준비합니다.
2. 필수 패키지를 설치합니다.
	```bash
	pip install opencv-python mediapipe numpy tflite-runtime
	```
	(ONNX 백엔드 사용 시 `onnxruntime` 추가 설치 필요)

3. 모델 파일(`yolo26n_int8.tflite`, `yolo26n.onnx`)을 프로젝트 폴더에 위치시킵니다.

4. 설정 파일(`config_off_task_det.json`)을 필요에 맞게 수정합니다.

## 실행 방법

```bash
python off_task_detection.py
```

- 웹캠이 자동으로 인식되며, 실시간으로 Off-Task 상태를 감지합니다.
- 주요 파라미터 및 임계값은 `config_off_task_det.json`에서 조정할 수 있습니다.

## 파일 구조

- `off_task_detection.py` : 메인 실행 및 분석 로직
- `convert_yolopt_to_tflite.py` : YOLO 모델 변환 스크립트
- `tracker_viz_utils.py` : 트래커 시각화 유틸리티
- `config_off_task_det.json` : 설정 파일

## 참고 사항

- 라즈베리 파이 등 임베디드 환경에서 실시간 추론이 가능하도록 경량화 및 최적화되어 있습니다.
- 다양한 객체(휴대폰 등) 탐지 및 포즈/표정 분석을 통해 보다 정확한 Off-Task 감지가 가능합니다.
