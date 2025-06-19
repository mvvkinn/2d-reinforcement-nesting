# Reinforcement Learning for 2D Nesting Optimization

이 프로젝트는 강화학습(Reinforcement Learning)을 활용하여 2D 도면 자동 배치(Nesting) 최적화 문제를 해결하는 솔루션입니다. 주어진 도면(DXF 파일)을 기반으로 최적의 배치를 자동으로 찾아내는 AI 모델을 구현하였습니다.

## 프로젝트 개요

2D Nesting은 제한된 공간(시트)에 여러 개의 2D 도형을 최적으로 배치하여 재료 낭비를 최소화하는 문제입니다. 이 프로젝트에서는 다음과 같은 기능을 제공합니다:

- DXF 파일 형식의 2D 도면 처리 및 분석
- 도면의 기하학적 특성 추출 및 벡터화
- 강화학습 기반 배치 최적화 알고리즘 구현
- 학습된 모델을 통한 실시간 예측 및 시각화 GUI

## 시스템 요구사항

- Python 3.8 이상
- PyTorch 1.12 이상
- CUDA 지원 GPU (권장)
- 필수 라이브러리: ezdxf, shapely, scikit-learn, matplotlib, PySide6 등

## 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/reinforce-nesting.git
cd reinforce-nesting
```

2. 가상 환경 생성 및 활성화:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/MacOS
source .venv/bin/activate
```

3. 필수 라이브러리 설치:
```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
reinforce-nesting/
├── environment.py        # 강화학습 환경 구현
├── policy.py             # 정책 네트워크 모델 구현
├── train.py              # 모델 학습 스크립트
├── run_gui_predict.py    # GUI 기반 예측 도구
├── nesting_pro_bl_nfp.ipynb # 주피터 노트북 (데이터 전처리 및 실험용)
├── utils/                # 유틸리티 함수 모음
├── models/               # 학습된 모델 저장 디렉토리 
└── dataset/              # 데이터셋 디렉토리
```

## 주요 구성 요소

### 1. 환경 (environment.py)

강화학습을 위한 맞춤형 환경을 구현한 모듈입니다. `Gymnasium` 인터페이스를 따르며 다음과 같은 기능을 제공합니다:

- 2D 도형의 배치 시뮬레이션
- 충돌 감지 및 경계 검사
- 보상 함수 계산 (면적, 거리, 경계 기반)
- 도형 간 상호작용 처리

### 2. 정책 네트워크 (policy.py)

강화학습의 에이전트 역할을 하는 심층 신경망 모델입니다:

- 도형의 특성을 입력으로 받아 최적의 배치 액션(위치, 회전)을 출력
- Actor-Critic 아키텍처 사용
- 가우시안 정책을 통한 연속 행동 공간 탐색

### 3. 학습 스크립트 (train.py)

강화학습 모델을 훈련시키기 위한 스크립트입니다:

- PPO(Proximal Policy Optimization) 알고리즘 구현
- 체크포인트 저장 및 로드 기능
- 학습 진행 상태 모니터링 및 로깅

### 4. GUI 예측 도구 (run_gui_predict.py)

학습된 모델을 사용하여 실제 DXF 파일을 배치하는 그래픽 사용자 인터페이스:

- DXF 파일 로드 및 전처리
- 학습된 모델을 통한 배치 최적화
- 배치 결과 시각화 및 저장 기능

## 사용 방법

### 모델 학습

모델을 처음부터 학습하려면:

```bash
python train.py
```

학습 중간에 중단된 모델을 이어서 학습하려면:

```bash
python train.py --resume
```

### GUI 예측 도구 실행

```bash
python run_gui_predict.py
```

1. "DXF 파일 선택 및 예측" 버튼을 클릭합니다.
2. 배치하고 싶은 DXF 파일들을 선택합니다 (최대 20개).
3. 모델이 자동으로 배치를 계산하고 결과를 시각화합니다.

## 기술적 세부 사항

### DXF 전처리 과정

1. ezdxf 라이브러리를 통한 DXF 파일 파싱
2. 폴리곤 추출 및 정규화
3. 기하학적 특성 추출 (면적, 둘레, 컴팩트니스 등)
4. 푸리에 디스크립터 계산을 통한 형상 표현
5. 특성 벡터 정규화 및 패딩

### 강화학습 접근 방식

- **상태 표현**: 도형의 특성 + 시트의 상태 + 배치 히스토리
- **행동 공간**: 위치 조정 (x, y) + 회전 각도 (theta)
- **보상 함수**: 
  - 면적 활용 보상 (큰 도형 우선)
  - 근접성 보상 (도형 간 밀집도)
  - 경계 활용 보상 (가장자리 활용)
- **알고리즘**: PPO(Proximal Policy Optimization)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 기여 방법

1. 이 저장소를 포크합니다.
2. 새 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`).
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`).
5. Pull Request를 생성합니다.

## 문의 사항

질문이나 문제가 있으면 이슈를 생성하거나 [이메일 주소]로 연락하세요.
