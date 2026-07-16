# 🟣 AI03-Project-1: 경구약제 이미지 객체 검출 (Object Detection)
> **소형 객체 누락 및 데이터 불균형 한계를 극복하기 위한 2-Stage 모델 엔지니어링 & 하이브리드 앙상블 파이프라인**

## ⚪ Intro
- **프로젝트 기간:** 7/15(화) ~ 7/31(목)
- **최종 발표:** 7/31(목) 
- **[최종 보고서 (PDF)](https://drive.google.com/file/d/1voFiNrO8XJeMeRKljphgkNjDnpDwOGYt/view?usp=sharing)**
- **멤버 및 역할:**
  * **전수현 (Project Lead):** Baseline 구축, 학습 및 평가 파이프라인 개발
  * **조은영 (Data Engineer):** Albumentations 기반 데이터 Augmentation, 데이터 전처리 및 EDA
  * **배진석 (ML Engineer):** YOLO 아키텍처 실험 및 성능 튜닝
  * **조계승 (AI Engineer / R&D):** Faster R-CNN 백본(ResNet-50) 설계, 소형 객체 특화 Anchor 조정 및 3차 실험 리드, Two-Phase 분리 학습 기법 설계, WBF 이종 앙상블 파이프라인 구축

---

## ⚪ Experiments

### 1️⃣ 소형 객체(미세 알약) 탐지를 위한 Anchor Parameter Optimization
* **문제 정의:** 일반적인 Object Detection 모델은 Feature Map 축소 과정에서 극소형 객체의 픽셀 정보가 유실되어 바운딩 박스가 누락되는 현상이 빈번했습니다.
* **해결 방안:** 데이터셋 내 소형 객체의 종횡비(Ratio) 및 스케일을 전수 정량 조사하여, Anchor Box 크기를 극소 영역까지 커버할 수 있도록 하이퍼파라미터를 재조정했습니다. 
* **결과:** 단일 Faster R-CNN 기준 baseline 대비 **mAP@0.5 성능을 0.6034에서 0.6383으로 극대화**했습니다.

### 2️⃣ 훈련 단계 분리 학습 (Two-Phase Training) 기법 설계
* **문제 정의:** 무거운 사전 학습(Pre-trained) 백본과 초기화 상태의 검출 헤드(R-CNN Head)를 동시에 End-to-End로 학습할 경우, 학습 초기 안정되지 않은 오차가 역전파되어 백본의 피처 추출 가중치를 훼손하는 현상이 발생했습니다.
* **해결 전략:**
  * **Phase 1 (Backbone Freezing, 1~20 Epoch):** Backbone 가중치를 고정하고 검출 헤드 레이어만 높은 학습률($LR=1e-5$)로 빠르게 수렴 유도.
  * **Phase 2 (Whole-Network Fine-Tuning, 21~50 Epoch):** 가중치 락을 전면 해제하고 미세 학습률($LR=1e-6$)로 전체 가중치를 부드럽게 동조 정밀 튜닝.
* **결과:** 모델의 안정적인 수렴과 미세 패턴(알약 각인, 경계선) 검출력의 극적인 강건성 확보.

---

## ⚪ Quantitative Performance (정량적 성과)

| Model 설정 | mAP @ 0.5 | 특이 사항 |
| :--- | :---: | :--- |
| **1차: ResNet-50 Baseline** | 0.6034 | 기본적인 Faster R-CNN 베이스라인 성능 |
| **2차: MobileNetV3 (경량형)** | 0.5712 | 모바일 배포를 위한 경량화 검증 (소형 탐지 성능 하락) |
| **3차: Faster R-CNN + Anchor 튜닝 & Two-Phase** | **0.6383** | **소형 객체 검출 능력 대폭 향상 (단일 최고 모델)** |
| **YOLOv8 단일 모델** | 0.9922 | 1-Stage 기반의 전체 검출 성능 우수 모델 |
| **Final WBF Ensemble (YOLOv8 + Faster R-CNN 3차)** | **0.9947** | **서로 다른 아키텍처 간 상호보완을 통한 최종 스코어** |

---

## ⚪ 프로젝트 구조 (Directory Structure)
```bash
AI03-Project-1/
├── data/                    # 원본 및 전처리 데이터
├── notebooks/               # 실험 및 EDA (BASELINE.ipynb, 개인 실험 노트북)
├── src/                     # 핵심 구현 소스코드 ⭐
│   ├── assets/              # 데이터 로더, transforms, 전처리 (은영)
│   ├── model.py             # YOLOv8 및 Faster R-CNN 백본/헤드 아키텍처 설계 (계승, 진석)
│   ├── train.py             # Two-Phase 학습 루프 및 학습 엔진 (수현)
│   ├── eval.py              # 검증 및 메트릭 산출 (수현)
│   ├── ensemble.py          # Weighted Boxes Fusion(WBF) 앙상블 모듈 (계승)
│   └── config.py            # 하이퍼파라미터 configuration
├── experiments/             # 실험 설정(YAML) 및 결과 메트릭 저장소
└── output/                  # 최종 Submission 캐글 파일 (.csv) 및 오차행렬 시각화 이미지
