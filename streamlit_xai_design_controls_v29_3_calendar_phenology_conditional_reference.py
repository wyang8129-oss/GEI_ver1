# =============================================================
# v29.3 Calendar × Phenology 이중정렬 + 조건부 우수작기 Reference 확장본
# - v29.2 통합 GEI 원인분해/생장균형/방향별 환경기여 분석 유지
# - 정식일 기준 DAP/WAP, 달력 월·계절, GDD(Base 사용자설정) 자동 계산
# - 외기온·외부일사 7일 맥락을 조사일과 정렬하여 작기 시작시점 차이 보정
# - WAP 1~4, 5~8 ... 생육단계/Phase별 GEI·NGR·환경 profile 저장
# - 상위 생산작기 중 현재 WAP·계절·GDD·외기온·외부일사가 유사한 작기 자동 검색
# - 고정 Reference 대신 Calendar × Phenology 조건부 Reference(Q25/Median/Q75) 생성
# - Lag SHAP 평가샘플을 조사일에 재연결해 Phase × Season별 Peak Lag 요약 가능
# - SQLite 기존 작기 레코드에 정식시점/thermal-time/phase profile을 JSON으로 누적 저장
# - Python 3.9 호환
# =============================================================

# =============================================================
# v29.0 누적 학습형 작기 Knowledge Base + 우수작기 Benchmarking 확장본
# - v28.3의 GEI 임계점 날짜매핑/원환경 맥락/XAI/Counterfactual 기능 유지
# - 작기별 환경·생육·수확·GEI·임계값·Lag SHAP·성과를 SQLite 표준 레코드로 저장
# - 다작기 ranking 및 상위 20% 우수작기 reference profile 자동 생성
# - 현재 작기 vs 우수작기 유사도, 편차, 위험노출 차이 실시간 진단
# - 저장된 작기별 GEI 가중치를 생산성 점수로 누적학습하여 historical GEI weight 제공
# - 작물×품종×계절 필터를 통한 비교집단 정합성 강화
# - CSV 내보내기/가져오기 및 DB 레코드 삭제 지원
# - Python 3.9 호환
# =============================================================

# =============================================================
# v28.3 GEI 임계점 시계열 + 원환경 맥락 선택표시 확장본
# - v28.2의 GEI-domain Centered ALE/관측 GEI 상세 임계값 조사일자 매핑 유지
# - 각 조사일 hover에 GEI, 보간 ALE, 관측-grid 반응, 생육 변화량, NGR 표시
# - 날짜 기반 GEI + NGR 시계열 위에 원환경 맥락 패널 선택 표시
# - 표시옵션: 평균·최저·최고 / 주간·야간 평균 / 없음
# - 원환경 요약은 현재 선택한 GEI 누적기간(window_days)과 동일 기간으로 계산
# - 온도/습도/CO2/일사량 자동 연동, 통합 GEI는 원환경 단일축 표시 제외
# - v28.1 데이터 기반 GEI 가중치 및 기존 XAI/GEI 기능 유지
# - Python 3.9 호환
# =============================================================

# =============================================================
# v28.1 데이터 기반 통합 GEI 가중치 최적화 확장본
# - 기존 동일가중 통합 GEI(각 환경 0.25)를 기준선으로 유지
# - SHAP mean(|SHAP|), 표준화 회귀계수, 절대 상관계수 기반 가중치 자동 추정
# - 제약조건 w>=0, sum(w)=1의 데이터 기반 탐색으로 최적화 가중치 도출
# - SHAP/회귀/상관/최적화 결과의 평균을 합의가중(Consensus)으로 제공
# - 원자료/변화량/월평균 NGR/생육단계 NGR 중 가중치 학습 반응기준 선택
# - 작기 메타데이터가 있으면 Leave-One-Crop-Cycle-Out, 없으면 TimeSeriesSplit 검증
# - 동일가중 vs 데이터 기반 가중 GEI의 CV R2/RMSE/상관 및 방향일치 비교
# - 선택 가중치를 현재 GEI 및 1~7주 GEI 데이터셋 전체에 일관 적용
# - Python 3.9 호환
# =============================================================

# =============================================================
# v28.0 GEI 정규화 생육반응 기준 개선본
# - GEI 반응곡선 기본 기준을 생육단계 기대 변화량 대비 방식으로 변경
# - 조사 간 변화량/일평균 변화량/7일환산 변화량 자동 계산
# - 월평균 변화량 대비(가능 시 leave-one-out 월평균) 신규 추가
# - 생육단계별 기대 7일 변화량(1~2차 추세) 대비 반응률 신규 추가
# - 0% = 기대 성장속도, 음수 = 실제 길이 감소가 아니라 기대 성장속도 대비 억제 의미
# - 기존 전체평균/절대추세/직전조사/수동 기준 모드는 호환 유지
# - GEI 정렬 반응추세 감소/위험 후보와 Centered ALE 임계값 교차확인 구조 유지
# - Python 3.9 호환
# =============================================================

# =============================================================
# v27.7 동적 수확 Target명 + 센서 일/주 파생Feature + 그래프 UI 이동
# - v27.6 기존 기능 유지
# - 센서 평균기간 주/일(2~6일) 선택에 따라 매핑 Feature명이 정확히 생성/표시
# - GEI 최고기간/GEI 증감/Cluster 반응 Target은 None 제외 + 실제 선택 컬럼명 표시
# - 평균과중1/2는 선택된 생체중/건물중 등 실제 컬럼명 그대로 표시
# - 머신러닝 입력 환경변수 선택 아래에 환경 그래프 선택 / 생육·수확 그래프 선택 이동
# - 그래프 선택에서도 실제 수확 원본 컬럼명을 표시하되 내부 canonical 컬럼과 안전 연결
# - GEI 비교 모델 ANN/BPM/SVM 지원
# - Python 3.9 호환
# =============================================================

# =============================================================
# v27.6 GEI 일/주 누적기간 + ANN/BPM/SVM + XAI 레이아웃 개선본
# - v27.5 기존 기능 유지
# - GEI 환경 누적기간: 일 단위 2~6일 + 주 단위 1~7주 선택
# - 비교 회귀모델: ANN(MLP), BPM(Bayesian Ridge), SVM(SVR) 추가
# - 월별 일사광 최저: 1 이상 유효값만 사용
# - GEI 생육 반응곡선: 수동 기준값 입력 유지/강화
# - Y-window 상세내용 Expander 추가
# - Local XAI: LIME / SHAP Waterfall / SHAP Force 정량결과 구조 유지
# - Centered ALE: 그래프+정량결과, 우호/불리 후보, 자동해석 구조 강화
# - Centered ALE와 Bootstrap CI/Threshold 사이 구분선 추가
# - Python 3.9 호환
# =============================================================

# =============================================================
# v27.5 GEI 정량·월별요약·상세임계·Rule 프로파일 수동선택 확장본
# - v27.4 기존 기능 유지
# - 3. GEI 상승 분석: 조사일별 정량표 + 일평균 변화량 + 월 + 변화상태
# - 월별 환경(온도/습도/CO2/일사량) 최저/평균/최고 + 생육·수확 통계
# - 4. GEI ALE: 조사일수/고유 GEI 수에 맞춘 상세 bins 선택 + 관측 GEI 상세 임계 스캔
# - 5. GEI 생육반응: 변화량 컬럼 + 수동 기준값 모드
# - GEI Rule-based 프로파일: 대표 GEI 구간조합 수동 선택
# - Python 3.9 호환
# =============================================================

# =============================================================
# v27.3.1 Cluster 동적 라벨 KeyError 수정본
# - v27.3 기존 기능 유지
# - 2~3개 클러스터링 변수 선택 시 동적 Cluster명을 exposure/PCA/상관분석에 동일 적용
# - 4개 변수 선택 시 기존 5개 사전정의 환경유형 유지
# - Python 3.9 호환
# =============================================================

# =============================================================
# v27.3 Cluster 변수선택 + XAI 레이아웃 + ALE 분석 강화
# - v26.9 기존 기능 100% 유지
# - 전역 변수 중요도: SHAP Summary(전체폭) + Model FI/Permutation 2열
# - 변수 상호작용: SHAP Dependence + SHAP Interaction 2열
# - 변수 효과·경향: ICE+PDP 및 ICE/PDP/통합 자동해석 유지·이동
# - 개별 예측 설명: LIME | SHAP Waterfall | SHAP Force Plot 3열
# - 각 신규 SHAP 분석에 그래프 + 정량적 결과 + 자동해석 추가
# - Temporal/Lag SHAP, GEI, ALE, Bootstrap CI, Counterfactual 기존 기능 유지
# - GEI 이후 KMeans 5개 환경유형 + PCA + 조사일별 노출시간/비율 + 생육·수확 변화 상관분석 추가
# - Python 3.9 호환
# =============================================================

# =============================================================
# v26.8 머신러닝 입력 환경변수 선택 + 그래프 수확항목 동적 확장본
# - v26.7 기존 기능 유지
# - 머신러닝 입력 Feature 선택: 전체/기본 7개/파생변수 포함/초기화
# - 선택 Feature를 모델 및 XAI 공통 Feature 목록에 적용
# - 그래프로 표시할 항목에 착과수/개화수/평균과중을 매핑 선택 시 동적 표시
# =============================================================

# =============================================================
# v26.9 Temporal/Lag SHAP 수확 Target 연동본
# - 수확량 데이터 컬럼 선택의 수확수/착과수/개화수/평균과중을 Temporal SHAP에 연동
# - 동일 Target을 Lag SHAP 및 Lag SHAP × Feature Heatmap에 연동
# - v26.8 머신러닝 Feature 선택 및 기존 v26.7 이하 기능 유지
# =============================================================

# v26.7 수확 Target 확장 + XAI 세로/2열 레이아웃 개선본
# - v26.6 기존 기능 100% 유지
# - 수확량 컬럼: 조사일자/수확수/착과수/개화수/평균과중 유지
# - GEI 최고기간 Target: 수확수/착과수/개화수/평균과중 포함
# - GEI 상승 생육·수확 분석 Target: 수확수/착과수/개화수/평균과중 포함
# - 모델 예측대상: 수확수/착과수/개화수/평균과중 포함
# - Global XAI: SHAP Summary + Dependence 2열 / FI + Permutation 2열
# - ICE+PDP와 LIME을 각각 전체 폭 1개씩 세로 배치
# - 기본 Centered ALE와 Bootstrap/Threshold ALE를 각각 전체 폭 1개씩 세로 배치
# - Python 3.9 호환
# =============================================================

# =============================================================
# v26.6 GEI 기반 생육 반응 곡선 확장본
# - v26.5의 기존 GEI/XAI/Lag SHAP/Centered ALE 기능 100% 유지
# - GEI 기반 생육 반응 곡선 신규 추가
# - 조사일별 GEI ↔ 평균 대비 생육·수확 변화율(%) 산점도
# - 증가/유지/감소/위험 영역 자동 분류
# - 전체 조사 평균 / 생육추세 기대값 / 직전 조사 대비 기준 선택
# - GEI 정렬 이동평균 추세선, 감소 시작/위험 후보 자동 탐지
# - ALE 임계점과 생육 반응 곡선 임계점 동시 비교
# - 정량 결과표 + 자동 해석
# - Python 3.9 호환
# =============================================================

# =============================================================
# v26.5 Lag SHAP 확장본
# - v26.4 기존 구조/GEI/Global XAI/SHAP Dependence/LIME/Dual ALE 유지
# - Temporal SHAP 아래 Lag SHAP 신규 추가
# - Lag Importance Ranking
# - Peak Lag Detection
# - Cumulative Lag SHAP
# - Lag SHAP 95% Bootstrap Confidence Interval
# - Lag SHAP × Feature Heatmap (Mean |SHAP| / Mean SHAP 전환)
# - Python 3.9 호환
# =============================================================

# =============================================================
# v26.4 SHAP Dependence + LIME + Dual Centered ALE 확장본
# - v26.3 기존 구조/GEI/XAI/Bootstrap/Threshold/Counterfactual 유지
# - SHAP Summary 옆 SHAP Dependence Plot + 정량결과 + 자동해석 추가
# - ICE 샘플 수 최대 18개
# - ICE+PDP 옆 LIME 분석 추가
# - Centered ALE 단독 분석(왼쪽) + ALE Bootstrap/Threshold(오른쪽)
# - Python 3.9 호환
# =============================================================

# =============================================================
# v26.3 Centered ALE 신뢰성·임계점·목표제어 확장본
# - v26.2 Global XAI(SHAP/FI/Permutation) 및 기존 GEI/XAI 구조 유지
# - Centered ALE → 1D Bootstrap 95% CI
# - ALE 기반 Threshold Detection(감소 후보 임계점 자동 탐지)
# - Counterfactual Target Control(선택 Feature 1개 목표값 제어 시뮬레이션)
# - Python 3.9 호환
# =============================================================

# =============================================================
# v26.2 Global XAI 3종 중요도 통합본
# - SHAP Summary + Model Feature Importance + Permutation Importance
# - 3개 그래프 가로 배치
# - 각 방법별 정량표, Top Feature, Mean Importance, 자동 설명
# - SHAP/FI/Permutation 중요도 순위 비교 및 합의순위 제공
# - 기존 v26.1 코드 구조/GEI/XAI 기능 유지
# =============================================================

# =============================================================
# v26.1 컬럼매핑 + DIF/ADT/GDD/VPD 파생변수 추가본
# - 수확량 데이터 컬럼 선택: 개화수, 평균과중 추가(None 허용)
# - ADT: 일별 24시간 평균온도(10분 간격이면 최대 144개 관측값 기반)
# - DIF: 일별 주간(08~18시) 평균온도 - 야간(19~07시) 평균온도
# - GDD: Base 10℃, 일별 max(ADT-10, 0)을 선택 윈도우에서 누적
# - VPD: 10분 센서 온도·상대습도로 계산 후 일평균 → 선택 윈도우 평균
# - 기존 코드 구조와 GEI/XAI 기능 유지
# =============================================================

# =============================================================
# v25.2 일사량 GEI 유효광시간 분모 개선본
# - L0(0~1 W/m² 미만) 야간·무일사 누적시간은 그래프/데이터에 유지
# - 일사량 GEI 분자/분모에서 L0 제외
# - L1~L6 실제 유효광시간 합계를 일사량 GEI 분모시간으로 사용
# - 1~7주(7~49일) 변경 시 각 조사일 기준 유효광시간 자동 재계산
# - 온도·습도·CO₂ 기존 GEI 계산방식 유지
# =============================================================

# =============================================================
# 1~4주 수확수 이동평균 + 최고 R² 자동 탐색 기능 추가본
# - 사이드바 슬라이더로 1~4주 평균기간 선택
# - 1~4주 고정 이동평균 시계열 생성
# - 동일 데이터/동일 분할/동일 모델로 R² 공정 비교
# - 최고 R² 기간 추천 및 적용 버튼 제공
# - 수확수/1~4주평균수확수 선택 시에만 Y-window 및 Joint X–Y optimizer 표시
# - 표시 순서: Y-window optimizer → Joint X–Y 28개 조합 optimizer
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.naive_bayes import GaussianNB
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from itertools import permutations
from plotly.subplots import make_subplots
import matplotlib
import platform
import re
import gc
import sqlite3
import json
import os
from io import StringIO
from pathlib import Path

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

# -------------------------------------------------------------
# 기본 설정
# -------------------------------------------------------------
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rc('axes', unicode_minus=False)

st.set_page_config(layout="wide")

st.sidebar.markdown("## 🎨 대시보드 테마 설정")

bg_theme = st.sidebar.selectbox(
    "배경 테마 선택",
    [
        "Light Blue",
        "Dark Navy",
        "Mint",
        "Lavender",
        "White",
        "Warm Cream",
        "Greenhouse",
        "Soft Gray",
        "Peach",
        "Sky"
    ],
    index=0
)

theme_map = {
    "Light Blue": "linear-gradient(135deg,#f4f7fb 0%,#eef4ff 40%,#f8fbff 100%)",
    "Dark Navy": "linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%)",
    "Mint": "linear-gradient(135deg,#ecfeff 0%,#d1fae5 50%,#f0fdfa 100%)",
    "Lavender": "linear-gradient(135deg,#f5f3ff 0%,#ede9fe 50%,#faf5ff 100%)",
    "White": "#ffffff",
    "Warm Cream": "linear-gradient(135deg,#fff7ed 0%,#fffbeb 50%,#fef3c7 100%)",
    "Greenhouse": "linear-gradient(135deg,#ecfdf5 0%,#dcfce7 45%,#f0fdf4 100%)",
    "Soft Gray": "linear-gradient(135deg,#f8fafc 0%,#e2e8f0 50%,#f1f5f9 100%)",
    "Peach": "linear-gradient(135deg,#fff1f2 0%,#ffe4e6 50%,#fff7ed 100%)",
    "Sky": "linear-gradient(135deg,#eff6ff 0%,#dbeafe 50%,#e0f2fe 100%)"
}

selected_bg = theme_map[bg_theme]

# -------------------------------------------------------------
# 디자인 설정: 그래프 / 표 / Heatmap / 글자 크기
# -------------------------------------------------------------
st.sidebar.markdown("## 🖌️ 그래프·표 디자인 설정")

graph_theme = st.sidebar.selectbox(
    "그래프 스타일 선택",
    ["기본", "논문(Paper)", "발표(Presentation)", "다크모드", "스마트팜", "컬러풀"],
    index=0
)

table_theme = st.sidebar.selectbox(
    "표 스타일 선택",
    ["기본", "심플", "논문", "대시보드", "카드형"],
    index=3
)

heatmap_cmap = st.sidebar.selectbox(
    "Heatmap 색상 선택",
    ["YlOrRd", "RdYlGn", "Blues", "viridis", "turbo", "coolwarm", "Greens"],
    index=5
)

plotly_template = st.sidebar.selectbox(
    "Plotly 그래프 테마 선택",
    ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "presentation"],
    index=1
)

font_scale = st.sidebar.slider(
    "그래프/표 글자 크기 배율",
    min_value=0.8,
    max_value=1.6,
    value=1.0,
    step=0.1
)

line_width_scale = st.sidebar.slider(
    "그래프 선 굵기 배율",
    min_value=0.8,
    max_value=2.5,
    value=1.2,
    step=0.1
)


st.sidebar.markdown("## 🍅 수확수 예측 안정화 기능")

harvest_avg_weeks = st.sidebar.slider(
    "수확수 이동평균 기간 선택(주)",
    min_value=1,
    max_value=4,
    value=4,
    step=1,
    key="harvest_avg_weeks_slider",
    help=(
        "1주는 원자료와 동일하며, 2~4주는 해당 개수의 연속된 수확수 자료를 "
        "한 행씩 이동하면서 평균합니다."
    ),
)
selected_harvest_target = f"{harvest_avg_weeks}주평균수확수"

st.sidebar.caption(
    f"현재 선택: {harvest_avg_weeks}주 이동평균 · "
    "1~4주 R²를 동일 조건으로 비교해 최적 기간을 자동 추천합니다."
)


# 표 디자인별 CSS 값
_table_style_map = {
    "기본": {
        "header_bg": "#f1f5f9", "header_color": "#0f172a", "border": "#dbe7ff",
        "shadow": "0 3px 10px rgba(0,0,0,0.05)", "radius": "14px", "font": "12px"
    },
    "심플": {
        "header_bg": "#ffffff", "header_color": "#111827", "border": "#e5e7eb",
        "shadow": "none", "radius": "6px", "font": "12px"
    },
    "논문": {
        "header_bg": "#f8fafc", "header_color": "#000000", "border": "#111827",
        "shadow": "none", "radius": "0px", "font": "11px"
    },
    "대시보드": {
        "header_bg": "linear-gradient(135deg,#dbeafe,#eff6ff)", "header_color": "#1e3a8a", "border": "#bfdbfe",
        "shadow": "0 6px 18px rgba(37,99,235,0.12)", "radius": "16px", "font": "12px"
    },
    "카드형": {
        "header_bg": "linear-gradient(135deg,#ecfdf5,#f0fdfa)", "header_color": "#065f46", "border": "#bbf7d0",
        "shadow": "0 8px 24px rgba(15,118,110,0.12)", "radius": "18px", "font": "13px"
    },
}
_table_css = _table_style_map.get(table_theme, _table_style_map["대시보드"])







st.markdown(
    f"""
    <style>
    /* 선택형 표 디자인 */
    div[data-testid="stDataFrame"] {{
        border-radius: {_table_css['radius']} !important;
        overflow: hidden !important;
        border: 1px solid {_table_css['border']} !important;
        box-shadow: {_table_css['shadow']} !important;
        font-size: calc({_table_css['font']} * {font_scale}) !important;
    }}
    div[data-testid="stDataFrame"] * {{
        font-size: calc({_table_css['font']} * {font_scale}) !important;
    }}
    table {{
        font-size: calc({_table_css['font']} * {font_scale}) !important;
        border-collapse: collapse !important;
    }}
    thead tr th, .xai-table th {{
        background: {_table_css['header_bg']} !important;
        color: {_table_css['header_color']} !important;
        font-weight: 800 !important;
    }}
    .xai-table {{
        border: 1px solid {_table_css['border']} !important;
        border-radius: {_table_css['radius']} !important;
        box-shadow: {_table_css['shadow']} !important;
    }}
    .xai-card, .xai-subcard, .pretty-box {{
        border-radius: {_table_css['radius']} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <style>

    /* 전체 배경 */
    .stApp {{
        background: {selected_bg};
        transition: all 0.3s ease-in-out;
    }}

    /* 글자 색상 */
    h1,h2,h3,h4,p,span,label {{
        color: {"#ffffff" if bg_theme == "Dark Navy" else "#183b56"} !important;
    }}

    /* 메인 영역 */
    .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 0.5rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }}

    /* 카드 */
    .pretty-box {{
        background: rgba(255,255,255,0.82);
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.5);
        backdrop-filter: blur(8px);
        margin-bottom: 14px;
    }}

    /* dataframe */
    div[data-testid="stDataFrame"] {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #dbe7ff;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }}

    /* metric */
    [data-testid="metric-container"] {{
        background: linear-gradient(135deg,#ffffff,#f5f9ff);
        border: 1px solid #d9e8ff;
        padding: 12px;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }}

    /* 버튼 */
    .stButton > button {{
        border-radius: 12px;
        background: linear-gradient(135deg,#3b82f6,#2563eb);
        color: white;
        border: none;
        font-weight: 600;
    }}

    /* sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg,#183b56,#1e4f73);
    }}

    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    '''
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    div[data-testid="stDataFrame"] {
        font-size: 12px;
    }

    table {
        font-size: 12px !important;
    }

    .element-container {
        margin-bottom: 0.3rem;
    }
    </style>
    ''',
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    .xai-card {
        background: rgba(255,255,255,0.88);
        border-radius: 20px;
        padding: 20px 22px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.9);
        margin: 10px 0 18px 0;
    }
    .xai-hero {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 55%, #7c3aed 100%);
        border-radius: 22px;
        padding: 24px;
        color: white !important;
        box-shadow: 0 14px 34px rgba(37,99,235,0.25);
        margin-bottom: 18px;
    }
    .xai-hero * { color: white !important; }
    .xai-pill-wrap {
        display:flex; gap:12px; flex-wrap:wrap; margin-top:16px;
    }
    .xai-pill {
        flex:1; min-width:180px;
        background: rgba(255,255,255,0.16);
        border:1px solid rgba(255,255,255,0.28);
        border-radius:16px; padding:14px;
        backdrop-filter: blur(8px);
    }
    .xai-pill .label { font-size:13px; opacity:0.9; }
    .xai-pill .value { font-size:25px; font-weight:900; margin-top:4px; }
    .xai-subcard {
        background: linear-gradient(135deg,#ffffff,#f8fbff);
        border-radius:16px;
        padding:16px;
        border:1px solid #e2e8f0;
        box-shadow:0 5px 16px rgba(15,23,42,0.06);
        line-height:1.65;
        font-size:15px;
    }
    .xai-note {
        border-left: 5px solid #2563eb;
        background: #eff6ff;
        padding: 12px 14px;
        border-radius: 12px;
        line-height: 1.65;
        margin: 10px 0;
    }
    .xai-table {
        width:100%;
        border-collapse:collapse;
        font-size:14px;
        background:white;
        border-radius:14px;
        overflow:hidden;
    }
    .xai-table th {
        background:#f1f5f9;
        color:#0f172a !important;
        text-align:left;
        padding:10px;
        border-bottom:1px solid #cbd5e1;
    }
    .xai-table td {
        padding:10px;
        border-bottom:1px solid #e5e7eb;
        vertical-align:top;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# -------------------------------------------------------------
# 프리미엄 분석 섹션 디자인
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    .xai-section-banner {
        position: relative;
        overflow: hidden;
        margin: 18px 0 14px 0;
        padding: 18px 20px;
        border-radius: 20px;
        background:
            radial-gradient(circle at 92% 10%, rgba(255,255,255,0.28), transparent 32%),
            linear-gradient(135deg, #0f766e 0%, #2563eb 58%, #7c3aed 100%);
        box-shadow: 0 14px 32px rgba(37, 99, 235, 0.20);
        border: 1px solid rgba(255,255,255,0.25);
    }
    .xai-section-banner * {
        color: #ffffff !important;
    }
    .xai-section-kicker {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 10px;
        margin-bottom: 8px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.24);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .xai-section-title {
        font-size: 23px;
        line-height: 1.25;
        font-weight: 900;
        letter-spacing: -0.03em;
    }
    .xai-section-subtitle {
        margin-top: 7px;
        font-size: 14px;
        line-height: 1.65;
        opacity: 0.93;
    }
    .xai-panel-label {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 2px 0 9px 0;
        font-size: 14px;
        font-weight: 900;
        color: #17324d !important;
        letter-spacing: -0.01em;
    }
    .xai-panel-label::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: linear-gradient(135deg, #14b8a6, #2563eb);
        box-shadow: 0 0 0 5px rgba(37,99,235,0.10);
    }
    .xai-insight-card {
        margin: 12px 0;
        padding: 15px 17px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,246,255,0.94));
        border: 1px solid rgba(191,219,254,0.95);
        box-shadow: 0 8px 22px rgba(15,23,42,0.07);
        line-height: 1.75;
        color: #17324d !important;
    }
    .xai-insight-card b {
        color: #1d4ed8 !important;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1.15rem;
    }
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.94);
        box-shadow: 0 10px 26px rgba(15,23,42,0.08) !important;
    }
    [data-testid="stMetric"] {
        min-height: 112px;
    }
    [data-testid="stMetricValue"] {
        font-weight: 900;
        letter-spacing: -0.03em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_stylish_section(title, subtitle, kicker="XAI ANALYTICS"):
    """그래프와 결과표 섹션의 공통 프리미엄 헤더."""
    st.markdown(
        f"""
        <div class="xai-section-banner">
            <div class="xai-section-kicker">{kicker}</div>
            <div class="xai-section-title">{title}</div>
            <div class="xai-section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_label(text):
    st.markdown(
        f'<div class="xai-panel-label">{text}</div>',
        unsafe_allow_html=True,
    )

st.title("🍅 설명가능 AI 기반 토마토 생육(수확) 분석 통합 대시보드")

# -------------------------------------------------------------
# 전역 안전 기본값: 자동 패치 블록의 런타임 NameError 방지
# -------------------------------------------------------------
target_col = None
report_target = None
metrics = {}
df = pd.DataFrame()
features = []
model_choice = None


# -------------------------------------------------------------
# 공통 유틸
# -------------------------------------------------------------
def safe_predict(model, X_input, feature_names):
    """
    모델 학습에 사용한 feature_names만 강제로 맞춰 예측합니다.
    입력 데이터에 4주평균수확수, 누적수확수, 누적착과수, 착과잔량(Fruit Load) 등
    학습 당시 없던 컬럼이 섞여 있어도 자동으로 제거합니다.
    """
    feature_names = list(feature_names)
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    if isinstance(X_input, pd.Series):
        X_input = pd.DataFrame([X_input])
    elif isinstance(X_input, np.ndarray):
        arr = np.asarray(X_input)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n = min(arr.shape[1], len(feature_names))
        X_input = pd.DataFrame(arr[:, :n], columns=feature_names[:n])
    elif not isinstance(X_input, pd.DataFrame):
        raise TypeError("X_input은 Series, ndarray, DataFrame 중 하나여야 합니다.")

    X_input = X_input.copy()
    X_input = X_input.reindex(columns=feature_names)
    X_input = X_input.apply(pd.to_numeric, errors="coerce")
    X_input = X_input.fillna(X_input.mean(numeric_only=True)).fillna(0)
    return model.predict(X_input)

def make_model(model_choice: str):
    """회귀 예측용 비교 모델 생성."""
    if model_choice == "RandomForest":
        return RandomForestRegressor(random_state=42)
    if model_choice == "GradientBoosting":
        return GradientBoostingRegressor(random_state=42)
    if model_choice == "XGBoost":
        return XGBRegressor(random_state=42, objective="reg:squarederror")
    if model_choice == "LGBM":
        return LGBMRegressor(random_state=42)
    if model_choice in ["ANN", "ANN(인공신경망)"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )),
        ])
    if model_choice in ["BPM", "BPM(베이지안 확률 모델)"]:
        # 연속형 생육·수확량 예측을 위한 Bayesian probabilistic regression
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", BayesianRidge()),
        ])
    if model_choice in ["SVM", "SVM(서포트벡터머신)"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
        ])
    if model_choice == "GaussianNB":
        # 하위 버전 호환용. GaussianNB는 분류기이므로 연속형 회귀 비교에는 권장하지 않습니다.
        return GaussianNB()
    raise ValueError("지원하지 않는 모델")









def get_model_feature_names(model, fallback_features):
    """모델이 fit에 사용한 feature명을 우선 사용합니다."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback_features)


def align_xai_input(X_input, features, model=None):
    """
    SHAP/ICE/PDP/ALE 입력을 모델 학습 Feature만 포함하도록 정렬합니다.
    파생 컬럼(4주평균수확수, 누적수확수, 누적착과수, 착과잔량 등)은 자동 제거됩니다.
    """
    feature_names = get_model_feature_names(model, features) if model is not None else list(features)

    if isinstance(X_input, pd.Series):
        out = pd.DataFrame([X_input])
    elif isinstance(X_input, pd.DataFrame):
        out = X_input.copy()
    else:
        arr = np.asarray(X_input)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n = min(arr.shape[1], len(feature_names))
        out = pd.DataFrame(arr[:, :n], columns=feature_names[:n])

    out = out.reindex(columns=feature_names)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.fillna(out.mean(numeric_only=True)).fillna(0)
    return out


# align_shap_sample_table 함수 삭제됨

def get_xai_features(features, model=None):
    """XAI 분석에 사용할 최종 feature list."""
    return get_model_feature_names(model, features) if model is not None else list(features)


def remove_derived_columns_for_xai(df_input, features, model=None):
    """프론트엔드/XAI 출력 직전에도 파생 컬럼을 제거해 feature mismatch를 방지합니다."""
    return align_xai_input(df_input, features, model)


def compute_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }



def add_harvest_enhancement_features(df, date_col="조사일자"):
    """
    매핑 데이터에 생육/수확 데이터 기준 파생 컬럼을 추가합니다.

    수확수 이동평균:
    - 1주평균수확수: 현재 수확수 1개 값(원자료와 동일)
    - 2주평균수확수: 1~2, 2~3, 3~4 ... 고정 2개 이동평균
    - 3주평균수확수: 1~3, 2~4, 3~5 ... 고정 3개 이동평균
    - 4주평균수확수: 1~4, 2~5, 3~6 ... 고정 4개 이동평균
    - 각 평균은 해당 기간만큼 관측값이 모인 시점부터 계산합니다.

    추가 파생변수:
    - 누적착과수 = 착과수 누적합
    - 누적수확수 = 수확수 누적합
    - 착과잔량(Fruit Load) = 누적착과수 - 누적수확수
    """
    df = df.copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    harvest_available = False
    if "수확수" in df.columns:
        harvest_series = pd.to_numeric(
            df["수확수"],
            errors="coerce",
        )
        df["수확수"] = harvest_series
        harvest_available = harvest_series.notna().any()

        for window in range(1, 5):
            avg_col = f"{window}주평균수확수"
            if harvest_available:
                # min_periods=window로 설정하여 불완전한 초기 부분평균은 만들지 않습니다.
                df[avg_col] = (
                    harvest_series
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            else:
                df[avg_col] = np.nan

        df["누적수확수"] = (
            harvest_series.fillna(0).cumsum()
            if harvest_available
            else np.nan
        )
    else:
        for window in range(1, 5):
            avg_col = f"{window}주평균수확수"
            if avg_col not in df.columns:
                df[avg_col] = np.nan

        if "누적수확수" not in df.columns:
            df["누적수확수"] = np.nan

    fruit_set_available = False
    if "착과수" in df.columns:
        fruit_set_series = pd.to_numeric(
            df["착과수"],
            errors="coerce",
        )
        df["착과수"] = fruit_set_series
        fruit_set_available = fruit_set_series.notna().any()
        df["누적착과수"] = (
            fruit_set_series.fillna(0).cumsum()
            if fruit_set_available
            else np.nan
        )
    else:
        if "누적착과수" not in df.columns:
            df["누적착과수"] = np.nan

    if harvest_available and fruit_set_available:
        df["착과잔량(Fruit Load)"] = (
            df["누적착과수"] - df["누적수확수"]
        )
    else:
        df["착과잔량(Fruit Load)"] = np.nan
    return df

def get_harvest_boost_extra_features(df):
    """파생 수확 변수는 모델/XAI Feature로 사용하지 않습니다."""
    return []


def is_harvest_target(target_col):
    target_name = str(target_col)
    return (
        target_name in ["수확수", "착과수", "착과잔량(Fruit Load)"]
        or re.fullmatch(r"[1-4]주평균수확수", target_name) is not None
    )


def is_harvest_window_optimizer_target(target_col):
    """Y-window 및 X×Y 28개 조합 최적화를 표시할 수확수 계열 대상인지 판별합니다."""
    target_name = str(target_col).strip()
    return (
        target_name == "수확수"
        or re.fullmatch(r"[1-4]주평균수확수", target_name) is not None
    )


def build_harvest_comparison_report(model_choice, base_features, df, target_cols):
    """
    수확수와 지정한 1~4주 평균수확수를 동일 feature set으로 비교 평가.
    반환: 비교 성능표 DataFrame
    """
    rows = []
    for tcol in target_cols:
        if tcol not in df.columns:
            continue
        temp = df.copy()
        features = [c for c in base_features if c in temp.columns and c != tcol]
        features = list(dict.fromkeys(features + get_harvest_boost_extra_features(temp)))
        features = [c for c in features if c in temp.columns and c != tcol]
        model_df = temp[features + [tcol]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(model_df) < 8 or len(features) == 0:
            continue
        X_cmp = model_df[features]
        y_cmp = model_df[tcol]
        X_tr, X_te, y_tr, y_te = train_test_split(X_cmp, y_cmp, test_size=0.3, random_state=42)
        try:
            m = make_model(model_choice)
            m.fit(X_tr, y_tr)
            pred = safe_predict(m, X_te, features)
            met = compute_metrics(y_te, pred)
            rows.append({
                "예측대상": tcol,
                "MSE": met["MSE"],
                "MAE": met["MAE"],
                "R2": met["R2"],
                "사용변수수": len(features),
                "데이터수": len(model_df)
            })
        except Exception as e:
            rows.append({
                "예측대상": tcol,
                "MSE": np.nan,
                "MAE": np.nan,
                "R2": np.nan,
                "사용변수수": len(features),
                "데이터수": len(model_df),
                "오류": str(e)
            })
    return pd.DataFrame(rows)




def evaluate_harvest_average_windows(
    model_choice,
    base_features,
    df,
    windows=(1, 2, 3, 4),
    test_size=0.2,
):
    """
    1~4주 수확수 이동평균의 예측성능을 동일 조건으로 비교합니다.

    공정 비교 원칙:
    - 동일한 환경 Feature 사용
    - 네 이동평균이 모두 존재하는 공통 유효행 사용
    - 동일한 train/test 행 분할과 random_state 사용
    - 사용자가 선택한 동일 모델 사용
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    target_cols = [
        f"{int(window)}주평균수확수"
        for window in windows
        if f"{int(window)}주평균수확수" in df.columns
    ]

    features = [
        col for col in list(base_features)
        if col in df.columns and col not in target_cols
    ]
    features = list(dict.fromkeys(features))

    if not features or len(target_cols) == 0:
        return pd.DataFrame()

    required_cols = features + target_cols
    common_df = (
        df[required_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    # R² 계산에는 테스트 표본이 최소 2개 필요합니다.
    if len(common_df) < 6:
        return pd.DataFrame()

    row_indices = np.arange(len(common_df))
    train_idx, test_idx = train_test_split(
        row_indices,
        test_size=test_size,
        random_state=42,
    )

    if len(test_idx) < 2 or len(train_idx) < 2:
        return pd.DataFrame()

    X_all = common_df[features]
    rows = []

    for target_name in target_cols:
        window = int(str(target_name).split("주", 1)[0])
        y_all = common_df[target_name]

        try:
            model = make_model(model_choice)
            model.fit(
                X_all.iloc[train_idx],
                y_all.iloc[train_idx],
            )
            prediction = safe_predict(
                model,
                X_all.iloc[test_idx],
                features,
            )
            metric = compute_metrics(
                y_all.iloc[test_idx],
                prediction,
            )

            rows.append({
                "평균기간(주)": window,
                "예측대상": target_name,
                "MSE": float(metric["MSE"]),
                "MAE": float(metric["MAE"]),
                "R2": float(metric["R2"]),
                "공통 데이터수": int(len(common_df)),
                "학습 데이터수": int(len(train_idx)),
                "평가 데이터수": int(len(test_idx)),
                "오류": "",
            })
        except Exception as exc:
            rows.append({
                "평균기간(주)": window,
                "예측대상": target_name,
                "MSE": np.nan,
                "MAE": np.nan,
                "R2": np.nan,
                "공통 데이터수": int(len(common_df)),
                "학습 데이터수": int(len(train_idx)),
                "평가 데이터수": int(len(test_idx)),
                "오류": str(exc),
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("평균기간(주)")
        .reset_index(drop=True)
    )


def apply_best_harvest_window(best_window):
    """최고 R² 기간을 사이드바 슬라이더와 예측대상에 반영합니다."""
    best_window = int(best_window)
    st.session_state["harvest_avg_weeks_slider"] = best_window
    st.session_state["_last_harvest_avg_weeks"] = None
    st.session_state["target_col_select"] = f"{best_window}주평균수확수"


def render_harvest_average_window_optimizer(
    df,
    features,
    model_choice,
    selected_window,
    selected_x_window=None,
):
    """1~4주 평균 R² 비교표·그래프·최고기간 추천을 가로형으로 출력합니다."""
    selected_x_text = (
        f"{int(selected_x_window)}주"
        if selected_x_window is not None
        else "현재 선택 기간"
    )

    render_stylish_section(
        "🏆 수확수 목표변수 평균기간 최적화 (Y-window, 1~4주)",
        (
            f"환경 입력기간(X-window)은 {selected_x_text}로 고정하고, "
            "예측대상 수확수의 평균기간(Y-window)만 1~4주로 바꾸어 성능을 비교합니다."
        ),
        kicker="Y-WINDOW OPTIMIZER",
    )

    with st.expander(
        "📖 Y-window 최적화 상세내용 열기/닫기",
        expanded=False,
    ):
        st.markdown(
            f"""
            <div class="xai-insight-card">
                <b>이 표에서 바뀌는 것은 수확수 평균기간(Y)입니다.</b><br>
                환경 입력기간(X)은 <b>{selected_x_text}</b>로 동일하게 고정됩니다.<br><br>
                예를 들어 <b>3주</b> 행은 “{selected_x_text} 환경정보로
                3주평균수확수를 예측한 결과”입니다.
                따라서 이 표는 <b>수확수를 몇 주 평균해야 가장 안정적으로 예측되는가?</b>를 찾습니다.<br><br>
                <b>평가 원칙</b><br>
                • 1~4주 Y-window가 모두 존재하는 공통 조사일만 사용<br>
                • 동일 환경 Feature 사용<br>
                • 동일 학습/평가 분할 사용<br>
                • R²는 높을수록, MSE·MAE는 낮을수록 우수<br>
                • 최종 추천은 최고 R²를 우선으로 표시
            </div>
            """,
            unsafe_allow_html=True,
        )

    comparison_df = evaluate_harvest_average_windows(
        model_choice=model_choice,
        base_features=features,
        df=df,
        windows=(1, 2, 3, 4),
        test_size=0.2,
    )

    if comparison_df.empty:
        st.warning(
            "1~4주 R² 비교에 필요한 공통 유효 데이터가 부족합니다. "
            "4주평균까지 계산한 뒤 최소 6개 이상의 공통 행이 필요합니다."
        )
        return comparison_df, None

    valid_df = comparison_df.dropna(subset=["R2"]).copy()
    if valid_df.empty:
        st.warning(
            "선택한 모델에서 1~4주 평균의 R²를 계산하지 못했습니다. "
            "회귀모델(RandomForest, GradientBoosting, XGBoost, LGBM)을 선택하세요."
        )
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )
        return comparison_df, None

    best_row = valid_df.loc[valid_df["R2"].idxmax()]
    best_window = int(best_row["평균기간(주)"])
    best_r2 = float(best_row["R2"])

    selected_rows = valid_df[
        valid_df["평균기간(주)"] == int(selected_window)
    ]
    selected_r2 = (
        float(selected_rows.iloc[0]["R2"])
        if not selected_rows.empty
        else np.nan
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("현재 선택 평균기간", f"{int(selected_window)}주")
    with metric_col2:
        st.metric(
            "현재 선택 R²",
            f"{selected_r2:.4f}" if np.isfinite(selected_r2) else "계산 불가",
        )
    with metric_col3:
        delta_text = (
            f"{best_r2 - selected_r2:+.4f}"
            if np.isfinite(selected_r2)
            else None
        )
        st.metric(
            "최고 R² 평균기간",
            f"{best_window}주 · {best_r2:.4f}",
            delta=delta_text,
        )

    graph_col, table_col = st.columns([1.14, 0.86], gap="large")

    with graph_col:
        render_panel_label("1~4주 이동평균 R² 비교 그래프")

        chart_df = valid_df.sort_values("평균기간(주)").copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_df["평균기간(주)"],
                y=chart_df["R2"],
                mode="lines+markers+text",
                text=[f"{v:.3f}" for v in chart_df["R2"]],
                textposition="top center",
                line=dict(width=4, color="#2563eb", shape="spline"),
                marker=dict(
                    size=12,
                    color="#ffffff",
                    line=dict(width=3, color="#2563eb"),
                ),
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.10)",
                name="R²",
                hovertemplate="%{x}주 평균<br>R²=%{y:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[best_window],
                y=[best_r2],
                mode="markers",
                marker=dict(
                    size=20,
                    symbol="star",
                    color="#f59e0b",
                    line=dict(width=2, color="#ffffff"),
                ),
                name=f"최고 {best_window}주",
                hovertemplate=f"최고 성능<br>{best_window}주 평균<br>R²={best_r2:.4f}<extra></extra>",
            )
        )
        fig.add_vline(
            x=int(selected_window),
            line_width=2,
            line_dash="dot",
            line_color="#0f766e",
            annotation_text=f"현재 {int(selected_window)}주",
            annotation_position="top left",
        )
        fig.update_layout(
            height=370,
            title=dict(
                text=f"수확수 이동평균 기간별 R² · {model_choice}",
                x=0.02,
                xanchor="left",
            ),
            xaxis=dict(
                title="이동평균 기간",
                tickmode="array",
                tickvals=[1, 2, 3, 4],
                ticktext=["1주", "2주", "3주", "4주"],
                showgrid=False,
            ),
            yaxis=dict(
                title="R² (결정계수)",
                zeroline=True,
                zerolinecolor="rgba(100,116,139,0.35)",
                gridcolor="rgba(148,163,184,0.22)",
            ),
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", y=1.13, x=1, xanchor="right"),
            margin=dict(l=50, r=18, t=78, b=48),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.78)",
        )
        display_plotly(fig)

    with table_col:
        render_panel_label(
            f"Y-window별 모델 성능 결과표 · X-window {selected_x_text} 고정"
        )
        display_cols = [
            "평균기간(주)",
            "MSE",
            "MAE",
            "R2",
            "공통 데이터수",
            "평가 데이터수",
        ]
        display_df = comparison_df[display_cols].copy()
        display_df["평균기간(주)"] = (
            display_df["평균기간(주)"]
            .astype(int)
            .astype(str)
            .add("주")
        )
        display_df = display_df.rename(
            columns={
                "평균기간(주)": "평균기간",
                "공통 데이터수": "공통자료",
                "평가 데이터수": "평가자료",
            }
        )
        st.dataframe(
            display_df.round(4),
            use_container_width=True,
            hide_index=True,
            height=370,
        )
        st.markdown(
            f"""
            <div class="xai-insight-card" style="margin-top:10px;">
                <b>표 읽는 방법</b><br>
                • 평균기간: 수확수 목표변수(Y)를 몇 주 평균했는지 표시합니다.<br>
                • R²: 높을수록 해당 Y-window의 설명력이 좋습니다.<br>
                • MSE·MAE: 낮을수록 예측오차가 작습니다.<br>
                • 공통자료: 1~4주 평균을 공정하게 비교하기 위해 함께 사용한 동일 표본 수입니다.<br><br>
                현재 표에서는 환경 입력기간(X)이 <b>{selected_x_text}</b>로 고정되어 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if best_window == int(selected_window):
        st.success(
            f"현재 선택한 {best_window}주 평균이 가장 높은 R²={best_r2:.4f}를 보였습니다."
        )
    else:
        improvement = (
            best_r2 - selected_r2
            if np.isfinite(selected_r2)
            else np.nan
        )
        improvement_text = (
            f" 현재 선택보다 R²가 {improvement:.4f} 높습니다."
            if np.isfinite(improvement)
            else ""
        )
        st.success(
            f"최고 성능은 {best_window}주 평균이며 R²={best_r2:.4f}입니다."
            f"{improvement_text}"
        )
        st.button(
            f"✅ 최고 R²인 {best_window}주 평균을 적용",
            key="apply_best_harvest_window_button",
            on_click=apply_best_harvest_window,
            args=(best_window,),
        )

    st.caption(
        "비교의 공정성을 위해 1~4주 평균값과 환경 Feature가 모두 존재하는 "
        "동일한 행, 동일한 학습·평가 분할을 사용했습니다."
    )
    return comparison_df, best_window


@st.cache_data(show_spinner=False)
def evaluate_xy_window_grid(
    week_dfs,
    model_choice,
    x_windows=tuple(range(1, 8)),
    y_windows=tuple(range(1, 5)),
    test_size=0.2,
):
    """
    환경 입력기간(X-window) 1~7주와 수확수 평균기간(Y-window) 1~4주의
    총 28개 조합을 동일 표본과 동일 분할로 비교합니다.

    공정 비교 원칙:
    - 28개 조합에서 공통으로 사용할 수 있는 조사일자만 사용
    - 동일한 train/test 행 인덱스와 random_state=42 사용
    - 각 X-window에는 해당 기간의 환경 Feature만 사용
    - 각 Y-window에는 해당 기간의 평균수확수만 예측대상으로 사용
    """
    if not isinstance(week_dfs, dict) or not week_dfs:
        return pd.DataFrame()

    prepared = {}
    target_cols = [f"{int(yw)}주평균수확수" for yw in y_windows]
    master_df = None
    feature_map = {}

    for xw in x_windows:
        if int(xw) not in week_dfs:
            continue

        wk_df = add_harvest_enhancement_features(
            week_dfs[int(xw)].copy()
        )

        if "조사일자" not in wk_df.columns:
            continue

        wk_df["조사일자"] = pd.to_datetime(
            wk_df["조사일자"],
            errors="coerce",
        )
        wk_features = get_environment_feature_columns(wk_df)
        wk_features = [
            col for col in wk_features
            if col in wk_df.columns
        ]

        if not wk_features:
            continue

        feature_map[int(xw)] = wk_features

        feature_part = wk_df[
            ["조사일자"] + wk_features
        ].copy()

        if master_df is None:
            available_targets = [
                col for col in target_cols
                if col in wk_df.columns
            ]
            if len(available_targets) != len(target_cols):
                return pd.DataFrame()

            target_part = wk_df[
                ["조사일자"] + target_cols
            ].copy()
            master_df = target_part.merge(
                feature_part,
                on="조사일자",
                how="inner",
            )
        else:
            master_df = master_df.merge(
                feature_part,
                on="조사일자",
                how="inner",
            )

    if master_df is None or len(feature_map) != len(tuple(x_windows)):
        return pd.DataFrame()

    numeric_cols = target_cols + [
        feature
        for xw in x_windows
        for feature in feature_map.get(int(xw), [])
    ]

    common_df = master_df[
        ["조사일자"] + numeric_cols
    ].copy()
    common_df[numeric_cols] = common_df[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    common_df = (
        common_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=numeric_cols)
        .sort_values("조사일자")
        .reset_index(drop=True)
    )

    if len(common_df) < 6:
        return pd.DataFrame()

    row_indices = np.arange(len(common_df))
    train_idx, test_idx = train_test_split(
        row_indices,
        test_size=test_size,
        random_state=42,
    )

    if len(train_idx) < 2 or len(test_idx) < 2:
        return pd.DataFrame()

    rows = []

    for xw in x_windows:
        xw = int(xw)
        wk_features = feature_map[xw]
        X_all = common_df[wk_features]

        for yw in y_windows:
            yw = int(yw)
            target_name = f"{yw}주평균수확수"
            y_all = common_df[target_name]

            try:
                model = make_model(model_choice)
                model.fit(
                    X_all.iloc[train_idx],
                    y_all.iloc[train_idx],
                )
                prediction = safe_predict(
                    model,
                    X_all.iloc[test_idx],
                    wk_features,
                )
                metric = compute_metrics(
                    y_all.iloc[test_idx],
                    prediction,
                )

                rows.append({
                    "환경 입력기간 X(주)": xw,
                    "수확수 평균기간 Y(주)": yw,
                    "예측대상": target_name,
                    "MSE": float(metric["MSE"]),
                    "MAE": float(metric["MAE"]),
                    "R2": float(metric["R2"]),
                    "Feature수": int(len(wk_features)),
                    "공통 데이터수": int(len(common_df)),
                    "학습 데이터수": int(len(train_idx)),
                    "평가 데이터수": int(len(test_idx)),
                    "오류": "",
                })
            except Exception as exc:
                rows.append({
                    "환경 입력기간 X(주)": xw,
                    "수확수 평균기간 Y(주)": yw,
                    "예측대상": target_name,
                    "MSE": np.nan,
                    "MAE": np.nan,
                    "R2": np.nan,
                    "Feature수": int(len(wk_features)),
                    "공통 데이터수": int(len(common_df)),
                    "학습 데이터수": int(len(train_idx)),
                    "평가 데이터수": int(len(test_idx)),
                    "오류": str(exc),
                })

    return pd.DataFrame(rows)


def apply_best_xy_windows(best_x_window, best_y_window):
    """28개 조합 중 최고 X/Y 기간을 두 슬라이더와 예측대상에 반영합니다."""
    best_x_window = int(best_x_window)
    best_y_window = int(best_y_window)

    st.session_state["weeks"] = best_x_window
    st.session_state["weeks_slider_1"] = best_x_window
    st.session_state["harvest_avg_weeks_slider"] = best_y_window
    st.session_state["_last_harvest_avg_weeks"] = None
    st.session_state["target_col_select"] = (
        f"{best_y_window}주평균수확수"
    )


def render_xy_window_joint_optimizer(
    week_dfs,
    model_choice,
    selected_x_window,
    selected_y_window,
):
    """
    X-window 1~7주 × Y-window 1~4주, 총 28개 조합을
    Heatmap과 R² 행렬표로 표시하고 최고 조합을 추천합니다.
    """
    render_stylish_section(
        "🧭 환경 X-window × 수확수 Y-window 28개 조합 동시 최적화",
        (
            "과거 환경을 몇 주 볼 것인지(X)와 수확수를 몇 주 평균할 것인지(Y)를 "
            "동시에 바꾸어 최종 최고 R² 조합을 탐색합니다."
        ),
        kicker="JOINT X–Y WINDOW OPTIMIZER",
    )

    st.markdown(
        """
        <div class="xai-insight-card">
            <b>가장 쉽게 이해하는 방법</b><br><br>
            <b>X-window = 과거 환경을 몇 주까지 볼 것인가?</b><br>
            예: 환경 7주는 조사일 기준 과거 7주 온도·습도·CO₂·일사량·수분부족분 중 선택한 변수를 모델 입력으로 사용합니다.<br><br>
            <b>Y-window = 수확수를 몇 주 평균할 것인가?</b><br>
            예: 3주평균수확수는 1~3번째, 2~4번째, 3~5번째 수확수를 순차적으로 평균한 값입니다.<br><br>
            아래 표의 <b>각 칸은 하나의 X–Y 조합에서 계산된 R²</b>입니다.
            따라서 4행 × 7열 = 총 28개 조합 중 가장 큰 값이
            환경기간과 수확수 평균기간을 동시에 고려한 최종 후보입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("X-window 1~7주 × Y-window 1~4주, 총 28개 조합을 평가하고 있습니다..."):
        result_df = evaluate_xy_window_grid(
            week_dfs=week_dfs,
            model_choice=model_choice,
            x_windows=tuple(range(1, 8)),
            y_windows=tuple(range(1, 5)),
            test_size=0.2,
        )

    if result_df.empty:
        st.warning(
            "28개 조합을 공정하게 비교할 공통 유효 데이터가 부족합니다. "
            "모든 1~7주 환경 Feature와 1~4주 평균수확수가 동시에 존재하는 "
            "조사일이 최소 6개 이상 필요합니다."
        )
        return result_df, None

    valid_df = result_df.dropna(subset=["R2"]).copy()
    if valid_df.empty:
        st.warning("28개 조합에서 유효한 R²가 계산되지 않았습니다.")
        return result_df, None

    best_row = valid_df.loc[valid_df["R2"].idxmax()]
    best_x = int(best_row["환경 입력기간 X(주)"])
    best_y = int(best_row["수확수 평균기간 Y(주)"])
    best_r2 = float(best_row["R2"])

    current_row = valid_df[
        (valid_df["환경 입력기간 X(주)"] == int(selected_x_window))
        & (valid_df["수확수 평균기간 Y(주)"] == int(selected_y_window))
    ]
    current_r2 = (
        float(current_row.iloc[0]["R2"])
        if not current_row.empty
        else np.nan
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    with metric_1:
        st.metric("현재 환경 X-window", f"{int(selected_x_window)}주")
    with metric_2:
        st.metric("현재 수확수 Y-window", f"{int(selected_y_window)}주")
    with metric_3:
        st.metric(
            "현재 조합 R²",
            f"{current_r2:.4f}" if np.isfinite(current_r2) else "계산 불가",
        )
    with metric_4:
        delta = (
            best_r2 - current_r2
            if np.isfinite(current_r2)
            else None
        )
        st.metric(
            "28개 중 최고 조합",
            f"X {best_x}주 · Y {best_y}주",
            delta=f"R² {best_r2:.4f}" if delta is None else f"+{delta:.4f}",
        )

    r2_matrix = (
        valid_df
        .pivot_table(
            index="수확수 평균기간 Y(주)",
            columns="환경 입력기간 X(주)",
            values="R2",
            aggfunc="mean",
        )
        .reindex(index=range(1, 5), columns=range(1, 8))
    )

    graph_col, table_col = st.columns([1.06, 0.94], gap="large")

    with graph_col:
        render_panel_label("28개 조합 R² Heatmap")
        z_values = r2_matrix.to_numpy(dtype=float)
        text_values = np.where(
            np.isfinite(z_values),
            np.vectorize(lambda v: f"{v:.3f}")(z_values),
            "",
        )

        fig_xy = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=[f"환경 {x}주" for x in r2_matrix.columns],
                y=[f"{y}주평균수확수" for y in r2_matrix.index],
                text=text_values,
                texttemplate="%{text}",
                textfont=dict(size=13),
                colorscale="RdYlGn",
                colorbar=dict(title="R²"),
                hovertemplate=(
                    "X-window: %{x}<br>"
                    "Y-window: %{y}<br>"
                    "R²=%{z:.4f}<extra></extra>"
                ),
                zmid=0,
            )
        )
        fig_xy.add_trace(
            go.Scatter(
                x=[f"환경 {best_x}주"],
                y=[f"{best_y}주평균수확수"],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=22,
                    color="#f59e0b",
                    line=dict(width=2, color="#ffffff"),
                ),
                name="최고 R²",
                hovertemplate=(
                    f"최고 조합<br>X={best_x}주<br>Y={best_y}주"
                    f"<br>R²={best_r2:.4f}<extra></extra>"
                ),
            )
        )
        fig_xy.update_layout(
            height=440,
            title=dict(
                text=f"X-window × Y-window R² 행렬 · {model_choice}",
                x=0.02,
                xanchor="left",
            ),
            xaxis_title="환경 입력기간 X-window",
            yaxis_title="수확수 평균기간 Y-window",
            margin=dict(l=80, r=20, t=70, b=55),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.78)",
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        )
        display_plotly(fig_xy)

    with table_col:
        render_panel_label("28개 조합 R² 결과표")
        display_matrix = r2_matrix.copy()
        display_matrix.index = [
            f"{int(y)}주평균수확수(Y)"
            for y in display_matrix.index
        ]
        display_matrix.columns = [
            f"환경 {int(x)}주(X)"
            for x in display_matrix.columns
        ]
        display_matrix.index.name = "수확수 평균기간"

        try:
            styled_matrix = (
                display_matrix.style
                .format("{:.4f}")
                .highlight_max(
                    axis=None,
                    props=(
                        "background: linear-gradient(135deg,#fef3c7,#fde68a);"
                        "color:#92400e;font-weight:900;"
                    ),
                )
                .background_gradient(
                    cmap="RdYlGn",
                    axis=None,
                )
            )
            st.dataframe(
                styled_matrix,
                use_container_width=True,
                height=440,
            )
        except Exception:
            st.dataframe(
                display_matrix.round(4),
                use_container_width=True,
                height=440,
            )

    st.markdown(
        f"""
        <div class="xai-insight-card">
            <b>이 28개 조합표를 읽는 순서</b><br><br>
            ① 왼쪽 행에서 수확수 평균기간(Y)을 고릅니다.<br>
            ② 위쪽 열에서 환경 입력기간(X)을 고릅니다.<br>
            ③ 행과 열이 만나는 칸의 R²를 확인합니다.<br><br>
            예를 들어 <b>3주평균수확수(Y) × 환경 7주(X)</b> 칸은
            “과거 7주 환경정보로 3주평균수확수를 예측한 R²”입니다.<br><br>
            현재 최고 조합은 <b>환경 {best_x}주(X) × {best_y}주평균수확수(Y)</b>이며,
            R²는 <b>{best_r2:.4f}</b>입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    detail_cols = [
        "환경 입력기간 X(주)",
        "수확수 평균기간 Y(주)",
        "MSE",
        "MAE",
        "R2",
        "공통 데이터수",
        "평가 데이터수",
    ]
    with st.expander("📋 28개 조합의 MSE·MAE·R² 상세 결과 보기"):
        detail_df = (
            result_df[detail_cols]
            .sort_values("R2", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            detail_df.round(4),
            use_container_width=True,
            hide_index=True,
        )

    if (
        best_x == int(selected_x_window)
        and best_y == int(selected_y_window)
    ):
        st.success(
            f"현재 선택한 X {best_x}주 × Y {best_y}주 조합이 "
            f"28개 중 가장 높은 R²={best_r2:.4f}입니다."
        )
    else:
        improvement = (
            best_r2 - current_r2
            if np.isfinite(current_r2)
            else np.nan
        )
        improvement_text = (
            f" 현재 조합보다 R²가 {improvement:.4f} 높습니다."
            if np.isfinite(improvement)
            else ""
        )
        st.success(
            f"최고 조합은 환경 X-window {best_x}주 × "
            f"수확수 Y-window {best_y}주이며 R²={best_r2:.4f}입니다."
            f"{improvement_text}"
        )
        st.button(
            f"✅ 최고 조합 X {best_x}주 · Y {best_y}주 적용",
            key="apply_best_xy_window_button",
            on_click=apply_best_xy_windows,
            args=(best_x, best_y),
        )

    st.caption(
        "28개 조합은 모든 기간에서 공통으로 존재하는 동일 조사일, "
        "동일 학습·평가 분할, 동일 모델을 사용해 비교했습니다."
    )
    return result_df, (best_x, best_y)

def show_harvest_r2_recommendation(target_col, metrics, df):
    """원자료 수확수 R²가 낮을 때 1~4주 평균 비교를 추천합니다."""
    try:
        average_cols = [
            f"{window}주평균수확수"
            for window in range(1, 5)
            if f"{window}주평균수확수" in df.columns
        ]
        if (
            str(target_col) == "수확수"
            and isinstance(metrics, dict)
            and metrics.get("R2", 0) < 0.3
            and isinstance(df, pd.DataFrame)
            and average_cols
        ):
            st.warning(
                f"현재 수확수 R²={metrics['R2']:.3f}입니다. "
                "수확수는 주별 편차가 클 수 있으므로 1~4주 이동평균 R² 비교 결과에서 "
                "가장 높은 기간을 적용해 보세요."
            )
    except Exception:
        pass



def make_regression_models_for_harvest():
    """수확수 계열 안정화 모드에서 비교할 회귀모델."""
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    try:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror"
        )
    except Exception:
        pass
    try:
        models["LGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )
    except Exception:
        pass
    return models


def build_harvest_feature_set(df, target_col, base_features=None):
    """
    수확수 계열 예측을 위한 안정형 feature set.
    - target_col 자체는 제외
    - 수확수 이동평균 타깃과 원본 수확수 간 직접 누설을 방지
    - 선택한 1~4주 평균수확수는 환경 Feature만으로 예측
    """
    if base_features is None:
        base_features = []

    candidate = []
    candidate += [c for c in base_features if c in df.columns]
    candidate += get_harvest_boost_extra_features(df)

    # 환경 rolling feature 자동 포함
    env_keywords = ["평균주간온도", "평균야간온도", "평균주간습도", "평균야간습도", "평균주간CO", "평균야간CO", "누적일사", "평균수분부족분", "수분부족분", "VPD"]
    candidate += [c for c in df.columns if any(k in str(c) for k in env_keywords)]

    # 사용 불가 컬럼 제외
    exclude = {
        target_col, "조사일자", "date", "datetime", "수확중량", "수확량",
    }
    # 수확수 이동평균을 target으로 선택한 경우 원본·다른 이동평균값을
    # Feature에서 제외하여 직접적인 target leakage를 방지합니다.
    if re.fullmatch(r"[1-4]주평균수확수", str(target_col)):
        exclude.add("수확수")
        for window in range(1, 5):
            exclude.add(f"{window}주평균수확수")

    features = []
    for c in candidate:
        if c not in exclude and c in df.columns and c not in features:
            features.append(c)
    return features


def evaluate_harvest_high_performance(df, target_col, base_features=None, min_rows=10):
    """
    수확수/1~4주평균수확수 전용 고성능 평가.
    - 파생변수 생성
    - TimeSeriesSplit 가능 시 사용
    - RF/XGB/LGBM/GBR 비교
    - 최고 R² 모델 반환
    """
    work = add_harvest_enhancement_features(df)
    features = build_harvest_feature_set(work, target_col, base_features)

    if target_col not in work.columns or len(features) == 0:
        return None

    model_df = work[features + [target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(model_df) < min_rows:
        return None

    X_all = model_df[features]
    y_all = model_df[target_col]

    models = make_regression_models_for_harvest()
    rows = []
    best = None

    # 데이터가 충분하면 TimeSeriesSplit, 부족하면 holdout
    if len(model_df) >= 15:
        n_splits = min(5, max(2, len(model_df) // 5))
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for model_name, model in models.items():
            fold_metrics = []
            last_pack = None
            for tr_idx, te_idx in splitter.split(X_all):
                X_train, X_test = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_train, y_test = y_all.iloc[tr_idx], y_all.iloc[te_idx]
                if len(y_test) < 2:
                    continue
                try:
                    m = model.__class__(**model.get_params()) if hasattr(model, "get_params") else model
                    m.fit(X_train, y_train)
                    pred = safe_predict(m, X_test, features)
                    met = compute_metrics(y_test, pred)
                    fold_metrics.append(met)
                    last_pack = (m, X_train, X_test, y_train, y_test, pred)
                except Exception:
                    continue
            if fold_metrics:
                avg_mse = float(np.mean([m["MSE"] for m in fold_metrics]))
                avg_mae = float(np.mean([m["MAE"] for m in fold_metrics]))
                avg_r2 = float(np.mean([m["R2"] for m in fold_metrics]))
                rows.append({"Model": model_name, "MSE": avg_mse, "MAE": avg_mae, "R2": avg_r2, "CV": "TimeSeriesSplit", "FeatureCount": len(features), "Rows": len(model_df)})
                if best is None or avg_r2 > best["metrics"]["R2"]:
                    # 최종 모델은 전체 데이터로 재학습
                    final_model = model.__class__(**model.get_params()) if hasattr(model, "get_params") else model
                    final_model.fit(X_all, y_all)
                    best = {
                        "model_name": model_name,
                        "model": final_model,
                        "metrics": {"MSE": avg_mse, "MAE": avg_mae, "R2": avg_r2},
                        "features": features,
                        "X": X_all,
                        "y": y_all,
                        "df": model_df,
                        "cv_table": None
                    }
    else:
        for model_name, model in models.items():
            try:
                X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
                model.fit(X_train, y_train)
                pred = safe_predict(model, X_test, features)
                met = compute_metrics(y_test, pred)
                rows.append({"Model": model_name, "MSE": met["MSE"], "MAE": met["MAE"], "R2": met["R2"], "CV": "Holdout", "FeatureCount": len(features), "Rows": len(model_df)})
                if best is None or met["R2"] > best["metrics"]["R2"]:
                    final_model = model.__class__(**model.get_params()) if hasattr(model, "get_params") else model
                    final_model.fit(X_all, y_all)
                    best = {
                        "model_name": model_name,
                        "model": final_model,
                        "metrics": met,
                        "features": features,
                        "X": X_all,
                        "y": y_all,
                        "df": model_df,
                        "cv_table": None
                    }
            except Exception:
                continue

    if best is None:
        return None
    best["compare_table"] = pd.DataFrame(rows).sort_values("R2", ascending=False) if rows else pd.DataFrame()
    return best


def render_harvest_high_performance_panel(df, target_col, base_features=None):
    """고성능 수확수 모드는 제거되었습니다. 기존 호출이 남아도 화면에 표시하지 않습니다."""
    return None


def _model_feature_group_name(column_name):
    """매핑데이터 환경 Feature명을 UI의 12개 입력그룹으로 변환합니다."""
    name = str(column_name)
    if "평균주간온도" in name:
        return "주간온도"
    if "평균야간온도" in name:
        return "야간온도"
    if "평균주간습도" in name:
        return "주간습도"
    if "평균야간습도" in name:
        return "야간습도"
    if "평균주간CO" in name:
        return "주간CO₂"
    if "평균야간CO" in name:
        return "야간CO₂"
    if "평균누적일사량" in name:
        return "누적일사량"
    if "평균수분부족분" in name or "수분부족분" in name:
        return "수분부족분"
    if "ADT(" in name:
        return "ADT"
    if "DIF(" in name:
        return "DIF"
    if "GDD(" in name:
        return "GDD"
    if "VPD(" in name:
        return "VPD"
    return None


def get_environment_feature_columns(df, apply_user_selection=True):
    """
    모델/XAI에 사용할 환경 Feature를 반환합니다.

    v26.8:
    - 기존 주/야 온도·습도·CO₂·누적일사량·수분부족분 유지
    - ADT/DIF/GDD/VPD를 모델 입력 후보에 포함
    - 사용자가 '머신러닝 입력 환경변수 선택'에서 선택한 그룹만 반환
    - apply_user_selection=False이면 UI 후보 생성을 위해 전체 사용 가능 Feature 반환
    """
    exclude_keywords = [
        "수확수", "착과수", "개화수", "평균과중", "착과잔량", "누적착과수", "누적수확수",
        "Lag", "lag", "초장", "생장길이", "엽수", "엽장", "엽폭", "줄기굵기", "화방높이"
    ]

    base_patterns = [
        "평균주간온도", "평균야간온도",
        "평균주간습도", "평균야간습도",
        "평균주간CO", "평균야간CO",
        "평균누적일사량", "평균수분부족분",
        "ADT(", "DIF(", "GDD(", "VPD(",
    ]

    selected = []

    # 현재 df에 존재하는 1~7주 rolling 환경 Feature를 주차 순으로 수집합니다.
    for week in range(1, 8):
        week_prefix = f"{week}주"
        for key in base_patterns:
            candidates = []
            for col in df.columns:
                col_s = str(col)
                if (
                    col_s.startswith(week_prefix)
                    and key in col_s
                    and not any(ex in col_s for ex in exclude_keywords)
                ):
                    candidates.append(col)
            if candidates and candidates[0] not in selected:
                selected.append(candidates[0])

    # 주차 prefix가 없는 데이터에 대한 fallback
    if not selected:
        for key in base_patterns:
            for col in df.columns:
                col_s = str(col)
                if key in col_s and not any(ex in col_s for ex in exclude_keywords):
                    if col not in selected:
                        selected.append(col)
                    break

    selected = list(dict.fromkeys(selected))

    if apply_user_selection:
        chosen_groups = st.session_state.get("selected_model_feature_groups")
        if chosen_groups is not None:
            chosen_groups = set(chosen_groups)
            selected = [
                col for col in selected
                if _model_feature_group_name(col) in chosen_groups
            ]

    return selected

def get_report_target_name(target_col):
    """
    자동 리포트 표시명 보정:
    - 1~4주평균수확수 선택 시 리포트는 '수확수'로 표시
    - 착과잔량(Fruit Load) 선택 시 리포트는 '착과수'로 표시
    """
    target_name = str(target_col)
    if re.fullmatch(r"[1-4]주평균수확수", target_name):
        return "수확수"
    if target_name == "착과잔량(Fruit Load)":
        return "착과수"
    return target_name



def is_derived_prediction_target(target_col):
    """
    v22 변경:
    1~4주평균수확수와 착과잔량(Fruit Load)도 예측대상으로 선택하면
    모델평가 이후 SHAP, Feature Importance, Temporal SHAP,
    Feature × Week Heatmap, ICE/PDP, ALE까지 모두 수행합니다.
    """
    return False

def show_derived_target_stop_message(target_col):
    """파생 예측대상 선택 시 XAI 분석 중단 안내."""
    base_name = get_report_target_name(target_col)
    st.info(
        f"""
        ℹ️ 현재 예측대상은 **{target_col}** 입니다.

        이 변수는 **{base_name} 기반 파생변수**이므로 모델 평가 지표까지만 제공합니다.

        이후 분석은 수행하지 않습니다.

        - SHAP Summary
        - Feature Importance
        - Temporal SHAP
        - Feature × Week Heatmap
        - ICE + PDP
        - Centered ALE
        - 종합 리포트

        파생변수에 대해 XAI를 수행하면 원자료 해석이 왜곡될 수 있으므로, 논문 해석용 XAI는 **수확수** 또는 **착과수**를 선택해 수행하는 것을 권장합니다.
        """
    )


def align_feature_vector(features, values):
    """Feature명과 중요도/SHAP 배열 길이가 다를 때 짧은 길이에 맞춰 안전하게 정렬합니다."""
    features = list(features)
    values = np.asarray(values).reshape(-1)
    n = min(len(features), len(values))
    return features[:n], values[:n]


def render_r2_harvest_compare(df, features, model_choice):
    """현재 선택된 환경변수(수분부족분 포함)로 수확수와 4주평균수확수 R²를 비교합니다."""
    try:
        compare_targets = [c for c in ["수확수", "4주평균수확수"] if c in df.columns]
        rows = []

        for tcol in compare_targets:
            work = df[features + [tcol]].copy()
            work = work.apply(pd.to_numeric, errors="coerce").dropna()
            if len(work) < 6:
                continue

            X_cmp = work[features].copy()
            y_cmp = work[tcol].copy()
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_cmp, y_cmp, test_size=0.2, random_state=42
            )
            if len(y_te) < 2:
                continue

            m = make_model(model_choice)
            m.fit(X_tr, y_tr)
            pred = safe_predict(m, X_te, features)
            met = compute_metrics(y_te, pred)

            rows.append({
                "예측대상": tcol,
                "MSE": met["MSE"],
                "MAE": met["MAE"],
                "R2": met["R2"],
                "사용 Feature 수": len(features),
                "데이터 수": len(work)
            })

        if len(rows) == 0:
            return

        compare_df = pd.DataFrame(rows)

        st.markdown("### 🍅 수확수 vs 4주평균수확수 R² 비교")
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.bar(compare_df["예측대상"], compare_df["R2"])
        ax.set_ylabel("R² (결정계수)")
        ax.set_xlabel("예측대상")
        ax.set_title("수확수 vs 4주평균수확수 R² 비교")
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)
        display_matplotlib(fig)
        plt.close(fig)

        st.dataframe(compare_df.round(4), use_container_width=True, hide_index=True)

        if "수확수" in compare_df["예측대상"].values and "4주평균수확수" in compare_df["예측대상"].values:
            r2_raw = float(compare_df.loc[compare_df["예측대상"] == "수확수", "R2"].iloc[0])
            r2_ma4 = float(compare_df.loc[compare_df["예측대상"] == "4주평균수확수", "R2"].iloc[0])
            diff = r2_ma4 - r2_raw
            if diff > 0:
                st.success(f"4주평균수확수의 R²가 원자료 수확수보다 {diff:.3f} 높습니다. 수확수 변동성이 완화된 결과로 해석할 수 있습니다.")
            else:
                st.info(f"4주평균수확수의 R²가 원자료 수확수보다 {abs(diff):.3f} 낮거나 유사합니다. 데이터 길이와 수확 변동성을 함께 검토하세요.")
    except Exception as e:
        st.warning(f"수확수 R² 비교 그래프 생성 오류: {e}")


def feature_importance_table(model, features):
    """Feature Importance 안전 생성: Feature명과 중요도 배열 길이가 달라도 오류가 나지 않도록 보정합니다."""
    features = get_model_feature_names(model, features)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_).reshape(-1)
        n = min(len(features), len(importances))
        fi_df = pd.DataFrame({
            "Feature": features[:n],
            "Importance": importances[:n]
        }).sort_values("Importance", ascending=False)
    else:
        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": np.zeros(len(features))
        }).sort_values("Importance", ascending=False)
    return fi_df


def permutation_importance_table(
    model,
    X_eval,
    y_eval,
    features,
    n_repeats=30,
    random_state=42,
):
    """
    평가 데이터에서 Permutation Importance를 계산합니다.

    중요도 정의:
    - 각 Feature 값을 무작위로 섞었을 때 모델 점수가 얼마나 감소하는지 측정합니다.
    - Importance Mean이 클수록 해당 Feature에 대한 예측 의존도가 높습니다.
    - 음수 중요도는 해당 Feature를 섞었을 때 오히려 점수가 개선된 경우로,
      표본 변동, 과적합, 공선성 또는 낮은 유효정보의 신호일 수 있습니다.

    Python 3.9 호환 / feature mismatch 방지.
    """
    feature_names = get_model_feature_names(model, features)
    X_eval_aligned = align_xai_input(X_eval, feature_names, model)
    y_eval_series = pd.Series(y_eval, index=X_eval_aligned.index)
    y_eval_series = pd.to_numeric(y_eval_series, errors="coerce")

    valid_mask = y_eval_series.notna()
    X_eval_aligned = X_eval_aligned.loc[valid_mask].copy()
    y_eval_series = y_eval_series.loc[valid_mask].copy()

    if X_eval_aligned.empty or len(X_eval_aligned) < 3:
        return pd.DataFrame()

    result = permutation_importance(
        model,
        X_eval_aligned,
        y_eval_series,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        scoring=None,
        n_jobs=-1,
    )

    importances_mean = np.asarray(result.importances_mean).reshape(-1)
    importances_std = np.asarray(result.importances_std).reshape(-1)
    n = min(len(feature_names), len(importances_mean), len(importances_std))

    pi_df = pd.DataFrame({
        "Feature": list(feature_names)[:n],
        "Importance Mean": importances_mean[:n],
        "Importance Std": importances_std[:n],
    })
    pi_df["Abs Importance"] = pi_df["Importance Mean"].abs()
    pi_df = (
        pi_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["Importance Mean", "Importance Std"])
        .sort_values("Importance Mean", ascending=False)
        .reset_index(drop=True)
    )
    return pi_df


def explain_permutation_importance(pi_df, target_name, model_choice):
    """Permutation Importance 자동 요약."""
    if pi_df is None or pi_df.empty:
        return "Permutation Importance를 계산할 수 없습니다."

    top = pi_df.iloc[0]
    mean_imp = float(pi_df["Importance Mean"].mean())
    positive_count = int((pi_df["Importance Mean"] > 0).sum())
    negative_count = int((pi_df["Importance Mean"] < 0).sum())

    return (
        f"{model_choice} 모델에서 '{target_name}' 예측값을 기준으로 Feature를 반복적으로 섞어 "
        f"예측성능 변화를 평가했습니다. 가장 중요한 변수는 '{top['Feature']}'이며 "
        f"Permutation Importance Mean={top['Importance Mean']:.6f} "
        f"(Std={top['Importance Std']:.6f})입니다. 전체 Feature의 평균 중요도는 "
        f"{mean_imp:.6f}이며, 양의 중요도 Feature는 {positive_count}개, "
        f"음의 중요도 Feature는 {negative_count}개입니다. "
        "값이 클수록 해당 Feature의 정보를 보존하는 것이 모델 성능에 중요하다는 의미입니다."
    )


def build_global_xai_rank_comparison(shap_df, fi_df, pi_df):
    """
    SHAP / Model FI / Permutation Importance의 순위를 같은 Feature 기준으로 비교합니다.
    낮은 Rank가 더 중요하며, Mean Rank가 낮을수록 세 방법이 공통적으로 중요하게 본 변수입니다.
    """
    frames = []

    if shap_df is not None and not shap_df.empty:
        s = shap_df[["Feature", "Mean(|SHAP|)"]].copy()
        s["SHAP Rank"] = s["Mean(|SHAP|)"].rank(
            method="min", ascending=False
        )
        frames.append(s)

    if fi_df is not None and not fi_df.empty:
        f = fi_df[["Feature", "Importance"]].copy()
        f["FI Rank"] = f["Importance"].rank(
            method="min", ascending=False
        )
        frames.append(f)

    if pi_df is not None and not pi_df.empty:
        p = pi_df[["Feature", "Importance Mean", "Importance Std"]].copy()
        # Permutation Importance는 '성능 감소량' 자체를 기준으로 내림차순 순위
        p["Permutation Rank"] = p["Importance Mean"].rank(
            method="min", ascending=False
        )
        frames.append(p)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Feature", how="outer")

    rank_cols = [
        c for c in ["SHAP Rank", "FI Rank", "Permutation Rank"]
        if c in merged.columns
    ]
    if rank_cols:
        merged["Mean Rank"] = merged[rank_cols].mean(axis=1, skipna=True)
        merged["Rank Std"] = merged[rank_cols].std(axis=1, skipna=True).fillna(0.0)
        merged["Top3 Count"] = sum(
            (merged[c] <= 3).astype(int) for c in rank_cols
        )
        merged = merged.sort_values(
            ["Mean Rank", "Top3 Count", "Rank Std"],
            ascending=[True, False, True],
        )

    return merged.reset_index(drop=True)


def explain_global_xai_consensus(rank_df):
    """세 중요도 기법의 합의 결과를 자동 설명합니다."""
    if rank_df is None or rank_df.empty:
        return "세 중요도 기법의 순위 비교 결과를 생성할 수 없습니다."

    top = rank_df.iloc[0]
    rank_bits = []
    for col, label in [
        ("SHAP Rank", "SHAP"),
        ("FI Rank", "Model FI"),
        ("Permutation Rank", "Permutation"),
    ]:
        if col in rank_df.columns and pd.notna(top.get(col)):
            rank_bits.append(f"{label} {int(top[col])}위")

    top3_count = int(top.get("Top3 Count", 0))
    return (
        f"세 중요도 기법의 합의순위 1위는 '{top['Feature']}'입니다. "
        + (", ".join(rank_bits) + ". " if rank_bits else "")
        + f"세 방법 중 {top3_count}개 방법에서 Top 3에 포함되었습니다. "
        "SHAP은 예측값 기여 크기, Model Feature Importance는 모델 내부 학습구조, "
        "Permutation Importance는 Feature를 섞었을 때의 실제 성능 저하를 측정하므로 "
        "세 방법에서 동시에 상위권이면 결과의 해석적 일관성이 높은 Feature로 볼 수 있습니다."
    )


def render_global_xai_metric_cards(top_feature, mean_importance, method_label):
    """각 중요도 분석 아래 Top Feature / Mean Importance 정량 카드를 표시합니다."""
    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"{method_label} Top Feature", str(top_feature))
    with c2:
        if np.isfinite(mean_importance):
            st.metric(f"{method_label} Mean Importance", f"{mean_importance:.6f}")
        else:
            st.metric(f"{method_label} Mean Importance", "N/A")


def summarize_shap_results(shap_values, features):
    """SHAP 결과 안전 요약: SHAP 배열과 Feature 수가 달라도 짧은 길이에 맞춰 보정합니다."""
    values = getattr(shap_values, "values", shap_values)
    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    shap_mean_abs = np.abs(values).mean(axis=0)
    shap_mean_signed = values.mean(axis=0)
    n = min(len(features), len(shap_mean_abs), len(shap_mean_signed))

    shap_df = pd.DataFrame({
        "Feature": list(features)[:n],
        "Mean(|SHAP|)": shap_mean_abs[:n],
        "Mean(SHAP)": shap_mean_signed[:n],
    }).sort_values("Mean(|SHAP|)", ascending=False)
    return shap_df

def infer_controllable_features(feature_names):
    keywords = ["온도", "습도", "수분부족분", "HD", "CO2", "CO₂", "일사", "광", "temp", "hum", "humidity", "deficit", "co2", "solar"]
    selected = []
    for f in feature_names:
        low = str(f).lower()
        if any(k.lower() in low for k in keywords):
            selected.append(f)
    return selected


def build_window_feature_name(week, suffix):
    return f"{week}주{suffix}"






def apply_graph_design(fig):
    """사이드바에서 선택한 그래프 스타일을 matplotlib Figure에 공통 적용합니다."""
    base_font = 10 * font_scale
    title_font = 12 * font_scale
    label_font = 10 * font_scale
    tick_font = 9 * font_scale
    lw = 1.4 * line_width_scale

    style_cfg = {
        "기본": {"fig_bg": "white", "ax_bg": "white", "grid": "#d1d5db", "spine": "#64748b", "text": "#0f172a"},
        "논문(Paper)": {"fig_bg": "white", "ax_bg": "white", "grid": "#e5e7eb", "spine": "#111827", "text": "#000000"},
        "발표(Presentation)": {"fig_bg": "#ffffff", "ax_bg": "#f8fafc", "grid": "#cbd5e1", "spine": "#334155", "text": "#0f172a"},
        "다크모드": {"fig_bg": "#0f172a", "ax_bg": "#111827", "grid": "#334155", "spine": "#94a3b8", "text": "#f8fafc"},
        "스마트팜": {"fig_bg": "#f0fdf4", "ax_bg": "#ffffff", "grid": "#bbf7d0", "spine": "#16a34a", "text": "#14532d"},
        "컬러풀": {"fig_bg": "#fff7ed", "ax_bg": "#ffffff", "grid": "#fed7aa", "spine": "#f97316", "text": "#7c2d12"},
    }
    cfg = style_cfg.get(graph_theme, style_cfg["기본"])

    fig.patch.set_facecolor(cfg["fig_bg"])
    for ax in fig.get_axes():
        ax.set_facecolor(cfg["ax_bg"])
        ax.grid(True, linestyle="--", alpha=0.45, color=cfg["grid"])
        ax.title.set_fontsize(title_font)
        ax.title.set_fontweight("bold")
        ax.title.set_color(cfg["text"])
        ax.xaxis.label.set_fontsize(label_font)
        ax.yaxis.label.set_fontsize(label_font)
        ax.xaxis.label.set_color(cfg["text"])
        ax.yaxis.label.set_color(cfg["text"])
        ax.tick_params(axis="both", labelsize=tick_font, colors=cfg["text"])
        for spine in ax.spines.values():
            spine.set_color(cfg["spine"])
            spine.set_linewidth(max(0.8, lw * 0.7))
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), lw))
        for patch in ax.patches:
            try:
                patch.set_alpha(0.88)
            except Exception:
                pass
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_alpha(0.85)
            for txt in leg.get_texts():
                txt.set_fontsize(base_font)
                txt.set_color(cfg["text"])
    try:
        fig.tight_layout(pad=0.6)
    except Exception:
        pass
    return fig


def display_matplotlib(fig, use_container_width=True):
    """
    Matplotlib Figure를 메모리 안전하게 출력합니다.

    - Figure 크기와 DPI 상한을 적용해 RendererAgg 대용량 버퍼 생성을 방지
    - MemoryError 발생 시 더 작은 크기와 72 DPI로 한 번 재시도
    - 출력 후 Figure, Canvas, 가비지 메모리를 즉시 정리
    """
    try:
        apply_graph_design(fig)
    except Exception as e:
        st.warning(f"그래프 디자인 적용 중 경고: {e}")

    try:
        width, height = [float(v) for v in fig.get_size_inches()]
        width = max(2.0, min(width, 11.5))
        height = max(2.0, min(height, 7.5))

        target_dpi = 90.0
        max_pixels = 5_500_000
        pixel_count = width * height * (target_dpi ** 2)

        if pixel_count > max_pixels:
            scale = (max_pixels / pixel_count) ** 0.5
            width *= scale
            height *= scale

        fig.set_size_inches(width, height, forward=True)
        fig.set_dpi(target_dpi)

        try:
            st.pyplot(
                fig,
                use_container_width=use_container_width,
                clear_figure=True,
                dpi=int(target_dpi),
                bbox_inches=None,
            )
        except TypeError:
            st.pyplot(fig, use_container_width=use_container_width)
        except MemoryError:
            gc.collect()
            fig.set_size_inches(7.0, 4.2, forward=True)
            fig.set_dpi(72)
            try:
                st.pyplot(
                    fig,
                    use_container_width=use_container_width,
                    clear_figure=True,
                    dpi=72,
                    bbox_inches=None,
                )
            except TypeError:
                st.pyplot(fig)
    finally:
        try:
            fig.clear()
        except Exception:
            pass
        try:
            plt.close(fig)
            plt.close("all")
        except Exception:
            pass
        gc.collect()

def display_plotly(fig, use_container_width=True):
    """Plotly 그래프 템플릿을 적용해 안전하게 출력합니다."""
    try:
        fig.update_layout(
            template=plotly_template,
            font=dict(size=max(10, int(12 * font_scale))),
            margin=dict(l=20, r=20, t=45, b=20),
        )
    except Exception as e:
        st.warning(f"Plotly 디자인 적용 중 경고: {e}")
    st.plotly_chart(fig, use_container_width=use_container_width)


def style_dataframe(df):
    """표 디자인 선택에 따라 pandas Styler를 반환합니다. 오류 시 원본 DataFrame 반환."""
    try:
        if table_theme == "논문":
            return df.style.set_table_styles([
                {"selector": "th", "props": [("background-color", "#f8fafc"), ("color", "#000"), ("font-weight", "bold")]},
                {"selector": "td", "props": [("border-bottom", "1px solid #e5e7eb")]},
            ]).format(precision=3)
        if table_theme == "카드형":
            return df.style.set_table_styles([
                {"selector": "th", "props": [("background", "#ecfdf5"), ("color", "#065f46"), ("font-weight", "bold")]},
            ]).format(precision=3)
        return df
    except Exception:
        return df


def pretty_time_text(value):
    """보고서/그래프 설명에 표시되는 시간 문자열을 사람이 읽기 쉬운 형태로 변환합니다."""
    if value is None:
        return value
    txt = str(value)
    replacements = {
        "(0818시)": "(08~18시)",
        "(08~18시)": "(08~18시)",
        "(08018시)": "(08~18시)",
        "(1907시)": "(19~07시)",
        "(19~07시)": "(19~07시)",
        "(19007시)": "(19~07시)",
        "(0618시)": "(06~18시)",
        "(06~18시)": "(06~18시)",
        "0818시": "08~18시",
        "1907시": "19~07시",
        "0618시": "06~18시",
    }
    for old, new in replacements.items():
        txt = txt.replace(old, new)
    return txt


def format_interval_text(intervals, limit=3):
    """ALE/PDP 구간을 '23.15 ~ 31.16' 형식으로 안전하게 표시합니다."""
    if not intervals:
        return "없음"
    out = []
    for item in intervals[:limit]:
        try:
            a, b = item[0], item[1]
            out.append(f"{float(a):.2f} ~ {float(b):.2f}")
        except Exception:
            continue
    return ", ".join(out) if out else "없음"


def parse_week_and_base_feature(feature_name: str):
    m = re.search(r"(\d+)주(.+)", str(feature_name))
    if m:
        return int(m.group(1)), m.group(2)
    return None, str(feature_name)


def build_temporal_shap_tables(shap_values, features):
    values = getattr(shap_values, "values", shap_values)
    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    n = min(len(features), len(mean_abs), len(mean_signed))

    rows = []
    for feat, abs_val, signed_val in zip(list(features)[:n], mean_abs[:n], mean_signed[:n]):
        week, base_feat = parse_week_and_base_feature(feat)
        if week is not None:
            rows.append({
                "Feature": feat,
                "Week": week,
                "BaseFeature": base_feat,
                "Mean(|SHAP|)": float(abs_val),
                "Mean(SHAP)": float(signed_val)
            })

    if len(rows) == 0:
        return None, None, None

    temporal_df = pd.DataFrame(rows)
    week_df = temporal_df.groupby("Week", as_index=False).agg(
        TotalMeanAbsSHAP=("Mean(|SHAP|)", "sum"),
        AvgSignedSHAP=("Mean(SHAP)", "mean"),
        FeatureCount=("Feature", "count")
    ).sort_values("Week")

    heatmap_df = temporal_df.pivot_table(
        index="BaseFeature",
        columns="Week",
        values="Mean(|SHAP|)",
        aggfunc="sum"
    ).fillna(0)

    return temporal_df, week_df, heatmap_df


def build_lag_shap_analysis(
    shap_values,
    features,
    n_bootstrap=1000,
    confidence=0.95,
    random_state=42,
):
    """
    명시적 주차(lag) Feature의 SHAP 값을 Lag 단위로 집계합니다.

    Lag SHAP 정의(v26.5)
    --------------------
    각 평가 샘플 i와 lag w에 대해,

        LagAbsSHAP(i,w)
        = Σ_j |SHAP(i,j)|,   j ∈ features of lag w

    로 정의하고, 평가 샘플 평균을 해당 lag의 중요도로 사용합니다.

    95% Bootstrap CI
    ----------------
    이미 계산된 평가 샘플별 LagAbsSHAP 행을 복원추출하여
    각 lag 평균의 percentile confidence interval을 계산합니다.

    주의:
    - 이 CI는 '평가 샘플 구성에 따른 lag 중요도 평균의 불확실성'입니다.
    - 모델을 매 bootstrap마다 재학습하는 CI가 아닙니다.
    - Lag SHAP은 별도의 새로운 Shapley 알고리즘이라기보다,
      명시적으로 생성된 주차별 Feature의 SHAP을 lag 단위로 집계한 분석입니다.
    """
    values = getattr(shap_values, "values", shap_values)
    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    feature_list = list(features)
    n_features = min(values.shape[1], len(feature_list))
    values = values[:, :n_features]
    feature_list = feature_list[:n_features]

    lag_to_indices = {}
    lag_to_base_features = {}

    for idx, feat in enumerate(feature_list):
        week, base_feat = parse_week_and_base_feature(feat)
        if week is None:
            continue
        lag_to_indices.setdefault(int(week), []).append(idx)
        lag_to_base_features.setdefault(int(week), []).append(base_feat)

    if not lag_to_indices:
        return None

    lags = sorted(lag_to_indices.keys())
    n_samples = values.shape[0]

    sample_abs = np.zeros((n_samples, len(lags)), dtype=float)
    sample_signed = np.zeros((n_samples, len(lags)), dtype=float)

    for col_idx, lag in enumerate(lags):
        idxs = lag_to_indices[lag]
        lag_values = values[:, idxs]

        # 해당 lag에 속한 모든 환경 Feature의 영향 크기를 합산
        sample_abs[:, col_idx] = np.nansum(
            np.abs(lag_values),
            axis=1,
        )

        # 방향성은 feature 수에 크게 종속되지 않도록 평균 signed SHAP 사용
        sample_signed[:, col_idx] = np.nanmean(
            lag_values,
            axis=1,
        )

    mean_abs = np.nanmean(sample_abs, axis=0)
    mean_signed = np.nanmean(sample_signed, axis=0)
    std_abs = np.nanstd(sample_abs, axis=0, ddof=1) if n_samples > 1 else np.zeros(len(lags))

    # --------------------------------------------------------
    # 평가 샘플 bootstrap CI
    # --------------------------------------------------------
    rng = np.random.RandomState(int(random_state))
    boot_means = []

    if n_samples >= 2 and int(n_bootstrap) > 0:
        for _ in range(int(n_bootstrap)):
            idx = rng.randint(
                0,
                n_samples,
                size=n_samples,
            )
            boot_means.append(
                np.nanmean(sample_abs[idx, :], axis=0)
            )

    if boot_means:
        boot_matrix = np.asarray(boot_means, dtype=float)
        alpha = (1.0 - float(confidence)) / 2.0
        lower = np.nanpercentile(
            boot_matrix,
            100.0 * alpha,
            axis=0,
        )
        upper = np.nanpercentile(
            boot_matrix,
            100.0 * (1.0 - alpha),
            axis=0,
        )
    else:
        boot_matrix = np.empty((0, len(lags)))
        lower = mean_abs.copy()
        upper = mean_abs.copy()

    total = float(np.nansum(mean_abs))
    importance_pct = (
        mean_abs / total * 100.0
        if total > 0
        else np.zeros_like(mean_abs)
    )

    lag_df = pd.DataFrame({
        "Lag": lags,
        "MeanAbsLagSHAP": mean_abs,
        "StdLagSHAP": std_abs,
        "MeanSignedLagSHAP": mean_signed,
        "CI95_Lower": lower,
        "CI95_Upper": upper,
        "ImportancePct": importance_pct,
        "FeatureCount": [
            len(lag_to_indices[lag])
            for lag in lags
        ],
    })

    # Lag 1 → Lag 7 순으로 누적 기여도 계산
    lag_df = lag_df.sort_values("Lag").reset_index(drop=True)
    lag_df["CumulativePct"] = (
        lag_df["ImportancePct"].cumsum()
    )

    # 중요도 순위
    rank_order = (
        lag_df["MeanAbsLagSHAP"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )
    lag_df["Rank"] = rank_order

    peak_row = lag_df.loc[
        lag_df["MeanAbsLagSHAP"].idxmax()
    ].copy()

    # --------------------------------------------------------
    # Lag × Feature Heatmap
    # --------------------------------------------------------
    detail_rows = []
    for feat_idx, feat in enumerate(feature_list):
        week, base_feat = parse_week_and_base_feature(feat)
        if week is None:
            continue

        feat_values = values[:, feat_idx]
        detail_rows.append({
            "Lag": int(week),
            "BaseFeature": base_feat,
            "MeanAbsSHAP": float(
                np.nanmean(np.abs(feat_values))
            ),
            "MeanSignedSHAP": float(
                np.nanmean(feat_values)
            ),
        })

    detail_df = pd.DataFrame(detail_rows)

    if detail_df.empty:
        abs_heatmap = pd.DataFrame()
        signed_heatmap = pd.DataFrame()
    else:
        abs_heatmap = (
            detail_df
            .pivot_table(
                index="BaseFeature",
                columns="Lag",
                values="MeanAbsSHAP",
                aggfunc="sum",
            )
            .fillna(0.0)
        )
        signed_heatmap = (
            detail_df
            .pivot_table(
                index="BaseFeature",
                columns="Lag",
                values="MeanSignedSHAP",
                aggfunc="mean",
            )
            .fillna(0.0)
        )

        ordered_cols = sorted(abs_heatmap.columns)
        abs_heatmap = abs_heatmap.reindex(
            columns=ordered_cols
        )
        signed_heatmap = signed_heatmap.reindex(
            columns=ordered_cols
        )

    return {
        "lag_table": lag_df,
        "peak_lag": int(peak_row["Lag"]),
        "peak_importance": float(
            peak_row["MeanAbsLagSHAP"]
        ),
        "peak_share": float(
            peak_row["ImportancePct"]
        ),
        "peak_ci_lower": float(
            peak_row["CI95_Lower"]
        ),
        "peak_ci_upper": float(
            peak_row["CI95_Upper"]
        ),
        "sample_abs_matrix": sample_abs,
        "sample_signed_matrix": sample_signed,
        "bootstrap_matrix": boot_matrix,
        "valid_bootstrap": int(
            boot_matrix.shape[0]
        ),
        "requested_bootstrap": int(
            n_bootstrap
        ),
        "confidence": float(confidence),
        "detail_table": detail_df,
        "abs_heatmap": abs_heatmap,
        "signed_heatmap": signed_heatmap,
    }


def explain_lag_shap_result(
    lag_result,
    target_name="예측 대상",
):
    if (
        lag_result is None
        or lag_result.get("lag_table") is None
        or lag_result["lag_table"].empty
    ):
        return "Lag SHAP 결과를 해석할 수 없습니다."

    lag_df = lag_result["lag_table"].copy()
    peak_lag = int(lag_result["peak_lag"])
    peak_importance = float(
        lag_result["peak_importance"]
    )
    peak_share = float(
        lag_result["peak_share"]
    )
    lower = float(
        lag_result["peak_ci_lower"]
    )
    upper = float(
        lag_result["peak_ci_upper"]
    )

    top3 = (
        lag_df.sort_values(
            "MeanAbsLagSHAP",
            ascending=False,
        )
        .head(3)
    )
    top3_text = " → ".join(
        [
            (
                f"{int(r['Lag'])}주 전"
                f"({r['MeanAbsLagSHAP']:.4f}, "
                f"{r['ImportancePct']:.1f}%)"
            )
            for _, r in top3.iterrows()
        ]
    )

    peak_signed_rows = lag_df[
        lag_df["Lag"] == peak_lag
    ]
    peak_signed = (
        float(
            peak_signed_rows.iloc[0][
                "MeanSignedLagSHAP"
            ]
        )
        if not peak_signed_rows.empty
        else np.nan
    )
    if np.isfinite(peak_signed):
        peak_direction = (
            "평균적으로 예측값을 증가시키는 방향"
            if peak_signed > 0
            else "평균적으로 예측값을 감소시키는 방향"
            if peak_signed < 0
            else "평균 방향성이 거의 중립"
        )
    else:
        peak_direction = "평균 방향성을 계산하기 어려움"

    # 50% 누적 기여에 처음 도달하는 lag
    cum50_rows = lag_df[
        lag_df["CumulativePct"] >= 50.0
    ]
    cum50_lag = (
        int(cum50_rows.iloc[0]["Lag"])
        if not cum50_rows.empty
        else int(lag_df["Lag"].max())
    )

    return (
        f"'{target_name}' 예측에서 Peak Lag는 {peak_lag}주 전입니다. "
        f"Peak Lag의 Mean absolute Lag SHAP은 {peak_importance:.4f}, "
        f"전체 lag 중요도 비중은 {peak_share:.1f}%이며 "
        f"95% bootstrap CI는 {lower:.4f}~{upper:.4f}입니다. "
        f"이 시점의 signed SHAP 평균은 {peak_signed:+.4f}로 {peak_direction}입니다. "
        f"중요도 상위 3개 lag는 {top3_text} 순입니다. "
        f"최근 1주부터 과거 방향으로 누적했을 때 총 lag 중요도의 50% 이상은 "
        f"{cum50_lag}주 전까지 포함하면 도달합니다. "
        "따라서 Lag SHAP은 '몇 주 전 환경이 현재 예측에 가장 크게 반영되었는가'를 "
        "정량화하는 데 사용할 수 있습니다. 단, 여기서 Lag SHAP은 명시적 주차별 Feature의 "
        "SHAP 값을 lag 단위로 집계한 분석이며 인과적 시간효과를 직접 증명하는 것은 아닙니다."
    )


def explain_lag_feature_heatmap(
    lag_result,
    target_name="예측 대상",
):
    if lag_result is None:
        return "Lag SHAP × Feature Heatmap 결과를 해석할 수 없습니다."

    abs_hm = lag_result.get("abs_heatmap")
    signed_hm = lag_result.get("signed_heatmap")

    if (
        abs_hm is None
        or abs_hm.empty
    ):
        return "Lag SHAP × Feature Heatmap 결과를 해석할 수 없습니다."

    idx = np.unravel_index(
        np.nanargmax(abs_hm.values),
        abs_hm.shape,
    )
    best_feature = abs_hm.index[idx[0]]
    best_lag = int(abs_hm.columns[idx[1]])
    best_value = float(
        abs_hm.iloc[idx[0], idx[1]]
    )

    signed_value = np.nan
    if (
        signed_hm is not None
        and not signed_hm.empty
        and best_feature in signed_hm.index
        and best_lag in signed_hm.columns
    ):
        signed_value = float(
            signed_hm.loc[
                best_feature,
                best_lag,
            ]
        )

    direction = (
        "증가 방향"
        if np.isfinite(signed_value) and signed_value > 0
        else "감소 방향"
        if np.isfinite(signed_value) and signed_value < 0
        else "방향성 중립 또는 불명확"
    )

    return (
        f"Lag SHAP × Feature Heatmap에서 가장 강한 조합은 "
        f"'{pretty_time_text(best_feature)}' × {best_lag}주 전이며 "
        f"Mean(|SHAP|)={best_value:.4f}입니다. "
        f"동일 조합의 Mean(SHAP)은 "
        f"{signed_value:+.4f}로 {direction}입니다. "
        f"이는 '{target_name}' 예측에서 어느 환경변수의 어느 lag가 핵심인지 "
        "동시에 식별하는 결과입니다."
    )


def explain_shap_summary(shap_df):
    if shap_df is None or shap_df.empty:
        return "SHAP 결과를 요약할 수 없습니다."
    top = shap_df.iloc[0]
    total = shap_df["Mean(|SHAP|)"].sum()
    pct = 100 * top["Mean(|SHAP|)"] / total if total > 0 else 0
    direction = "증가" if top["Mean(SHAP)"] > 0 else "감소"
    return (
        f"가장 영향력이 큰 변수는 '{top['Feature']}'이며, 평균 절대 SHAP 기여도는 {top['Mean(|SHAP|)']:.3f} 입니다. "
        f"전체 중요도 기준 비중은 약 {pct:.1f}% 입니다. 평균 방향성은 예측 {direction} 쪽입니다."
    )


def explain_feature_importance(fi_df):
    if fi_df is None or fi_df.empty:
        return "Feature Importance를 요약할 수 없습니다."
    top = fi_df.iloc[0]
    total = fi_df["Importance"].sum()
    pct = 100 * top["Importance"] / total if total > 0 else 0
    return f"모델 기반 중요도에서 가장 큰 변수는 '{top['Feature']}'이며 중요도는 {top['Importance']:.3f}, 비중은 약 {pct:.1f}% 입니다."


def explain_temporal_shap(week_df, target_name="예측 대상", report_target=None):
    if report_target is not None:
        target_name = report_target
    if week_df is None or week_df.empty:
        return "Temporal SHAP를 계산할 수 없습니다."
    best = week_df.sort_values("TotalMeanAbsSHAP", ascending=False).iloc[0]
    return (
        f"가장 영향력이 큰 시간 구간은 {int(best['Week'])}주 전이며, 총 Mean(|SHAP|)는 {best['TotalMeanAbsSHAP']:.3f} 입니다. "
        f"즉, 모델은 이 시기의 환경이 현재 예측 대상인 ({target_name})에 가장 크게 작용했다고 해석합니다."
    )


def explain_heatmap(heatmap_df, target_name="예측 대상", report_target=None):
    if report_target is not None:
        target_name = report_target
    if heatmap_df is None or heatmap_df.empty:
        return "Feature × Week Heatmap을 계산할 수 없습니다."
    idx = np.unravel_index(np.argmax(heatmap_df.values), heatmap_df.shape)
    best_feat = heatmap_df.index[idx[0]]
    best_week = heatmap_df.columns[idx[1]]
    best_val = heatmap_df.iloc[idx[0], idx[1]]
    return (
        f"({target_name})에 가장 영향력이 큰 조합은 '{best_feat}' × '{best_week}주' 이며, Mean(|SHAP|)는 {best_val:.3f} 입니다. "
        f"즉, {best_week}주 전의 {best_feat} 관리가 핵심이라는 뜻입니다."
    )


def explain_shap_summary_detail(shap_df, target_name, model_choice):
    if shap_df is None or shap_df.empty:
        return "SHAP 상세 설명을 생성할 수 없습니다."

    top_rows = shap_df.head(5).copy()
    total = shap_df["Mean(|SHAP|)"].sum()
    top = top_rows.iloc[0]

    lines = []
    lines.append(
        f"선택한 모델은 {model_choice}이며, 예측 대상은 '{target_name}'입니다. "
        "SHAP Summary는 모델이 예측할 때 어떤 환경 변수를 중요하게 사용했는지 보여줍니다."
    )
    lines.append(
        f"가장 영향력이 큰 변수는 '{top['Feature']}'입니다. "
        f"Mean(|SHAP|) 값은 {top['Mean(|SHAP|)']:.4f}로, 전체 변수 중 예측값 변동에 가장 크게 기여했습니다."
    )

    if top["Mean(SHAP)"] > 0:
        direction_text = "평균적으로 예측값을 증가시키는 방향"
    elif top["Mean(SHAP)"] < 0:
        direction_text = "평균적으로 예측값을 감소시키는 방향"
    else:
        direction_text = "평균적으로 증가/감소 방향성이 뚜렷하지 않은 상태"

    lines.append(
        f"Mean(SHAP)은 {top['Mean(SHAP)']:.4f}로, '{top['Feature']}'는 {direction_text}으로 작용했습니다."
    )

    rank_text = []
    for i, r in top_rows.iterrows():
        pct = 100 * r["Mean(|SHAP|)"] / total if total > 0 else 0
        direction = "증가" if r["Mean(SHAP)"] > 0 else "감소" if r["Mean(SHAP)"] < 0 else "중립"
        rank_text.append(
            f"{len(rank_text)+1}순위: {r['Feature']} "
            f"(중요도 비중 {pct:.1f}%, 방향성: {direction})"
        )

    lines.append("상위 변수 해석: " + " / ".join(rank_text))
    lines.append(
        "해석 시 주의할 점은 Mean(|SHAP|)는 영향력의 크기이고, Mean(SHAP)은 평균적인 방향성입니다. "
        "따라서 중요도가 높더라도 개별 샘플에서는 값의 범위와 다른 변수 조합에 따라 반대 방향으로 작용할 수 있습니다."
    )

    return pretty_time_text("<br><br>".join(lines))


def explain_feature_importance_detail(fi_df, target_name, model_choice):
    if fi_df is None or fi_df.empty:
        return "Feature Importance 상세 설명을 생성할 수 없습니다."

    total = fi_df["Importance"].sum()
    top_rows = fi_df.head(5).copy()
    top = top_rows.iloc[0]

    lines = []
    lines.append(
        f"Feature Importance는 {model_choice} 모델이 '{target_name}'을 예측할 때 "
        "분기 또는 학습 과정에서 어떤 변수를 많이 활용했는지를 나타냅니다."
    )
    lines.append(
        f"가장 중요한 변수는 '{top['Feature']}'이며, 중요도 값은 {top['Importance']:.4f}입니다. "
        f"전체 중요도에서 차지하는 비중은 약 {100 * top['Importance'] / total if total > 0 else 0:.1f}%입니다."
    )

    rank_text = []
    for _, r in top_rows.iterrows():
        pct = 100 * r["Importance"] / total if total > 0 else 0
        rank_text.append(f"{r['Feature']}({pct:.1f}%)")

    lines.append("상위 중요 변수는 " + " → ".join(rank_text) + " 순서입니다.")
    lines.append(
        "Feature Importance는 변수의 사용 빈도와 예측 성능 기여를 보여주지만, 값이 커질 때 예측값이 증가하는지 감소하는지는 직접 알려주지 않습니다. "
        "방향성 해석은 SHAP, PDP, ICE, ALE 결과와 함께 보는 것이 적절합니다."
    )

    return "<br><br>".join(lines)


def explain_heatmap_detail(heatmap_df, temporal_df, target_name):
    if heatmap_df is None or heatmap_df.empty:
        return "Feature × Week Heatmap 상세 설명을 생성할 수 없습니다."

    idx = np.unravel_index(np.argmax(heatmap_df.values), heatmap_df.shape)
    best_feat = heatmap_df.index[idx[0]]
    best_week = int(heatmap_df.columns[idx[1]])
    best_val = heatmap_df.iloc[idx[0], idx[1]]

    week_sum = heatmap_df.sum(axis=0).sort_values(ascending=False)
    feat_sum = heatmap_df.sum(axis=1).sort_values(ascending=False)

    lines = []
    lines.append(
        f"Feature × Week Heatmap은 '{target_name}' 예측에서 환경 변수와 시간 구간이 결합되어 "
        "어느 시점의 어떤 환경이 가장 중요했는지를 보여줍니다."
    )
    lines.append(
        f"가장 강한 조합은 '{best_feat}' × '{best_week}주 전'이며, Mean(|SHAP|)는 {best_val:.4f}입니다. "
        f"이는 {best_week}주 전의 {best_feat} 변화가 현재 예측 대상에 가장 크게 반영되었다는 의미입니다."
    )
    lines.append(
        f"주차 기준으로는 {int(week_sum.index[0])}주 전의 총 영향도가 가장 큽니다. "
        f"변수 기준으로는 '{feat_sum.index[0]}'의 누적 영향도가 가장 큽니다."
    )
    lines.append(
        "이 결과는 환경제어 전략을 세울 때 '언제 제어해야 하는가'와 '어떤 환경변수를 우선 관리해야 하는가'를 동시에 판단하는 데 활용할 수 있습니다."
    )

    return "<br><br>".join(lines)


# explain_shap_sample_intro 함수 삭제됨

# explain_sample_index 함수 삭제됨

# explain_shap_sample_result 함수 삭제됨

def explain_ice_pdp_result(feature, mean_slope, std_slope, pdp_summary, target_name="예측 대상", report_target=None):
    if report_target is not None:
        target_name = report_target
    feature_display = pretty_time_text(feature)
    start, end = pdp_summary["best_interval"]
    slope_direction = "증가" if mean_slope > 0 else "감소" if mean_slope < 0 else "거의 변화 없음"

    ice_text = f"""
<div style="background:linear-gradient(135deg,#ffffff,#eef5ff); box-shadow:0 6px 20px rgba(0,0,0,0.06); padding:18px; border-radius:14px; line-height:1.75; font-size:15px; word-break:keep-all; overflow-wrap:break-word;">
<b>ICE 그래프 해석</b><br><br>
ICE 곡선은 개별 샘플마다 <b>{feature_display}</b> 값이 변할 때 예측대상값(<b>{target_name}</b>)이 어떻게 달라지는지 보여줍니다.<br><br>
<table style="width:100%; border-collapse:collapse; font-size:14px;">
<thead><tr style="border-bottom:2px solid #d1d5db;"><th style="text-align:left; padding:8px;">항목</th><th style="text-align:left; padding:8px;">결과 해석</th></tr></thead>
<tbody>
<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:8px;">선택 Feature</td><td style="padding:8px;">{feature_display}</td></tr>
<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:8px;">평균 기울기</td><td style="padding:8px;">{mean_slope:.4f} ± {std_slope:.4f}</td></tr>
<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:8px;">전체 경향</td><td style="padding:8px;">예측값({target_name})이 전반적으로 <b>{slope_direction}</b>하는 경향</td></tr>
<tr><td style="padding:8px;">해석 포인트</td><td style="padding:8px;">곡선들이 서로 많이 벌어져 있으면 개체별 또는 조사일자별 반응 차이가 크다는 의미입니다.</td></tr>
</tbody></table>
</div>
"""

    pdp_text = f"""
<div style="background:linear-gradient(135deg,#ffffff,#f8fbff); box-shadow:0 6px 20px rgba(0,0,0,0.06); padding:18px; border-radius:14px; line-height:1.75; font-size:15px; word-break:keep-all; overflow-wrap:break-word;">
<b>PDP 그래프 해석</b><br><br>
PDP는 모든 샘플에 대해 <b>{feature_display}</b>만 변화시켰을 때 평균 예측값(<b>{target_name} 평균값</b>)이 어떻게 변하는지 보여줍니다.<br><br>
<table style="width:100%; border-collapse:collapse; font-size:14px;">
<thead><tr style="border-bottom:2px solid #d1d5db;"><th style="text-align:left; padding:8px;">항목</th><th style="text-align:left; padding:8px;">결과</th></tr></thead>
<tbody>
<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:8px;">최적 구간</td><td style="padding:8px;"><b>{start:.3f} ~ {end:.3f}</b></td></tr>
<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:8px;">구간 평균 예측값</td><td style="padding:8px;">{pdp_summary['mean_val']:.3f}</td></tr>
<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:8px;">구간 최대 예측값</td><td style="padding:8px;">{pdp_summary['max_val']:.3f}</td></tr>
<tr><td style="padding:8px;">해석 포인트</td><td style="padding:8px;">PDP 최적 구간은 전체 평균 반응 기준으로 예측대상({target_name})이 높게 나타나는 관리 후보 구간입니다.</td></tr>
</tbody></table>
</div>
"""

    combined_text = f"""
<div style="background:linear-gradient(135deg,#ffffff,#eef5ff); box-shadow:0 6px 20px rgba(0,0,0,0.06); padding:18px; border-radius:14px; line-height:1.75; font-size:15px; word-break:keep-all; overflow-wrap:break-word;">
<b>ICE + PDP 통합 해석</b><br><br>
<table style="width:100%; border-collapse:collapse; font-size:14px;">
<thead><tr style="border-bottom:2px solid #d1d5db;"><th style="text-align:left; padding:8px;">구분</th><th style="text-align:left; padding:8px;">의미</th><th style="text-align:left; padding:8px;">의사결정 활용</th></tr></thead>
<tbody>
<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:8px;">ICE</td><td style="padding:8px;">개별 샘플의 반응</td><td style="padding:8px;">개체별·조사일자별 반응 차이 확인</td></tr>
<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:8px;">PDP</td><td style="padding:8px;">전체 평균 반응</td><td style="padding:8px;">평균적으로 유리한 제어 구간 파악</td></tr>
<tr><td style="padding:8px;">종합 판단</td><td style="padding:8px;">PDP가 상승하더라도 ICE 편차가 크면 일괄 제어보다 단계적 제어가 적합</td><td style="padding:8px;">최적 구간 {start:.3f} ~ {end:.3f}을 기준으로 현장 조건과 함께 검토</td></tr>
</tbody></table>
</div>
"""

    return ice_text, pdp_text, combined_text
def explain_centered_ale_result(feature, bin_centers, ale_vals, ale_summary, target_name="예측 대상", report_target=None):
    if report_target is not None:
        target_name = report_target
    if bin_centers is None or len(bin_centers) == 0:
        return "Centered ALE 결과를 설명할 수 없습니다."

    bin_centers = np.array(bin_centers, dtype=float)
    ale_vals = np.array(ale_vals, dtype=float)

    max_idx = int(np.argmax(ale_vals))
    min_idx = int(np.argmin(ale_vals))

    pos_mask = ale_vals > 0
    near_zero_mask = np.abs(ale_vals) <= max(0.05, 0.05 * (np.nanmax(np.abs(ale_vals)) + 1e-9))
    neg_mask = ale_vals < 0

    def get_range(mask):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None
        return float(bin_centers[idxs[0]]), float(bin_centers[idxs[-1]])

    pos_range = get_range(pos_mask)
    neg_range = get_range(neg_mask)
    zero_range = get_range(near_zero_mask)

    if pos_range is not None:
        pos_start, pos_end = pos_range
        first_sentence = (
            f"<b>{pos_start:.2f}~{pos_end:.2f}</b> 구간이 "
            f"<b>{target_name} 증가에 유리한 구간</b>으로 해석할 수 있습니다.<br>"
            f"다만 “값이 계속 높을수록 계속 증가한다”는 뜻은 아닙니다."
        )
    else:
        first_sentence = (
            f"현재 Centered ALE 결과에서는 <b>{target_name} 증가에 유리한 양의 ALE 구간</b>이 명확하지 않습니다."
        )

    if neg_range is not None:
        neg_text = f"약 {neg_range[0]:.2f}~{neg_range[1]:.2f}"
    else:
        neg_text = "뚜렷하지 않음"

    if zero_range is not None:
        zero_text = f"약 {zero_range[0]:.2f}~{zero_range[1]:.2f}"
    else:
        zero_text = "0 근처 구간이 뚜렷하지 않음"

    if pos_range is not None:
        pos_text = f"약 {pos_range[0]:.2f} 이상"
        pos_full = f"약 {pos_range[0]:.2f}~{pos_range[1]:.2f}"
    else:
        pos_text = "뚜렷하지 않음"
        pos_full = "뚜렷하지 않음"

    best_text = f"약 {bin_centers[max_idx]:.2f}"
    high_effect_text = "여전히 양수이나 최고점보다 낮음 → 증가 효과는 유지되나 약해짐"

    return f"""
<div style="background:linear-gradient(135deg,#ffffff,#eef5ff); 
            box-shadow:0 6px 20px rgba(0,0,0,0.06);
            padding:18px;
            border-radius:14px;
            line-height:1.85;
            font-size:16px;
            word-break:keep-all;
            overflow-wrap:break-word;">

{first_sentence}<br><br>

<b>그래프 해석은 다음과 같습니다.</b><br><br>

<b>Centered ALE = 0</b>은 평균적인 기준선입니다.<br>
0보다 아래는 해당 구간이 <b>{target_name} 예측을 평균보다 낮추는 방향</b>, 
0보다 위는 <b>평균보다 높이는 방향</b>으로 작용했다는 의미입니다.<br><br>

<b>그래프를 보면:</b><br><br>

<table style="width:100%; border-collapse:collapse; font-size:15px;">
<thead>
<tr style="border-bottom:2px solid #d1d5db;">
<th style="text-align:left; padding:8px;">{feature}</th>
<th style="text-align:left; padding:8px;">해석</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #e5e7eb;">
<td style="padding:8px;">{neg_text}</td>
<td style="padding:8px;">ALE가 음수 → {target_name} 예측 감소 구간</td>
</tr>
<tr style="border-bottom:1px solid #e5e7eb;">
<td style="padding:8px;">{zero_text}</td>
<td style="padding:8px;">0 근처 → 영향이 거의 중립</td>
</tr>
<tr style="border-bottom:1px solid #e5e7eb;">
<td style="padding:8px;">{pos_text}</td>
<td style="padding:8px;">ALE가 양수 → {target_name} 예측 증가 구간</td>
</tr>
<tr style="border-bottom:1px solid #e5e7eb;">
<td style="padding:8px;">{best_text} 부근</td>
<td style="padding:8px;">ALE가 가장 높음 → 가장 유리한 구간</td>
</tr>
<tr>
<td style="padding:8px;">양의 구간 후반부</td>
<td style="padding:8px;">{high_effect_text}</td>
</tr>
</tbody>
</table>

<br>
따라서 <b>{pos_full} 전체를 “{target_name} 증가구간”이라고 말할 수는 있지만</b>, 더 정확히는:<br><br>

<div style="border-left:5px solid #cbd5e1; padding-left:14px; font-weight:600;">
{feature}가 약 {pos_range[0]:.2f} 이상일 때 {target_name} 예측에 긍정적으로 작용했으며, 
특히 약 {bin_centers[max_idx]:.2f} 부근에서 가장 큰 증가 효과가 나타났다.
</div>

</div>
"""



def _counterfactual_recommendation(feature_name, change):
    """환경변수별 제어 방향 설명"""
    fname = str(feature_name)
    up = change > 0

    if "일사" in fname or "광" in fname:
        return (
            "보광등 점등, 차광 시간 축소, 피복재 세척, 광 투과율 개선"
            if up else
            "차광 스크린 활용, 환기·냉방 강화, 엽온 상승 억제"
        )
    if "주간온도" in fname:
        return (
            "난방·보온 강화 또는 환기 지연으로 주간온도 확보"
            if up else
            "천창 개방, 팬 가동, 차광·냉방으로 주간 고온 완화"
        )
    if "야간온도" in fname:
        return (
            "야간 보온, 난방, 보온커튼 활용"
            if up else
            "야간 환기, 보온커튼 개폐 조정, 과도한 야간온도 완화"
        )
    if "수분부족분" in fname or "humidity_deficit" in fname.lower():
        return (
            "환기·난방·제습으로 공기 수분부족분을 높이고 결로를 완화"
            if up else
            "포그·가습·관수 점검으로 과도한 건조와 높은 수분부족분을 완화"
        )
    if "습도" in fname:
        return (
            "가습, 관수량 점검, 과도한 환기 완화"
            if up else
            "환기, 제습, 난방, 순환팬으로 결로·과습 완화"
        )
    if "CO₂" in fname or "CO2" in fname:
        return (
            "주간 CO₂ 시비, 환기 타이밍 조정, 광량과 함께 관리"
            if up else
            "CO₂ 공급량 축소, 환기 강화, 과다 시비 점검"
        )
    return "현장 조건과 생육단계를 함께 검토하여 제어 방향 결정"


def _counterfactual_control_group(feature_name):
    fname = str(feature_name)
    if "일사" in fname or "광" in fname:
        return "광환경"
    if "온도" in fname:
        return "온도"
    if "수분부족분" in fname or "humidity_deficit" in fname.lower():
        return "수분부족분"
    if "습도" in fname:
        return "습도/VPD"
    if "CO₂" in fname or "CO2" in fname:
        return "CO₂"
    return "기타"


# generate_counterfactual 함수 삭제됨

def classify_environment_zone(feature_name, value):
    """
    환경변수를 최저한계구간, 저구간, 중간구간, 최적구간, 고구간, 최고한계구간으로 분류합니다.
    평균주간온도와 평균야간온도는 서로 다른 기준을 적용합니다.
    기준은 교육용 기본값이며 작물·품종·생육단계에 따라 조정할 수 있습니다.
    """
    if pd.isna(value):
        return "데이터없음", "값이 없어 구간을 판단할 수 없습니다."

    fname = str(feature_name)

    # 평균주간온도 기준
    if "주간온도" in fname:
        if value < 15:
            return "최저한계구간", "15℃ 미만: 저온 한계로 생육 정지, 양분 흡수 저하, 저온장해 위험이 큽니다."
        elif value < 20:
            return "저온구간", "15~20℃: 생육은 가능하지만 광합성·과실 비대가 둔화될 수 있습니다."
        elif value < 24:
            return "최적구간", "20~24℃: 주간 광합성과 증산 균형이 좋아 생육·수확 형성에 유리합니다."
        elif value < 30:
            return "고온구간", "24~30℃: 생육은 가능하나 증산·호흡 증가로 환기와 수분 관리가 중요합니다."
        else:
            return "최고한계구간", "30℃ 이상: 고온 스트레스, 착과 불량, 품질 저하, 엽온 상승 위험이 큽니다."

    # 평균야간온도 기준
    if "야간온도" in fname:
        if value < 10:
            return "최저한계구간", "10℃ 미만: 야간 저온 한계로 생육 정지와 저온장해 위험이 큽니다."
        elif value < 15:
            return "저온구간", "10~15℃: 호흡은 줄지만 생육 회복과 양분 이동이 둔화될 수 있습니다."
        elif value < 18:
            return "최적구간", "15~18℃: 야간 호흡과 당 소모가 적절하여 생육 균형 유지에 유리합니다."
        elif value < 23:
            return "고온구간", "18~23℃: 야간 호흡량 증가로 당 소모가 커질 수 있습니다."
        else:
            return "최고한계구간", "23℃ 이상: 야간 고온으로 호흡 과다, 생장 불균형, 품질 저하 위험이 있습니다."

    # 기타 온도 기준
    if "온도" in fname:
        if value < 15:
            return "최저한계구간", "15℃ 미만: 저온 한계로 생육 정지와 저온장해 위험이 있습니다."
        elif value < 20:
            return "저온구간", "15~20℃: 생육은 가능하지만 광합성·양분 흡수·과실 비대가 둔화될 수 있습니다."
        elif value < 24:
            return "최적구간", "20~24℃: 광합성과 호흡 균형이 좋아 안정적 생육에 유리합니다."
        elif value < 30:
            return "고온구간", "24~30℃: 생육은 가능하나 증산과 호흡 증가로 수분·환기 관리가 중요합니다."
        else:
            return "최고한계구간", "30℃ 이상: 고온 스트레스, 착과 불량, 품질 저하 위험이 증가합니다."

    # 수분부족분(HD) 기준
    # 기본 단위는 g/m³를 가정하며, 센서 단위가 다르면 기준을 조정해야 합니다.
    if "수분부족분" in fname or "humidity_deficit" in fname.lower():
        if value < 2:
            return (
                "최저한계구간",
                "2 g/m³ 미만: 공기 중 수분부족이 매우 작아 과습·결로·병해 위험이 커질 수 있습니다.",
            )
        elif value < 4:
            return (
                "낮은부족분구간",
                "2~4 g/m³: 다소 습한 상태로 야간 결로와 병해 위험을 함께 확인해야 합니다.",
            )
        elif value < 8:
            return (
                "최적구간",
                "4~8 g/m³: 증산과 수분흡수 균형이 비교적 안정적인 관리 후보 구간입니다.",
            )
        elif value < 12:
            return (
                "높은부족분구간",
                "8~12 g/m³: 건조 경향으로 증산량 증가와 수분 스트레스 가능성이 있습니다.",
            )
        else:
            return (
                "최고한계구간",
                "12 g/m³ 이상: 과도한 건조로 기공 폐쇄·위조·생육 저하 위험이 커질 수 있습니다.",
            )

    # 습도
    if "습도" in fname:
        if value < 40:
            return "최저한계구간", "40% 미만: 극건조로 VPD가 과도하게 높아져 위조·수분 스트레스 위험이 큽니다."
        elif value < 60:
            return "저습구간", "40~60%: 건조 경향으로 증산 과다와 생육 지연이 발생할 수 있습니다."
        elif value < 80:
            return "최적구간", "60~80%: 증산과 기공 조절이 안정적인 적정 습도 구간입니다."
        elif value < 90:
            return "다습구간", "80~90%: 야간 결로와 병해 위험이 증가할 수 있습니다."
        else:
            return "최고한계구간", "90% 이상: 과습·결로·병해 확산 위험이 높아 제습과 환기가 필요합니다."

    # CO2
    if "CO₂" in fname or "CO2" in fname:
        if value < 350:
            return "최저한계구간", "350ppm 미만: CO₂ 부족으로 광합성이 제한될 수 있습니다."
        elif value < 400:
            return "저농도구간", "350~400ppm: 외기 수준 이하로 광합성 원료가 부족할 수 있습니다."
        elif value < 800:
            return "중간구간", "400~800ppm: 일반 생육은 가능하나 적극적인 CO₂ 시비 효과는 제한적일 수 있습니다."
        elif value < 1200:
            return "최적구간", "800~1200ppm: 충분한 광·온도 조건에서 광합성 촉진에 유리합니다."
        else:
            return "최고한계구간", "1200ppm 이상: 과다 공급 또는 환기 부족 가능성이 있어 공급량과 환기를 점검해야 합니다."

    # 일사량
    if "일사" in fname or "광" in fname:
        if value < 300:
            return "최저한계구간", "매우 낮은 일사량: 광합성량과 동화산물 생산이 크게 부족할 수 있습니다."
        elif value < 500:
            return "저광구간", "낮은 일사량: 생육단계에 따라 보광이 필요할 수 있습니다."
        elif value < 1200:
            return "중간구간", "중간 일사량: 기본 광합성은 가능하나 생육단계에 따라 보광 판단이 필요합니다."
        elif value < 2000:
            return "최적구간", "충분한 일사량: 광합성과 당 생산에 유리하여 생육·수확 증가에 긍정적입니다."
        else:
            return "최고한계구간", "매우 높은 일사량: 고온·건조·엽온 상승이 동반될 수 있어 차광·냉방 관리가 필요합니다."

    return "기타", "해당 변수는 별도 기준 설정 후 해석하는 것이 좋습니다."

def build_monthly_environment_zone_table(df, date_col, feature_name, value_col):
    temp_df = df[[date_col, value_col]].copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
    temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors="coerce")
    temp_df = temp_df.dropna(subset=[date_col])

    if temp_df.empty:
        return pd.DataFrame(columns=["월", "평균값", "환경구간", "영향 설명"])

    temp_df["월"] = temp_df[date_col].dt.to_period("M").astype(str)

    monthly = (
        temp_df
        .groupby("월", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "평균값"})
    )

    zones = monthly["평균값"].apply(lambda v: classify_environment_zone(feature_name, v))
    monthly["환경구간"] = [z[0] for z in zones]
    monthly["영향 설명"] = [z[1] for z in zones]

    return monthly.round({"평균값": 3})


def environment_zone_reference_table(feature_name):
    rows = []
    fname = str(feature_name)

    if "주간온도" in fname:
        rows = [
            ["최저한계구간", "< 15℃", "생육 정지·저온장해 위험, 보온/난방 필요"],
            ["저온구간", "15~20℃", "광합성·과실 비대 둔화 가능"],
            ["최적구간", "20~24℃", "주간 광합성·증산 균형에 유리"],
            ["고온구간", "24~30℃", "증산·호흡 증가, 환기·수분 관리 필요"],
            ["최고한계구간", "≥ 30℃", "고온 스트레스·착과 불량·품질 저하 위험"],
        ]
    elif "야간온도" in fname:
        rows = [
            ["최저한계구간", "< 10℃", "야간 저온장해·생육 정지 위험, 보온 필요"],
            ["저온구간", "10~15℃", "양분 이동과 생육 회복 둔화 가능"],
            ["최적구간", "15~18℃", "호흡과 당 소모 균형에 유리"],
            ["고온구간", "18~23℃", "야간 호흡 증가로 당 소모 확대"],
            ["최고한계구간", "≥ 23℃", "야간 고온 스트레스·생장 불균형 위험"],
        ]
    elif "온도" in fname:
        rows = [
            ["최저한계구간", "< 15℃", "저온장해·생육 정지 위험"],
            ["저온구간", "15~20℃", "광합성·양분 흡수 둔화 가능"],
            ["최적구간", "20~24℃", "광합성·호흡 균형이 좋아 안정적 생육에 유리"],
            ["고온구간", "24~30℃", "증산·호흡 증가, 환기·수분 관리 필요"],
            ["최고한계구간", "≥ 30℃", "고온 스트레스, 착과 불량, 품질 저하 위험"],
        ]
    elif "수분부족분" in fname or "humidity_deficit" in fname.lower():
        rows = [
            ["최저한계구간", "< 2 g/m³", "수분부족이 매우 작아 과습·결로·병해 위험"],
            ["낮은부족분구간", "2~4 g/m³", "다소 습한 상태, 야간 결로 여부 점검"],
            ["최적구간", "4~8 g/m³", "증산과 수분흡수 균형이 비교적 안정적"],
            ["높은부족분구간", "8~12 g/m³", "건조 경향, 관수·가습 및 작물 수분상태 점검"],
            ["최고한계구간", "≥ 12 g/m³", "과도한 건조, 기공 폐쇄·위조 위험"],
        ]
    elif "습도" in fname:
        rows = [
            ["최저한계구간", "< 40%", "극건조, VPD 과다, 위조 위험"],
            ["저습구간", "40~60%", "건조 스트레스, 증산 과다 가능"],
            ["최적구간", "60~80%", "증산과 기공 조절이 안정적인 구간"],
            ["다습구간", "80~90%", "결로·병해 위험 증가"],
            ["최고한계구간", "≥ 90%", "과습·결로·병해 확산 위험"],
        ]
    elif "CO₂" in fname or "CO2" in fname:
        rows = [
            ["최저한계구간", "< 350ppm", "CO₂ 부족으로 광합성 제한 가능"],
            ["저농도구간", "350~400ppm", "외기 수준 이하, CO₂ 보충 검토"],
            ["중간구간", "400~800ppm", "일반 생육 가능, 시비 효과 제한적"],
            ["최적구간", "800~1200ppm", "광합성 촉진에 유리"],
            ["최고한계구간", "≥ 1200ppm", "과다 공급 또는 환기 부족 가능성"],
        ]
    elif "일사" in fname or "광" in fname:
        rows = [
            ["최저한계구간", "< 300", "광 부족, 동화산물 생산 크게 감소"],
            ["저광구간", "300~500", "보광 검토 필요"],
            ["중간구간", "500~1200", "기본 광합성 가능, 생육단계별 관리 필요"],
            ["최적구간", "1200~2000", "광합성과 당 생산에 유리"],
            ["최고한계구간", "≥ 2000", "고온·건조·엽온 상승 위험, 차광/냉방 검토"],
        ]
    else:
        rows = [["기타", "사용자 정의", "작물·생육단계에 맞는 기준 설정 필요"]]

    return pd.DataFrame(rows, columns=["환경구간", "기준", "영향"])

def compute_rolling_summary(
    sensor_df,
    yield_df,
    date_col_sensor,
    date_col_yield,
    temp_col=None,
    hum_col=None,
    co2_col=None,
    solar_col=None,
    moisture_deficit_col=None,
    harvest_count_col=None,
    harvest_weight_col=None,
    flower_count_col=None,
    avg_fruit_weight_col=None,
    avg_fruit_weight_col2=None,
    avg_fruit_weight_label="평균과중1",
    avg_fruit_weight_label2="평균과중2",
    growth_cols=None,
    week=1,
    window_days=None,
):
    """
    조사일 기준 과거 N주 환경요약과 수확·생육 데이터를 통합합니다.

    선택 컬럼이 None인 환경변수는 결과에서 제외합니다.
    수분부족분은 하루 평균을 계산한 뒤 분석기간의 일평균으로 요약합니다.

    v26.1 파생변수
    - ADT: 각 일자의 24시간 평균온도. 10분 간격 정상자료라면 144개 값의 평균.
    - DIF: 각 일자의 주간(08~18시) 평균온도 - 야간(19~07시) 평균온도.
    - GDD: Base 10℃, 일별 max(ADT - 10, 0)를 분석기간 동안 누적.
    - VPD: 각 10분 시점의 온도·RH로 계산 후 일평균, 이후 분석기간 평균.
    """
    growth_cols = growth_cols or {}
    days = int(window_days) if window_days is not None else int(week) * 7
    window_prefix = f"{days}일" if window_days is not None else f"{week}주"

    temp_day_col_name = f"{window_prefix}평균주간온도(08~18시)"
    temp_night_col_name = f"{window_prefix}평균야간온도(19~07시)"
    hum_day_col_name = f"{window_prefix}평균주간습도(08~18시)"
    hum_night_col_name = f"{window_prefix}평균야간습도(19~07시)"
    co2_day_col_name = f"{window_prefix}평균주간CO₂(08~18시)"
    co2_night_col_name = f"{window_prefix}평균야간CO₂(19~07시)"
    solar_col_name = f"{window_prefix}평균누적일사량(1일최대값기준)"
    moisture_deficit_col_name = f"{window_prefix}평균수분부족분(24시간)"

    adt_col_name = f"{window_prefix}ADT(24시간평균온도)"
    dif_col_name = f"{window_prefix}DIF(주간08~18-야간19~07)"
    gdd_col_name = f"{window_prefix}GDD(Base10℃누적)"
    vpd_col_name = f"{window_prefix}VPD(24시간평균)"

    results = []
    sensor_dates = pd.to_datetime(
        sensor_df[date_col_sensor],
        errors="coerce",
    )

    for _, row in yield_df.iterrows():
        survey_date = row[date_col_yield]
        start_date = survey_date - timedelta(days=days)

        mask = (
            (sensor_dates >= start_date)
            & (sensor_dates <= survey_date)
        )
        subset = sensor_df.loc[mask].copy()

        avg_values = {
            temp_day_col_name: np.nan,
            temp_night_col_name: np.nan,
            hum_day_col_name: np.nan,
            hum_night_col_name: np.nan,
            co2_day_col_name: np.nan,
            co2_night_col_name: np.nan,
            solar_col_name: np.nan,
            moisture_deficit_col_name: np.nan,
            adt_col_name: np.nan,
            dif_col_name: np.nan,
            gdd_col_name: np.nan,
            vpd_col_name: np.nan,
        }

        if not subset.empty:
            daytime_mask = (
                (subset["hour"] >= 8)
                & (subset["hour"] <= 18)
            )
            nighttime_mask = (
                (subset["hour"] >= 19)
                | (subset["hour"] <= 7)
            )

            if solar_col is not None and solar_col in subset.columns:
                daily_max_solar = (
                    subset.groupby("date")[solar_col]
                    .max()
                    .dropna()
                )
                if not daily_max_solar.empty:
                    avg_values[solar_col_name] = float(
                        pd.to_numeric(
                            daily_max_solar,
                            errors="coerce",
                        ).mean()
                    )

            if co2_col is not None and co2_col in subset.columns:
                co2_daytime = subset.loc[daytime_mask]
                co2_nighttime = subset.loc[nighttime_mask]

                if not co2_daytime.empty:
                    daily_co2_day = (
                        co2_daytime.groupby("date")[co2_col]
                        .mean()
                        .dropna()
                    )
                    if not daily_co2_day.empty:
                        avg_values[co2_day_col_name] = float(
                            pd.to_numeric(
                                daily_co2_day,
                                errors="coerce",
                            ).mean()
                        )

                if not co2_nighttime.empty:
                    daily_co2_night = (
                        co2_nighttime.groupby("date")[co2_col]
                        .mean()
                        .dropna()
                    )
                    if not daily_co2_night.empty:
                        avg_values[co2_night_col_name] = float(
                            pd.to_numeric(
                                daily_co2_night,
                                errors="coerce",
                            ).mean()
                        )

            if temp_col is not None and temp_col in subset.columns:
                temp_daytime = subset.loc[daytime_mask]
                temp_nighttime = subset.loc[nighttime_mask]

                if not temp_daytime.empty:
                    avg_values[temp_day_col_name] = float(
                        pd.to_numeric(
                            temp_daytime[temp_col],
                            errors="coerce",
                        ).mean()
                    )
                if not temp_nighttime.empty:
                    avg_values[temp_night_col_name] = float(
                        pd.to_numeric(
                            temp_nighttime[temp_col],
                            errors="coerce",
                        ).mean()
                    )

                # -------------------------------------------------
                # v26.1 파생변수: ADT / DIF / GDD
                # -------------------------------------------------
                temp_numeric = pd.to_numeric(
                    subset[temp_col],
                    errors="coerce",
                )

                temp_daily_source = subset[["date"]].copy()
                temp_daily_source["_temp_numeric"] = temp_numeric

                # ADT: 하루 24시간 온도 평균.
                # 10분 간격 자료가 완전하면 하루 144개 관측값 평균이 됩니다.
                daily_adt = (
                    temp_daily_source
                    .groupby("date")["_temp_numeric"]
                    .mean()
                    .dropna()
                )
                if not daily_adt.empty:
                    avg_values[adt_col_name] = float(daily_adt.mean())

                    # GDD: Base 10℃ 이상의 일평균 유효온도를 누적합니다.
                    daily_gdd = (daily_adt - 10.0).clip(lower=0.0)
                    avg_values[gdd_col_name] = float(daily_gdd.sum())

                # DIF: 일별 주간평균 - 야간평균을 먼저 만든 후
                # 선택한 1~7주 기간에서 그 일별 DIF의 평균을 사용합니다.
                day_temp_daily = (
                    temp_daytime.groupby("date")[temp_col]
                    .mean()
                    .apply(pd.to_numeric, errors="coerce")
                    .dropna()
                    if not temp_daytime.empty
                    else pd.Series(dtype=float)
                )
                night_temp_daily = (
                    temp_nighttime.groupby("date")[temp_col]
                    .mean()
                    .apply(pd.to_numeric, errors="coerce")
                    .dropna()
                    if not temp_nighttime.empty
                    else pd.Series(dtype=float)
                )
                if not day_temp_daily.empty and not night_temp_daily.empty:
                    dif_daily = (
                        pd.concat(
                            [
                                day_temp_daily.rename("day"),
                                night_temp_daily.rename("night"),
                            ],
                            axis=1,
                            join="inner",
                        )
                        .dropna()
                    )
                    if not dif_daily.empty:
                        avg_values[dif_col_name] = float(
                            (dif_daily["day"] - dif_daily["night"]).mean()
                        )

            if hum_col is not None and hum_col in subset.columns:
                hum_daytime = subset.loc[daytime_mask]
                hum_nighttime = subset.loc[nighttime_mask]

                if not hum_daytime.empty:
                    avg_values[hum_day_col_name] = float(
                        pd.to_numeric(
                            hum_daytime[hum_col],
                            errors="coerce",
                        ).mean()
                    )
                if not hum_nighttime.empty:
                    avg_values[hum_night_col_name] = float(
                        pd.to_numeric(
                            hum_nighttime[hum_col],
                            errors="coerce",
                        ).mean()
                    )

            # -----------------------------------------------------
            # v26.1 파생변수: VPD
            # 온도와 상대습도가 모두 선택된 경우 10분 시점별 계산
            # -----------------------------------------------------
            if (
                temp_col is not None
                and hum_col is not None
                and temp_col in subset.columns
                and hum_col in subset.columns
            ):
                vpd_temp = pd.to_numeric(
                    subset[temp_col],
                    errors="coerce",
                )
                vpd_rh = pd.to_numeric(
                    subset[hum_col],
                    errors="coerce",
                ).clip(lower=0.0, upper=100.0)

                # Tetens 식: 포화수증기압(kPa)
                svp = 0.6108 * np.exp(
                    (17.27 * vpd_temp) / (vpd_temp + 237.3)
                )
                vpd_values = svp * (1.0 - vpd_rh / 100.0)
                vpd_values = (
                    pd.Series(vpd_values, index=subset.index)
                    .replace([np.inf, -np.inf], np.nan)
                    .clip(lower=0.0)
                )

                vpd_daily_source = subset[["date"]].copy()
                vpd_daily_source["_vpd"] = vpd_values
                daily_vpd = (
                    vpd_daily_source
                    .groupby("date")["_vpd"]
                    .mean()
                    .dropna()
                )
                if not daily_vpd.empty:
                    avg_values[vpd_col_name] = float(daily_vpd.mean())

            if (
                moisture_deficit_col is not None
                and moisture_deficit_col in subset.columns
            ):
                daily_moisture_deficit = (
                    subset.groupby("date")[moisture_deficit_col]
                    .mean()
                    .dropna()
                )
                if not daily_moisture_deficit.empty:
                    avg_values[moisture_deficit_col_name] = float(
                        pd.to_numeric(
                            daily_moisture_deficit,
                            errors="coerce",
                        ).mean()
                    )

        result_row = {
            "조사일자": survey_date,
            "수확수": (
                row.get(harvest_count_col, np.nan)
                if harvest_count_col is not None
                else np.nan
            ),
            "착과수": (
                row.get(harvest_weight_col, np.nan)
                if harvest_weight_col is not None
                else np.nan
            ),
            "개화수": (
                row.get(flower_count_col, np.nan)
                if flower_count_col is not None
                else np.nan
            ),
            avg_fruit_weight_label: (
                row.get(avg_fruit_weight_col, np.nan)
                if avg_fruit_weight_col is not None
                else np.nan
            ),
            avg_fruit_weight_label2: (
                row.get(avg_fruit_weight_col2, np.nan)
                if avg_fruit_weight_col2 is not None
                else np.nan
            ),
        }

        # 실제로 선택한 환경변수만 매핑데이터에 추가합니다.
        if temp_col is not None:
            result_row[temp_day_col_name] = avg_values[
                temp_day_col_name
            ]
            result_row[temp_night_col_name] = avg_values[
                temp_night_col_name
            ]

        if hum_col is not None:
            result_row[hum_day_col_name] = avg_values[
                hum_day_col_name
            ]
            result_row[hum_night_col_name] = avg_values[
                hum_night_col_name
            ]

        if co2_col is not None:
            result_row[co2_day_col_name] = avg_values[
                co2_day_col_name
            ]
            result_row[co2_night_col_name] = avg_values[
                co2_night_col_name
            ]

        if solar_col is not None:
            result_row[solar_col_name] = avg_values[
                solar_col_name
            ]

        if moisture_deficit_col is not None:
            result_row[moisture_deficit_col_name] = avg_values[
                moisture_deficit_col_name
            ]

        # v26.1 파생환경 Feature는 필요한 원 센서가 선택된 경우에만
        # 매핑데이터에 추가합니다.
        if temp_col is not None:
            result_row[adt_col_name] = avg_values[adt_col_name]
            result_row[dif_col_name] = avg_values[dif_col_name]
            result_row[gdd_col_name] = avg_values[gdd_col_name]

        if temp_col is not None and hum_col is not None:
            result_row[vpd_col_name] = avg_values[vpd_col_name]

        for growth_feature, source_col in growth_cols.items():
            if source_col is not None and source_col in row.index:
                result_row[growth_feature] = row[source_col]
            else:
                result_row[growth_feature] = np.nan

        results.append(result_row)

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("조사일자")
        .reset_index(drop=True)
    )



def explain_environment_timeseries(feature_name, values):
    vals = pd.Series(values).dropna()
    if len(vals) == 0:
        return "데이터가 부족하여 시계열 설명을 생성할 수 없습니다."

    mean_v = vals.mean()
    min_v = vals.min()
    max_v = vals.max()
    std_v = vals.std()
    last_v = vals.iloc[-1]

    lines = [
        f"'{feature_name}' 시계열의 평균은 {mean_v:.2f}, 최소 {min_v:.2f}, 최대 {max_v:.2f}, 표준편차는 {std_v:.2f}입니다.",
        f"최근값은 {last_v:.2f}입니다."
    ]

    if "온도" in feature_name:
        if mean_v < 20:
            lines.append("20℃ 미만은 저온 구간으로 광합성·양분 흡수·과실 비대가 둔화될 수 있습니다.")
        elif mean_v < 24:
            lines.append("20~24℃는 비교적 적정 구간으로 광합성과 호흡 균형이 안정적입니다.")
        elif mean_v < 30:
            lines.append("24~30℃는 생육은 활발할 수 있으나 증산과 호흡이 증가하여 환기·수분 관리가 중요합니다.")
        else:
            lines.append("30℃ 이상은 고온 스트레스 위험 구간으로 착과 불량과 품질 저하가 발생할 수 있습니다.")
    elif "수분부족분" in feature_name or "humidity_deficit" in feature_name.lower():
        if mean_v < 2:
            lines.append(
                "수분부족분 2 g/m³ 미만은 과습·결로 위험이 높은 구간으로 "
                "환기·난방·제습 상태를 확인해야 합니다."
            )
        elif mean_v < 4:
            lines.append(
                "수분부족분 2~4 g/m³는 다소 습한 구간으로 "
                "야간 결로와 병해 발생 여부를 함께 확인하는 것이 좋습니다."
            )
        elif mean_v < 8:
            lines.append(
                "수분부족분 4~8 g/m³는 증산과 수분흡수 균형이 "
                "비교적 안정적인 관리 후보 구간입니다."
            )
        elif mean_v < 12:
            lines.append(
                "수분부족분 8~12 g/m³는 건조 경향이므로 "
                "관수·가습과 작물의 위조 여부를 점검해야 합니다."
            )
        else:
            lines.append(
                "수분부족분 12 g/m³ 이상은 과도한 건조 구간으로 "
                "기공 폐쇄·위조·생육 저하 위험이 커질 수 있습니다."
            )
        lines.append(
            "수분부족분 기준은 g/m³ 단위를 가정한 교육용 기본값이며, "
            "센서 단위와 작물 생육단계에 따라 조정해야 합니다."
        )
    elif "습도" in feature_name:
        if mean_v < 60:
            lines.append("60% 미만은 건조 구간으로 VPD 상승과 수분 스트레스 가능성이 있습니다.")
        elif mean_v < 80:
            lines.append("60~80%는 비교적 적정 습도 구간으로 증산 균형 유지에 유리합니다.")
        else:
            lines.append("80% 이상은 다습 구간으로 결로와 병해 위험이 증가할 수 있습니다.")
    elif "CO₂" in feature_name or "CO2" in feature_name:
        if mean_v < 400:
            lines.append("400ppm 미만은 CO₂ 부족 구간으로 광합성이 제한될 수 있습니다.")
        elif mean_v < 800:
            lines.append("400~800ppm은 일반적인 생육 가능 구간입니다.")
        elif mean_v < 1200:
            lines.append("800~1200ppm은 광합성 촉진에 유리한 구간입니다.")
        else:
            lines.append("1200ppm 이상은 과다 구간으로 환기 부족 또는 CO₂ 낭비 가능성이 있습니다.")
    elif "일사" in feature_name or "광" in feature_name:
        lines.append("일사량은 광합성 에너지 공급량과 연관되며 온도·CO₂·수분 상태와 함께 해석해야 합니다.")

    return "<br><br>".join(lines)


def generate_comprehensive_report(
    model_choice,
    target_col,
    metrics,
    weekly_metrics_df=None,
    shap_df=None,
    fi_df=None,
    week_importance=None,
    heatmap_df=None,
    cf_result=None,
    ice_feature=None,
    ice_mean_slope=None,
    ice_std_slope=None,
    pdp_summary=None,
    ale_summary=None,
    bin_centers=None,
    ale_vals=None,
):
    lines = []

    lines.append(
        f"<b>1. 분석 개요</b><br>"
        f"본 분석은 <b>{model_choice}</b> 모델을 이용하여 "
        f"<b>{target_col}</b>을 예측하고, 모델 성능과 XAI 결과를 종합적으로 해석한 리포트입니다."
    )

    if metrics is not None:
        lines.append(
            f"<b>2. 선택 주차 모델 성능</b><br>"
            f"현재 선택된 주차 기준 모델 성능은 "
            f"MSE=<b>{metrics.get('MSE', np.nan):.4f}</b>, "
            f"MAE=<b>{metrics.get('MAE', np.nan):.4f}</b>, "
            f"R²=<b>{metrics.get('R2', np.nan):.4f}</b>입니다. "
            "MSE와 MAE는 낮을수록 오차가 작고, R²는 높을수록 설명력이 높습니다."
        )

    if weekly_metrics_df is not None and not weekly_metrics_df.empty:
        best_r2 = weekly_metrics_df.sort_values("R2", ascending=False).iloc[0]
        best_mse = weekly_metrics_df.sort_values("MSE", ascending=True).iloc[0]
        best_mae = weekly_metrics_df.sort_values("MAE", ascending=True).iloc[0]

        lines.append(
            f"<b>3. 1~7주 모델 성능 비교</b><br>"
            f"1~7주 전체 비교에서 R²가 가장 높은 구간은 <b>{int(best_r2['Week'])}주</b> "
            f"(R²={best_r2['R2']:.4f})입니다. "
            f"MSE가 가장 낮은 구간은 <b>{int(best_mse['Week'])}주</b> "
            f"(MSE={best_mse['MSE']:.4f})이며, "
            f"MAE가 가장 낮은 구간은 <b>{int(best_mae['Week'])}주</b> "
            f"(MAE={best_mae['MAE']:.4f})입니다. "
            "따라서 성능 기준으로 어떤 기간의 환경 누적 정보가 예측에 가장 적합한지 판단할 수 있습니다."
        )

    if shap_df is not None and not shap_df.empty:
        top_shap = shap_df.iloc[0]
        direction = "증가" if top_shap["Mean(SHAP)"] > 0 else "감소" if top_shap["Mean(SHAP)"] < 0 else "중립"
        lines.append(
            f"<b>4. SHAP Summary 종합 해석</b><br>"
            f"SHAP 기준 가장 영향력이 큰 변수는 <b>{pretty_time_text(top_shap['Feature'])}</b>입니다. "
            f"Mean(|SHAP|)={top_shap['Mean(|SHAP|)']:.4f}, "
            f"Mean(SHAP)={top_shap['Mean(SHAP)']:.4f}로 나타났습니다. "
            f"이는 해당 변수가 예측값에 가장 크게 기여했으며, 평균적으로 예측값을 <b>{direction}</b>시키는 방향으로 작용했음을 의미합니다."
        )

    if fi_df is not None and not fi_df.empty:
        top_fi = fi_df.iloc[0]
        total_fi = fi_df["Importance"].sum()
        pct = 100 * top_fi["Importance"] / total_fi if total_fi > 0 else 0
        lines.append(
            f"<b>5. Feature Importance 종합 해석</b><br>"
            f"모델 기반 Feature Importance에서 가장 중요한 변수는 <b>{pretty_time_text(top_fi['Feature'])}</b>이며, "
            f"중요도는 {top_fi['Importance']:.4f}, 전체 중요도 비중은 약 {pct:.1f}%입니다. "
            "Feature Importance는 모델이 어떤 변수를 많이 활용했는지를 보여주며, 방향성은 SHAP과 함께 해석하는 것이 적절합니다."
        )

    if week_importance is not None and not week_importance.empty:
        best_week = week_importance.sort_values("TotalMeanAbsSHAP", ascending=False).iloc[0]
        signed = best_week["AvgSignedSHAP"]
        signed_text = "긍정적" if signed > 0 else "부정적" if signed < 0 else "중립적"
        lines.append(
            f"<b>6. Temporal SHAP 종합 해석</b><br>"
            f"시간 구간별 SHAP 분석 결과, 가장 영향력이 큰 시점은 <b>{int(best_week['Week'])}주 전</b>입니다. "
            f"이 구간의 TotalMeanAbsSHAP는 {best_week['TotalMeanAbsSHAP']:.4f}, "
            f"AvgSignedSHAP는 {signed:.4f}입니다. "
            f"이는 해당 시기의 환경조건이 예측에 가장 강하게 반영되었고, 평균 방향성은 <b>{signed_text}</b>으로 해석됨을 의미합니다."
        )

    if heatmap_df is not None and not heatmap_df.empty:
        idx = np.unravel_index(np.argmax(heatmap_df.values), heatmap_df.shape)
        best_feat = heatmap_df.index[idx[0]]
        best_week_hm = heatmap_df.columns[idx[1]]
        best_val = heatmap_df.iloc[idx[0], idx[1]]
        lines.append(
            f"<b>7. Feature × Week Heatmap 종합 해석</b><br>"
            f"변수와 주차 조합 중 가장 영향력이 큰 조합은 "
            f"<b>{pretty_time_text(best_feat)} × {int(best_week_hm)}주</b>이며, "
            f"Mean(|SHAP|)={best_val:.4f}입니다. "
            "이는 특정 변수 자체뿐 아니라, 해당 변수가 어느 시점에 누적되었는지가 예측에 중요하다는 것을 보여줍니다."
        )

    if cf_result is not None:
        delta = cf_result["cf_pred"] - cf_result["base_pred"]
        direction = "증가" if delta > 0 else "감소"
        lines.append(
            f"<b>8.  환경제어 시뮬레이션 해석</b><br>"
            f" 분석 결과 예측값은 {cf_result['base_pred']:.4f}에서 "
            f"{cf_result['cf_pred']:.4f}로 {direction}했습니다. "
            f"변화량은 {delta:.4f}입니다. "
            "이는 일부 제어 가능한 환경변수를 조정할 경우 예측 결과 개선 가능성이 있음을 의미합니다."
        )

    if ice_feature is not None and pdp_summary is not None and ice_mean_slope is not None:
        start, end = pdp_summary["best_interval"]
        slope_dir = "증가" if ice_mean_slope > 0 else "감소" if ice_mean_slope < 0 else "변화가 작음"
        lines.append(
            f"<b>9. ICE + PDP 통합 그래프 해석</b><br>"
            f"선택 Feature는 <b>{pretty_time_text(ice_feature)}</b>입니다. "
            f"ICE 평균 기울기는 {ice_mean_slope:.4f} ± {ice_std_slope:.4f}로, "
            f"개별 샘플 기준 예측값은 전반적으로 <b>{slope_dir}</b>하는 경향을 보입니다. "
            f"PDP 기준 예측이 높은 최적 구간은 <b>{start:.3f} ~ {end:.3f}</b>이며, "
            f"이 구간 평균 예측값은 {pdp_summary['mean_val']:.4f}입니다."
        )

    if ale_summary is not None and bin_centers is not None and ale_vals is not None and len(bin_centers) > 0:
        max_idx = int(np.argmax(ale_vals))
        min_idx = int(np.argmin(ale_vals))

        pos_text = "없음"
        neg_text = "없음"

        if ale_summary.get("pos_intervals"):
            pos_text = format_interval_text(ale_summary.get("pos_intervals", []), limit=3)

        if ale_summary.get("neg_intervals"):
            neg_text = format_interval_text(ale_summary.get("neg_intervals", []), limit=3)

        lines.append(
            f"<b>10. Centered ALE 종합 해석</b><br>"
            f"Centered ALE 기준 가장 우호적인 중심값은 약 <b>{bin_centers[max_idx]:.3f}</b>이며 "
            f"ALE={ale_vals[max_idx]:.4f}입니다. "
            f"가장 불리한 중심값은 약 <b>{bin_centers[min_idx]:.3f}</b>이며 "
            f"ALE={ale_vals[min_idx]:.4f}입니다. "
            f"양의 ALE 구간은 {pos_text}, 음의 ALE 구간은 {neg_text}입니다. "
            "이는 선택 Feature의 임계구간 또는 관리 우선구간을 판단하는 데 활용할 수 있습니다."
        )




    return "<br><br>".join(lines)



def select_optional_column(
    label,
    columns,
    key,
    preferred_names=(),
    help_text=None,
):
    """
    CSV 컬럼을 선택합니다.

    - 모든 선택 목록의 첫 항목에 None을 제공합니다.
    - preferred_names와 일치하는 컬럼이 있으면 자동 선택합니다.
    - None을 선택한 변수는 이후 매핑·그래프·모델 Feature에서 제외됩니다.
    """
    column_list = list(columns)
    options = [None] + column_list

    preferred_lookup = {
        str(name).strip().lower()
        for name in preferred_names
        if name is not None
    }

    default_index = 0
    for idx, column_name in enumerate(column_list, start=1):
        if str(column_name).strip().lower() in preferred_lookup:
            default_index = idx
            break

    return st.selectbox(
        label,
        options,
        index=default_index,
        key=key,
        format_func=lambda value: "None" if value is None else str(value),
        help=help_text,
    )



# =============================================================
# GEI 기반 환경구간 누적시간 · 생육/수확 증감 · Centered ALE 모듈
# =============================================================
GEI_STAGE_TABLE = pd.DataFrame(
    [
        [0, 20, "정상", "#dcfce7"],
        [20, 40, "관심", "#ecfccb"],
        [40, 60, "경계", "#fef3c7"],
        [60, 80, "주의", "#fed7aa"],
        [80, 100.0001, "위험", "#fecaca"],
    ],
    columns=["하한", "상한", "위험단계", "색상"],
)


def get_default_gei_zone_config():
    """완숙토마토 교육·연구용 환경구간 기본값."""
    return {
        "온도": {
            #"labels": ["T1 한계저온도(10도미만)", "T2 저온(10~13도)", "T3 중저온(13~20도)", "T4 최적온(20~25도)", "T5 적온(25~28도)", "T6 중고온(28~30도)", "T7 고온(30~33도)","T8 한계고온(33도이상)"],
            "labels": ["T1 한계저온도(10도미만)", "T2 저온(10~15도)", "T3 중저온(15~20도)", "T4 최적온(20~25도)", "T5 적온(25~30도)", "T6 고온(30도이상)"],
            "edges": [-np.inf, 10, 15, 20, 25, 30, np.inf],
            "weights": [4.0, 3.0, 2.0, 0.0, 1.0, 4.0],
            "unit": "℃",
        },
        "습도": {
            "labels": ["H1 최저습(40%미만)", "H2 저습(40~60%)", "H3 최적습(60~80%)", "H4 중습(80~90%)", "H5 고습(90~95%)", "H6 최악습(95%이상)"],
            "edges": [-np.inf, 40, 60, 80, 90, 95, np.inf],
            "weights": [4.0, 3.0, 0.0, 1.0, 2.0, 4.0],
            "unit": "%",
        },
        "CO₂": {
            "labels": ["C1 최저CO₂(350미만)", "C2 저CO₂(350-450)", "C3 중CO₂(450-550)", "C4 최적CO₂(550-650)", "C5 고CO₂(650-750)", "C6 최고CO₂(750이상)"],
            "edges": [-np.inf, 350, 450, 550, 650, 750, np.inf],
            "weights": [3.0, 2.5, 2.0, 1.5, 1.0, 0.0],
            "unit": "ppm",
        },
        "일사량": {
            "labels": ["L0 일사없음(0)", "L1 최저일사(1-60미만)", "L2 저일사(60-150)", "L3 적정일사(150-280)", "L4 중일사(280-500)", "L5 고일사(500-700)", "L6 최고일사(700이상)"],
            "edges": [-np.inf, 1, 60, 150, 280, 500, 700, np.inf],
            "weights": [0.0, 4.0, 2.5, 1.0, 0.0, 1.5, 3.0],
            "unit": "W/m²",
            # L0(0~1 W/m² 미만)는 야간/무일사 시간으로 별도 누적하되
            # 일사량 GEI의 분자·분모에서는 제외합니다.
            # 따라서 1~7주 누적기간이 바뀔 때마다 실제 광이 존재한
            # L1~L6 누적시간 합계가 자동으로 일사량 GEI 분모시간이 됩니다.
            "exclude_from_gei_denominator": ["L0 일사없음(0)"],
            "denominator_description": "유효광시간(L1~L6 누적시간 합계)",
        },
    }


def _finite_zone_text(left, right, unit):
    if np.isneginf(left):
        return f"< {right:g}{unit}"
    if np.isposinf(right):
        return f"≥ {left:g}{unit}"
    return f"{left:g} ~ {right:g}{unit}"


def gei_stage(score):
    if pd.isna(score):
        return "계산불가", "#e5e7eb"
    score = float(np.clip(score, 0, 100))
    row = GEI_STAGE_TABLE[
        (GEI_STAGE_TABLE["하한"] <= score)
        & (score < GEI_STAGE_TABLE["상한"])
    ]
    if row.empty:
        return "위험", "#fecaca"
    return row.iloc[0]["위험단계"], row.iloc[0]["색상"]


def infer_interval_hours(sensor_dates):
    """센서 간격의 중앙값으로 1개 관측치가 대표하는 시간을 계산합니다."""
    dt = pd.Series(pd.to_datetime(sensor_dates, errors="coerce")).dropna().sort_values()
    diffs = dt.diff().dt.total_seconds().div(3600).dropna()
    diffs = diffs[(diffs > 0) & (diffs <= 6)]
    if diffs.empty:
        return 1.0 / 6.0
    return float(np.clip(diffs.median(), 1 / 60, 1.0))


def classify_environment_series(values, config):
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=config["edges"],
        labels=config["labels"],
        right=False,
        include_lowest=True,
    )


def compute_one_environment_gei(subset, value_col, config, interval_hours):
    """
    환경구간 누적시간과 GEI를 계산합니다.

    반환값
    -------
    hours_dict : dict
        모든 환경구간의 누적시간(h). 일사량 L0도 표시·저장을 위해 유지합니다.
    score : float
        0~100 GEI 위험지수.
    coverage_hours : float
        센서값이 정상적으로 존재한 전체 유효시간(h). 데이터충족률 계산용입니다.
    gei_denominator_hours : float
        실제 GEI 점수의 분모로 사용한 시간(h).

    일사량 처리
    -----------
    config["exclude_from_gei_denominator"]에 지정된 구간은 누적시간에는 남지만
    GEI 분자와 분모에서는 제외합니다. 현재 일사량의 L0(0~1 W/m² 미만)가 이에
    해당하므로, 일사량 GEI는 실제 광이 존재한 L1~L6 시간만으로 계산됩니다.

        일사량 GEI =
            Σ(L1~L6 누적시간 × 위험가중치)
            -------------------------------- × 100
            (L1~L6 유효광시간 합계 × 최대위험가중치)

    1~7주 선택 시 subset 자체가 7~49일로 바뀌므로 유효광시간 분모도
    각 누적기간에서 자동으로 다시 계산됩니다.
    """
    labels = list(config["labels"])
    if value_col is None or value_col not in subset.columns or subset.empty:
        return {label: 0.0 for label in labels}, np.nan, 0.0, 0.0

    classified = classify_environment_series(subset[value_col], config)
    valid = classified.notna()

    # 센서값 자체가 존재한 전체 시간: 데이터 누락 여부를 평가할 때 사용
    coverage_hours = float(valid.sum() * interval_hours)

    hours = (
        classified[valid]
        .value_counts(sort=False)
        .reindex(labels, fill_value=0)
        .astype(float)
        * interval_hours
    )

    weights = pd.Series(config["weights"], index=labels, dtype=float)

    # GEI 분모에서 제외할 정상 비활성 구간(예: 일사량 야간 L0)
    excluded_labels = set(config.get("exclude_from_gei_denominator", []))
    gei_labels = [label for label in labels if label not in excluded_labels]

    # 제외 구간이 잘못 설정되어 전부 빠지는 상황을 방지
    if not gei_labels:
        gei_labels = labels

    gei_hours = hours.reindex(gei_labels, fill_value=0.0)
    gei_weights = weights.reindex(gei_labels).fillna(0.0)

    # 일사량의 경우 L1~L6 합계, 나머지 환경은 전체 유효시간과 동일
    gei_denominator_hours = float(gei_hours.sum())

    max_weight = float(gei_weights.max()) if len(gei_weights) and float(gei_weights.max()) > 0 else 1.0
    denominator = gei_denominator_hours * max_weight

    numerator = float((gei_hours * gei_weights).sum())
    score = (
        numerator / denominator * 100.0
        if denominator > 0
        else np.nan
    )

    return (
        hours.to_dict(),
        float(np.clip(score, 0, 100)) if np.isfinite(score) else np.nan,
        coverage_hours,
        gei_denominator_hours,
    )


def build_weekly_gei_dataset(
    sensor_df,
    yield_df,
    date_col_sensor,
    date_col_yield,
    env_column_map,
    growth_column_map,
    zone_config,
    window_days=7,
    start_date=None,
    end_date=None,
):
    """각 조사일 직전 N일 환경구간 누적시간과 GEI를 생육/수확값에 결합합니다."""
    sensor = sensor_df.copy()
    survey = yield_df.copy()
    sensor[date_col_sensor] = pd.to_datetime(sensor[date_col_sensor], errors="coerce")
    survey[date_col_yield] = pd.to_datetime(survey[date_col_yield], errors="coerce")
    sensor = sensor.dropna(subset=[date_col_sensor]).sort_values(date_col_sensor)
    survey = survey.dropna(subset=[date_col_yield]).sort_values(date_col_yield)

    if start_date is not None:
        survey = survey[survey[date_col_yield] >= pd.Timestamp(start_date)]
    if end_date is not None:
        survey = survey[survey[date_col_yield] <= pd.Timestamp(end_date)]

    interval_hours = infer_interval_hours(sensor[date_col_sensor])
    expected_hours = float(window_days * 24)
    rows = []

    for _, survey_row in survey.iterrows():
        survey_date = pd.Timestamp(survey_row[date_col_yield])
        window_start = survey_date - pd.Timedelta(days=int(window_days))
        # 조사일 00:00까지의 직전 기간을 사용하여 미래정보 포함을 방지합니다.
        mask = (
            (sensor[date_col_sensor] >= window_start)
            & (sensor[date_col_sensor] < survey_date)
        )
        subset = sensor.loc[mask]
        result = {
            "조사일자": survey_date,
            "환경기간시작": window_start,
            "환경기간종료": survey_date,
            "분석기간일수": int(window_days),
            "기대누적시간(h)": expected_hours,
        }

        # 작기/생육단계/품종/작물 메타데이터가 원본 조사파일에 있으면
        # 가중치의 작기 독립 검증 및 생육단계별 프로파일 분석에 활용할 수 있도록 보존합니다.
        metadata_aliases = {
            "작기": ["작기", "작기번호", "crop_cycle", "CropCycle", "cycle", "Cycle"],
            "생육단계": ["생육단계", "growth_stage", "GrowthStage", "stage", "Stage"],
            "품종": ["품종", "cultivar", "Cultivar", "variety", "Variety"],
            "작물": ["작물", "crop", "Crop"],
        }
        for canonical_name, candidates in metadata_aliases.items():
            for candidate in candidates:
                if candidate in survey_row.index and pd.notna(survey_row.get(candidate, np.nan)):
                    result[canonical_name] = survey_row.get(candidate)
                    break

        gei_values = []
        coverage_values = []

        for env_name, value_col in env_column_map.items():
            if env_name not in zone_config or value_col is None:
                continue
            hours, score, coverage, gei_denominator_hours = compute_one_environment_gei(
                subset, value_col, zone_config[env_name], interval_hours
            )
            for label, hour_value in hours.items():
                result[f"{label} 누적시간(h)"] = float(hour_value)

            result[f"{env_name} GEI"] = score

            # 센서 전체 유효시간: 데이터충족률 계산용
            result[f"{env_name} 유효시간(h)"] = coverage
            result[f"{env_name} 데이터충족률(%)"] = (
                min(100.0, coverage / expected_hours * 100)
                if expected_hours
                else np.nan
            )

            # 실제 GEI 분모로 사용된 시간.
            # 온도·습도·CO₂는 전체 유효시간이며,
            # 일사량은 L0(야간/무일사)를 제외한 L1~L6 유효광시간입니다.
            result[f"{env_name} GEI분모시간(h)"] = gei_denominator_hours

            if env_name == "일사량":
                dark_hours = max(0.0, float(coverage - gei_denominator_hours))
                result["일사량 유효광시간(h)"] = float(gei_denominator_hours)
                result["일사량 야간·무일사 제외시간(h)"] = dark_hours
                result["일사량 유효광시간비율(%)"] = (
                    gei_denominator_hours / coverage * 100.0
                    if coverage > 0
                    else np.nan
                )

            if np.isfinite(score):
                gei_values.append(score)
            if coverage > 0:
                coverage_values.append(min(100.0, coverage / expected_hours * 100))

        result["통합 GEI"] = float(np.mean(gei_values)) if gei_values else np.nan
        result["통합 데이터충족률(%)"] = float(np.mean(coverage_values)) if coverage_values else np.nan
        stage, _ = gei_stage(result["통합 GEI"])
        result["GEI 위험단계"] = stage

        for output_name, source_col in growth_column_map.items():
            result[output_name] = (
                pd.to_numeric(pd.Series([survey_row.get(source_col, np.nan)]), errors="coerce").iloc[0]
                if source_col is not None
                else np.nan
            )
        rows.append(result)

    gei_df = pd.DataFrame(rows).sort_values("조사일자").reset_index(drop=True)
    growth_targets = [c for c in growth_column_map if c in gei_df.columns]
    gei_df["GEI 변화량"] = pd.to_numeric(gei_df.get("통합 GEI"), errors="coerce").diff()
    for target in growth_targets:
        numeric = pd.to_numeric(gei_df[target], errors="coerce")
        gei_df[target] = numeric
        gei_df[f"{target} 변화량"] = numeric.diff()
        tolerance = max(float(numeric.std(skipna=True) or 0) * 0.02, 1e-9)
        gei_df[f"{target} 증감"] = np.select(
            [gei_df[f"{target} 변화량"] > tolerance, gei_df[f"{target} 변화량"] < -tolerance],
            ["↑ 증가", "↓ 감소"],
            default="→ 유지",
        )
        gei_df[f"GEI상승시 {target} 반응"] = np.select(
            [
                (gei_df["GEI 변화량"] > 0) & (gei_df[f"{target} 변화량"] > tolerance),
                (gei_df["GEI 변화량"] > 0) & (gei_df[f"{target} 변화량"] < -tolerance),
                gei_df["GEI 변화량"] > 0,
            ],
            ["GEI↑ · 생육↑", "GEI↑ · 생육↓", "GEI↑ · 유지"],
            default="GEI 비상승",
        )
    return gei_df


# =============================================================
# 데이터 기반 통합 GEI 가중치 최적화 모듈 (v28.1)
# =============================================================
GEI_ENV_ORDER = ["온도", "습도", "CO₂", "일사량"]
GEI_WEIGHT_METHOD_LABELS = {
    "equal": "동일가중(기준선)",
    "shap": "SHAP 중요도",
    "regression": "표준화 회귀",
    "correlation": "상관계수",
    "optimization": "최적화",
    "consensus": "합의가중(권장)",
    "manual": "수동가중",
    "historical": "누적작기 학습가중",
}


def _normalize_nonnegative_weights(values, feature_names):
    """음수가중치를 허용하지 않고 합이 1이 되도록 안전 정규화합니다."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) != len(feature_names):
        arr = np.ones(len(feature_names), dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        arr = np.ones(len(feature_names), dtype=float)
        total = float(arr.sum())
    arr = arr / total
    return {name: float(value) for name, value in zip(feature_names, arr)}


def _weighted_gei_series(df, weights):
    """
    개별 GEI를 행별 유효값에 대해 재정규화하여 통합 GEI를 계산합니다.
    일부 센서 GEI가 결측이어도 존재하는 환경의 가중치 합으로 다시 나눕니다.
    """
    if df is None or df.empty or not weights:
        return pd.Series(np.nan, index=getattr(df, "index", None), dtype=float)

    numerator = pd.Series(0.0, index=df.index, dtype=float)
    denominator = pd.Series(0.0, index=df.index, dtype=float)
    for feature, weight in weights.items():
        if feature not in df.columns or not np.isfinite(weight) or weight <= 0:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        numerator.loc[valid] = numerator.loc[valid] + values.loc[valid] * float(weight)
        denominator.loc[valid] = denominator.loc[valid] + float(weight)

    out = numerator / denominator.replace(0, np.nan)
    return out.clip(0, 100)


def _prepare_gei_weight_target(gei_df, target_col, response_basis, gei_features):
    """가중치 추정용 반응변수와 개별 GEI를 조사일 기준으로 정렬합니다."""
    if gei_df is None or gei_df.empty or target_col not in gei_df.columns or not gei_features:
        return pd.DataFrame(), "사용 불가"

    base_cols = ["조사일자"] + [c for c in gei_features if c in gei_df.columns] + [target_col]
    for extra in ["작기", "생육단계", "품종", "작물"]:
        if extra in gei_df.columns and extra not in base_cols:
            base_cols.append(extra)

    work = gei_df[base_cols].copy()
    work["조사일자"] = pd.to_datetime(work["조사일자"], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.sort_values("조사일자").reset_index(drop=True)

    if response_basis == "원자료":
        work["가중치학습반응"] = work[target_col]
        label = f"{target_col} 원자료"
    elif response_basis == "조사간 변화량":
        work["가중치학습반응"] = work[target_col].diff()
        label = f"{target_col} 조사간 변화량"
    else:
        baseline_mode = (
            "월평균 변화량 대비(신규 권장)"
            if response_basis == "월평균 NGR"
            else "생육단계 기대 변화량 대비(최종 권장)"
        )
        response_df, _meta = build_gei_growth_response_curve(
            gei_df=gei_df,
            gei_feature=gei_features[0],
            target_col=target_col,
            baseline_mode=baseline_mode,
            stable_band_pct=2.0,
            danger_pct=-10.0,
        )
        if response_df.empty:
            work["가중치학습반응"] = work[target_col].diff()
            label = f"{target_col} 조사간 변화량(NGR 계산 불가 대체)"
        else:
            response_map = response_df[["조사일자", "반응률(%)"]].copy()
            response_map["조사일자"] = pd.to_datetime(response_map["조사일자"], errors="coerce")
            response_map = response_map.drop_duplicates("조사일자", keep="last")
            work = work.merge(response_map, on="조사일자", how="left")
            work["가중치학습반응"] = pd.to_numeric(work["반응률(%)"], errors="coerce")
            label = (
                f"{target_col} 월평균 변화량 대비 NGR(%)"
                if response_basis == "월평균 NGR"
                else f"{target_col} 생육단계 기대 변화량 대비 NGR(%)"
            )

    numeric_cols = gei_features + ["가중치학습반응"]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols).reset_index(drop=True)
    return work, label


def _weighted_gei_cv_metrics(calibration_df, weights, gei_features):
    """
    통합 GEI 하나로 반응변수를 설명하는 투명한 선형 CV 성능을 계산합니다.
    작기 컬럼이 3개 이상이면 Leave-One-Crop-Cycle-Out, 아니면 TimeSeriesSplit을 사용합니다.
    """
    if calibration_df is None or calibration_df.empty or len(calibration_df) < 6:
        return {"CV_R2": np.nan, "CV_RMSE": np.nan, "Pearson": np.nan, "Spearman": np.nan, "검증": "표본부족"}

    temp = calibration_df.copy().sort_values("조사일자").reset_index(drop=True)
    temp["_weighted_gei"] = _weighted_gei_series(temp, weights)
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna(subset=["_weighted_gei", "가중치학습반응"]).reset_index(drop=True)
    if len(temp) < 6:
        return {"CV_R2": np.nan, "CV_RMSE": np.nan, "Pearson": np.nan, "Spearman": np.nan, "검증": "표본부족"}

    X = temp[["_weighted_gei"]].to_numpy(dtype=float)
    y = temp["가중치학습반응"].to_numpy(dtype=float)
    pred = np.full(len(temp), np.nan, dtype=float)
    validation_name = "TimeSeriesSplit"

    if "작기" in temp.columns and temp["작기"].notna().sum() == len(temp) and temp["작기"].nunique() >= 3:
        groups = temp["작기"].astype(str).to_numpy()
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(X, y, groups=groups))
        validation_name = "Leave-One-Crop-Cycle-Out"
    else:
        n_splits = min(5, max(2, len(temp) // 6))
        n_splits = min(n_splits, len(temp) - 1)
        if n_splits < 2:
            return {"CV_R2": np.nan, "CV_RMSE": np.nan, "Pearson": np.nan, "Spearman": np.nan, "검증": "표본부족"}
        splitter = TimeSeriesSplit(n_splits=n_splits)
        splits = list(splitter.split(X))

    for train_idx, test_idx in splits:
        if len(train_idx) < 2 or len(test_idx) < 1:
            continue
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])

    valid = np.isfinite(pred) & np.isfinite(y)
    cv_r2 = r2_score(y[valid], pred[valid]) if valid.sum() >= 2 else np.nan
    cv_rmse = float(np.sqrt(mean_squared_error(y[valid], pred[valid]))) if valid.sum() >= 1 else np.nan
    pearson = float(pd.Series(X[:, 0]).corr(pd.Series(y), method="pearson"))
    spearman = float(pd.Series(X[:, 0]).corr(pd.Series(y), method="spearman"))
    return {
        "CV_R2": float(cv_r2) if np.isfinite(cv_r2) else np.nan,
        "CV_RMSE": cv_rmse,
        "Pearson": pearson if np.isfinite(pearson) else np.nan,
        "Spearman": spearman if np.isfinite(spearman) else np.nan,
        "검증": validation_name,
    }


def estimate_data_driven_gei_weights(
    gei_df,
    target_col,
    response_basis="생육단계 NGR",
    shap_model_name="RandomForest",
    optimization_trials=1200,
):
    """동일/SHAP/표준화회귀/상관/최적화/합의 가중치를 일괄 산출합니다."""
    gei_features = [f"{env} GEI" for env in GEI_ENV_ORDER if f"{env} GEI" in gei_df.columns]
    if not gei_features:
        return {}, pd.DataFrame(), pd.DataFrame(), "개별 GEI 없음"

    calibration_df, response_label = _prepare_gei_weight_target(
        gei_df, target_col, response_basis, gei_features
    )
    equal_weights = _normalize_nonnegative_weights(np.ones(len(gei_features)), gei_features)
    methods = {"equal": equal_weights}

    if len(calibration_df) < max(8, len(gei_features) + 3):
        methods.update({
            "shap": equal_weights.copy(),
            "regression": equal_weights.copy(),
            "correlation": equal_weights.copy(),
            "optimization": equal_weights.copy(),
            "consensus": equal_weights.copy(),
        })
    else:
        X = calibration_df[gei_features].astype(float)
        y = calibration_df["가중치학습반응"].astype(float)

        # A. SHAP mean(|SHAP|) 기반
        shap_values_abs = None
        try:
            model = make_model(shap_model_name)
            model.fit(X, y)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            except Exception:
                explainer = shap.Explainer(model, X)
                explained = explainer(X)
                shap_values = explained.values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_arr = np.asarray(shap_values, dtype=float)
            if shap_arr.ndim == 3:
                shap_arr = shap_arr[..., 0]
            if shap_arr.ndim == 2 and shap_arr.shape[1] == len(gei_features):
                shap_values_abs = np.nanmean(np.abs(shap_arr), axis=0)
        except Exception:
            shap_values_abs = None

        if shap_values_abs is None or not np.isfinite(shap_values_abs).any() or float(np.nansum(shap_values_abs)) <= 0:
            try:
                fallback_model = RandomForestRegressor(n_estimators=300, random_state=42)
                fallback_model.fit(X, y)
                perm = permutation_importance(fallback_model, X, y, n_repeats=15, random_state=42, scoring="r2")
                shap_values_abs = np.clip(np.asarray(perm.importances_mean, dtype=float), 0.0, None)
            except Exception:
                shap_values_abs = np.ones(len(gei_features), dtype=float)
        methods["shap"] = _normalize_nonnegative_weights(shap_values_abs, gei_features)

        # B. 표준화 회귀계수 절대값 기반
        try:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            Xz = x_scaler.fit_transform(X)
            yz = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()
            reg = LinearRegression().fit(Xz, yz)
            beta_abs = np.abs(np.asarray(reg.coef_, dtype=float))
        except Exception:
            beta_abs = np.ones(len(gei_features), dtype=float)
        methods["regression"] = _normalize_nonnegative_weights(beta_abs, gei_features)

        # C. 절대 상관계수 기반 (보조지표)
        corr_abs = []
        for feature in gei_features:
            corr = pd.Series(X[feature]).corr(pd.Series(y), method="pearson")
            corr_abs.append(abs(float(corr)) if pd.notna(corr) and np.isfinite(corr) else 0.0)
        methods["correlation"] = _normalize_nonnegative_weights(corr_abs, gei_features)

        # D. 제약 최적화: Dirichlet 후보 탐색 + 투명한 CV R²/방향성 결합
        rng = np.random.default_rng(42)
        candidate_arrays = [
            np.asarray(list(equal_weights.values()), dtype=float),
            np.asarray(list(methods["shap"].values()), dtype=float),
            np.asarray(list(methods["regression"].values()), dtype=float),
            np.asarray(list(methods["correlation"].values()), dtype=float),
        ]
        trials = int(np.clip(optimization_trials, 200, 5000))
        candidate_arrays.extend(rng.dirichlet(np.ones(len(gei_features)), size=trials))

        best_score = -np.inf
        best_weights = equal_weights.copy()
        for arr in candidate_arrays:
            candidate = _normalize_nonnegative_weights(arr, gei_features)
            metric = _weighted_gei_cv_metrics(calibration_df, candidate, gei_features)
            cv_r2 = metric.get("CV_R2", np.nan)
            pearson = metric.get("Pearson", np.nan)
            base_score = float(cv_r2) if np.isfinite(cv_r2) else -10.0
            # GEI는 위험지수이므로 NGR/증가량과 음의 방향일수록 해석 일관성이 높습니다.
            direction_bonus = (-float(pearson)) * 0.15 if np.isfinite(pearson) else 0.0
            score = base_score + direction_bonus
            if score > best_score:
                best_score = score
                best_weights = candidate
        methods["optimization"] = best_weights

        # E. 합의가중: SHAP + 회귀 + 상관 + 최적화 평균
        consensus_array = np.mean(
            [np.asarray(list(methods[m].values()), dtype=float) for m in ["shap", "regression", "correlation", "optimization"]],
            axis=0,
        )
        methods["consensus"] = _normalize_nonnegative_weights(consensus_array, gei_features)

    # 방법별 비교표
    rows = []
    for method_key in ["equal", "shap", "regression", "correlation", "optimization", "consensus"]:
        weights = methods.get(method_key, equal_weights)
        metric = _weighted_gei_cv_metrics(calibration_df, weights, gei_features)
        row = {
            "방법": GEI_WEIGHT_METHOD_LABELS.get(method_key, method_key),
            "method_key": method_key,
            "CV R²": metric.get("CV_R2", np.nan),
            "CV RMSE": metric.get("CV_RMSE", np.nan),
            "Pearson r": metric.get("Pearson", np.nan),
            "Spearman ρ": metric.get("Spearman", np.nan),
            "검증방식": metric.get("검증", ""),
        }
        for feature in gei_features:
            row[feature.replace(" GEI", " 가중치")] = weights.get(feature, 0.0)
        rows.append(row)
    comparison_df = pd.DataFrame(rows)

    # 개별 환경의 방향성 진단표
    direction_rows = []
    if not calibration_df.empty:
        for feature in gei_features:
            p = calibration_df[feature].corr(calibration_df["가중치학습반응"], method="pearson")
            s = calibration_df[feature].corr(calibration_df["가중치학습반응"], method="spearman")
            direction_rows.append({
                "환경 GEI": feature,
                "Pearson r": p,
                "Spearman ρ": s,
                "위험지수 방향일치": "✅" if pd.notna(p) and p < 0 else ("⚠️" if pd.notna(p) else "N/A"),
            })
    direction_df = pd.DataFrame(direction_rows)
    return methods, comparison_df, direction_df, response_label


def apply_gei_weight_methods(gei_df, methods, selected_method_key, manual_weights=None):
    """모든 후보 통합 GEI 열을 만들고 선택한 가중방식을 표준 '통합 GEI'에 적용합니다."""
    if gei_df is None or gei_df.empty:
        return gei_df
    out = gei_df.copy()
    gei_features = [f"{env} GEI" for env in GEI_ENV_ORDER if f"{env} GEI" in out.columns]
    if not gei_features:
        return out

    if "통합 GEI" in out.columns:
        out["통합 GEI(기존)"] = pd.to_numeric(out["통합 GEI"], errors="coerce")

    for method_key, weights in methods.items():
        label = GEI_WEIGHT_METHOD_LABELS.get(method_key, method_key)
        out[f"통합 GEI[{label}]"] = _weighted_gei_series(out, weights)

    if manual_weights is not None:
        normalized_manual = _normalize_nonnegative_weights(
            [manual_weights.get(f, 0.0) for f in gei_features], gei_features
        )
        out["통합 GEI[수동가중]"] = _weighted_gei_series(out, normalized_manual)
        methods = dict(methods)
        methods["manual"] = normalized_manual

    selected_weights = methods.get(selected_method_key, methods.get("equal", {}))
    out["통합 GEI"] = _weighted_gei_series(out, selected_weights)
    out["통합 GEI 가중방식"] = GEI_WEIGHT_METHOD_LABELS.get(selected_method_key, selected_method_key)
    out["GEI 위험단계"] = out["통합 GEI"].apply(lambda x: gei_stage(x)[0])
    return out


def render_gei_weight_optimizer(gei_df, growth_targets):
    """Streamlit UI: 데이터 기반 GEI 가중치 비교·선택·적용."""
    render_stylish_section(
        "⚖️ 데이터 기반 통합 GEI 가중치 최적화",
        "온도·습도·CO₂·일사량을 고정 25%로 합산하는 기준선과 SHAP·표준화 회귀·상관·제약 최적화 가중치를 비교합니다. 가중치는 영향의 크기만 반영하며, SHAP 부호 자체를 가중치로 사용하지 않습니다.",
        kicker="DATA-DRIVEN GEI WEIGHTS",
    )

    gei_features = [f"{env} GEI" for env in GEI_ENV_ORDER if f"{env} GEI" in gei_df.columns]
    if len(gei_features) < 2:
        st.info("가중치 비교에는 최소 2개 이상의 개별 환경 GEI가 필요합니다.")
        equal = _normalize_nonnegative_weights(np.ones(len(gei_features)), gei_features)
        return apply_gei_weight_methods(gei_df, {"equal": equal}, "equal"), equal, "equal", pd.DataFrame()

    c1, c2, c3, c4 = st.columns([1.2, 1.35, 1.1, 0.9])
    with c1:
        weight_target = st.selectbox(
            "가중치 학습 생육·수확 Target",
            options=list(growth_targets),
            key="gei_weight_target_v281",
        )
    with c2:
        response_basis = st.selectbox(
            "가중치 학습 반응 기준",
            ["생육단계 NGR", "월평균 NGR", "조사간 변화량", "원자료"],
            index=0,
            key="gei_weight_response_basis_v281",
            help="누적형 생육지표는 생육단계 NGR 사용을 권장합니다. 0%가 기대 성장속도이고 음수는 기대보다 느린 성장입니다.",
        )
    with c3:
        shap_model_name = st.selectbox(
            "SHAP 가중치 모델",
            ["RandomForest", "XGBoost", "LGBM", "GradientBoosting"],
            index=0,
            key="gei_weight_shap_model_v281",
        )
    with c4:
        optimization_trials = st.selectbox(
            "최적화 탐색수",
            [500, 1200, 2500],
            index=1,
            key="gei_weight_trials_v281",
        )

    # 생육단계 컬럼이 있으면 해당 단계만으로 가중치를 별도 학습할 수 있습니다.
    calibration_source = gei_df.copy()
    if "생육단계" in gei_df.columns and gei_df["생육단계"].dropna().nunique() >= 2:
        stages = ["전체"] + sorted(gei_df["생육단계"].dropna().astype(str).unique().tolist())
        selected_stage = st.selectbox("생육단계별 가중치 프로파일", stages, key="gei_weight_stage_v281")
        if selected_stage != "전체":
            calibration_source = gei_df[gei_df["생육단계"].astype(str) == selected_stage].copy()
            st.caption(f"현재 가중치는 생육단계 '{selected_stage}' 자료로 학습합니다. 선택 가중치는 현재 화면의 GEI 계산에 적용됩니다.")

    methods, comparison_df, direction_df, response_label = estimate_data_driven_gei_weights(
        calibration_source,
        target_col=weight_target,
        response_basis=response_basis,
        shap_model_name=shap_model_name,
        optimization_trials=optimization_trials,
    )

    # v29.0: 과거 저장 작기에서 누적학습된 환경간 GEI 가중치를 다음 작기의 선택지로 제공합니다.
    historical_weights = st.session_state.get("historical_gei_weights", {})
    if isinstance(historical_weights, dict) and historical_weights:
        historical_vector = [float(historical_weights.get(feature, 0.0) or 0.0) for feature in gei_features]
        if np.isfinite(np.asarray(historical_vector, dtype=float)).any() and float(np.nansum(historical_vector)) > 0:
            methods["historical"] = _normalize_nonnegative_weights(historical_vector, gei_features)

    method_options = ["equal", "shap", "regression", "correlation", "optimization", "consensus"]
    if "historical" in methods:
        method_options.append("historical")
    method_options.append("manual")
    selected_method_key = st.selectbox(
        "최종 통합 GEI에 적용할 가중방식",
        method_options,
        index=5,
        format_func=lambda key: GEI_WEIGHT_METHOD_LABELS.get(key, key),
        key="gei_weight_method_v281",
    )

    manual_weights = None
    if selected_method_key == "manual":
        manual_weights = {}
        manual_cols = st.columns(len(gei_features))
        for idx, feature in enumerate(gei_features):
            with manual_cols[idx]:
                manual_weights[feature] = st.number_input(
                    feature.replace(" GEI", " 가중치"),
                    min_value=0.0,
                    max_value=1.0,
                    value=float(1.0 / len(gei_features)),
                    step=0.01,
                    key=f"manual_gei_weight_{idx}_v281",
                )
        manual_weights = _normalize_nonnegative_weights(
            [manual_weights[f] for f in gei_features], gei_features
        )

    updated = apply_gei_weight_methods(
        gei_df,
        methods=methods,
        selected_method_key=selected_method_key,
        manual_weights=manual_weights,
    )

    if selected_method_key == "manual" and manual_weights is not None:
        selected_weights = manual_weights
    else:
        selected_weights = methods.get(selected_method_key, methods.get("equal", {}))

    st.markdown(
        f"""
        <div class="xai-insight-card">
            <b>가중치 학습 반응:</b> {response_label}<br>
            <b>통합 GEI:</b> Σ(w<sub>j</sub> × GEI<sub>j</sub>), &nbsp; w<sub>j</sub> ≥ 0, &nbsp; Σw<sub>j</sub> = 1<br>
            <b>현재 적용:</b> {GEI_WEIGHT_METHOD_LABELS.get(selected_method_key, selected_method_key)}<br>
            <b>중요:</b> SHAP 가중치는 mean(|SHAP|)로 계산하여 영향 <u>크기</u>를 사용합니다. 양/음 방향은 각 환경 GEI의 위험구간 정의 및 방향성 진단에서 별도로 확인합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 선택 가중치 카드
    weight_cols = st.columns(len(gei_features))
    for idx, feature in enumerate(gei_features):
        with weight_cols[idx]:
            value = float(selected_weights.get(feature, 0.0))
            st.metric(feature.replace(" GEI", ""), f"{value * 100:.1f}%")

    left, right = st.columns([1.45, 0.85], gap="large")
    with left:
        render_panel_label("가중치 도출 방법별 비교")
        if not comparison_df.empty:
            show = comparison_df.drop(columns=["method_key"], errors="ignore").copy()
            weight_cols_names = [c for c in show.columns if c.endswith("가중치")]
            for c in weight_cols_names:
                show[c] = show[c] * 100.0
            rename_map = {c: c.replace("가중치", "가중치(%)") for c in weight_cols_names}
            show = show.rename(columns=rename_map)
            st.dataframe(show.round(4), use_container_width=True, hide_index=True, height=320)
    with right:
        render_panel_label("개별 GEI 방향성 진단")
        if not direction_df.empty:
            st.dataframe(direction_df.round(4), use_container_width=True, hide_index=True, height=220)
        st.caption("위험 GEI는 높을수록 불리하도록 설계되므로 NGR/증가량과 음의 상관이 해석상 자연스럽습니다. 양의 상관이 반복되면 해당 환경구간 위험가중치 자체를 재검토하세요.")

    # 가중치 bar chart
    chart_df = pd.DataFrame({
        "환경": [f.replace(" GEI", "") for f in gei_features],
        "가중치": [float(selected_weights.get(f, 0.0)) * 100.0 for f in gei_features],
    })
    fig = go.Figure(go.Bar(
        x=chart_df["환경"],
        y=chart_df["가중치"],
        text=[f"{v:.1f}%" for v in chart_df["가중치"]],
        textposition="outside",
        marker=dict(color=["#ef4444", "#3b82f6", "#22c55e", "#f59e0b"][:len(chart_df)]),
    ))
    fig.update_layout(
        height=330,
        title=f"현재 통합 GEI 환경가중치 · {GEI_WEIGHT_METHOD_LABELS.get(selected_method_key, selected_method_key)}",
        yaxis_title="가중치(%)",
        yaxis=dict(range=[0, max(40, float(chart_df["가중치"].max()) * 1.25)]),
        margin=dict(l=45, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
    )
    st.plotly_chart(fig, use_container_width=True, key="gei_weight_bar_v281")

    return updated, selected_weights, selected_method_key, comparison_df


def centered_ale_1d(model, X, feature, bins=10):
    """회귀모델에 대한 안정형 1차원 Centered ALE 계산.

    안정성 보완 사항
    - 존재하지 않는 ``np.digit`` 대신 ``np.digitize`` 사용
    - NaN/Inf 제거 및 유효 표본만 분석
    - 중복 분위수 경계 제거
    - bin index 범위 강제 제한
    - 빈 구간과 예측 결과의 비정상값 안전 처리
    """
    if not isinstance(X, pd.DataFrame) or X.empty or feature not in X.columns:
        return pd.DataFrame()

    try:
        requested_bins = max(2, int(bins))
    except (TypeError, ValueError):
        requested_bins = 10

    # 분석 Feature를 숫자로 변환하고 NaN/Inf 행은 제외합니다.
    x_numeric = pd.to_numeric(X[feature], errors="coerce")
    finite_mask = np.isfinite(x_numeric.to_numpy(dtype=float))
    if int(finite_mask.sum()) < 4:
        return pd.DataFrame()

    X_valid = X.loc[finite_mask].copy()
    x_valid = pd.to_numeric(X_valid[feature], errors="coerce").astype(float)
    unique_count = int(x_valid.nunique(dropna=True))
    if unique_count < 4:
        return pd.DataFrame()

    # 표본의 고유값 수보다 많은 구간을 만들지 않습니다.
    actual_bins = min(requested_bins, unique_count - 1)
    quantiles = np.linspace(0.0, 1.0, actual_bins + 1)
    edges = np.quantile(x_valid.to_numpy(dtype=float), quantiles)
    edges = np.unique(np.asarray(edges, dtype=float))
    edges = edges[np.isfinite(edges)]

    if len(edges) < 3:
        return pd.DataFrame()

    # 최소·최대값이 경계에서 빠지지 않도록 명시적으로 보정합니다.
    edges[0] = float(x_valid.min())
    edges[-1] = float(x_valid.max())

    # np.digitize는 0 ~ len(edges)-2 범위의 bin index를 반환하도록 구성합니다.
    bin_idx = np.digitize(
        x_valid.to_numpy(dtype=float),
        bins=edges[1:-1],
        right=False,
    )
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    local_effects = []
    counts = []
    centers = []

    for idx in range(len(edges) - 1):
        row_mask = bin_idx == idx
        count = int(np.count_nonzero(row_mask))
        counts.append(count)
        centers.append(float((edges[idx] + edges[idx + 1]) / 2.0))

        if count == 0:
            local_effects.append(0.0)
            continue

        low = X_valid.iloc[np.flatnonzero(row_mask)].copy()
        high = low.copy()
        low[feature] = float(edges[idx])
        high[feature] = float(edges[idx + 1])

        try:
            pred_low = np.asarray(model.predict(low), dtype=float).reshape(-1)
            pred_high = np.asarray(model.predict(high), dtype=float).reshape(-1)
            effect = pred_high - pred_low
            effect = effect[np.isfinite(effect)]
            local_effects.append(float(np.mean(effect)) if effect.size else 0.0)
        except Exception:
            # 특정 구간 예측 실패가 전체 대시보드를 중단시키지 않도록 처리합니다.
            local_effects.append(0.0)

    if not local_effects:
        return pd.DataFrame()

    accumulated = np.cumsum(np.asarray(local_effects, dtype=float))
    counts_arr = np.asarray(counts, dtype=float)
    total_count = float(counts_arr.sum())

    if total_count > 0:
        center_value = float(np.average(accumulated, weights=counts_arr))
    else:
        center_value = float(np.mean(accumulated))

    centered = accumulated - center_value

    result = pd.DataFrame(
        {
            "구간중심": np.asarray(centers, dtype=float),
            "Centered ALE": np.asarray(centered, dtype=float),
            "표본수": np.asarray(counts, dtype=int),
            "구간하한": np.asarray(edges[:-1], dtype=float),
            "구간상한": np.asarray(edges[1:], dtype=float),
        }
    )
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["구간중심", "Centered ALE", "구간하한", "구간상한"]
    ).reset_index(drop=True)


def detect_ale_threshold(ale_df):
    if ale_df.empty:
        return None
    x = ale_df["구간중심"].to_numpy(dtype=float)
    y = ale_df["Centered ALE"].to_numpy(dtype=float)
    if len(x) < 3:
        return None
    slopes = np.gradient(y, x)
    negative_candidates = np.where((y < 0) & (slopes < 0))[0]
    idx = int(negative_candidates[0]) if len(negative_candidates) else int(np.argmin(y))
    return {
        "threshold": float(x[idx]),
        "ale": float(y[idx]),
        "best": float(x[int(np.argmax(y))]),
        "worst": float(x[int(np.argmin(y))]),
    }



def build_gei_growth_response_curve(
    gei_df,
    gei_feature,
    target_col,
    baseline_mode="생육단계 기대 변화량 대비(최종 권장)",
    stable_band_pct=2.0,
    danger_pct=-10.0,
    manual_baseline=None,
):
    """
    조사일별 GEI와 생육·수확 반응률(%)을 연결한 기술통계용 데이터셋을 생성합니다.

    신규 권장 기준
    -------------
    1) 월평균 변화량 대비(신규 권장)
       - 조사 간격이 6~8일처럼 달라도 비교 가능하도록 실제 변화량을 7일 환산 변화량으로 표준화합니다.
       - 각 조사월의 7일 환산 변화량 평균을 기대 변화량으로 사용합니다.
       - 가능하면 현재 관측치를 제외한 leave-one-out 월평균을 사용하여 자기참조 편향을 줄입니다.
       Response(%) = (7일환산 변화량 - 월 기대 변화량) / |월 기대 변화량| × 100

    2) 생육단계 기대 변화량 대비(최종 권장)
       - 조사 시작일부터의 경과일을 생육단계의 연속형 대리변수로 사용합니다.
       - 7일 환산 변화량에 대해 1~2차 다항 추세를 적합해 기대 변화량을 계산합니다.
       - 누적형 생육지표(초장·생장길이 등)의 단순 시간 증가 효과를 줄이는 데 목적이 있습니다.
       Response(%) = (7일환산 변화량 - 단계별 기대 변화량) / |단계별 기대 변화량| × 100

    기존 기준(호환 유지)
    --------------------
    3) 전체 조사 평균 대비
       Response(%) = (관측값 - 전체 조사 평균) / |전체 조사 평균| × 100
    4) 생육추세 기대값 대비(기존)
       절대 관측값의 조사순서 추세 대비 편차율
    5) 직전 조사 대비
       Response(%) = (현재값 - 직전값) / |직전값| × 100
    6) 수동 기준값 대비
       연구자가 입력한 절대 기준값 대비 편차율

    상태 판정
    ---------
    증가: Response > +stable_band_pct
    유지: -stable_band_pct <= Response <= +stable_band_pct
    감소: danger_pct < Response < -stable_band_pct
    위험: Response <= danger_pct

    주의
    ----
    본 곡선은 관찰적/기술적 반응 시각화입니다.
    Centered ALE처럼 다른 Feature의 상관구조를 모델 내부에서 반영한 효과곡선과 동일한 의미가 아닙니다.
    논문에서는 반응곡선 전환 후보를 ALE/Bootstrap CI와 교차검증하는 것을 권장합니다.
    """
    if (
        gei_df is None
        or gei_df.empty
        or gei_feature not in gei_df.columns
        or target_col not in gei_df.columns
    ):
        return pd.DataFrame(), {}

    cols = ["조사일자", gei_feature, target_col]
    work = gei_df[cols].copy()
    work["조사일자"] = pd.to_datetime(work["조사일자"], errors="coerce")
    work[gei_feature] = pd.to_numeric(work[gei_feature], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = (
        work.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["조사일자", gei_feature, target_col])
        .sort_values("조사일자")
        .reset_index(drop=True)
    )

    if len(work) < 3:
        return pd.DataFrame(), {}

    # --------------------------------------------------------
    # 시간축 및 변화량 파생
    # --------------------------------------------------------
    work["변화량"] = work[target_col].diff()
    work["조사간격(일)"] = work["조사일자"].diff().dt.total_seconds() / 86400.0
    valid_gap = work["조사간격(일)"].where(work["조사간격(일)"] > 0)
    work["일평균 변화량"] = work["변화량"] / valid_gap
    work["7일환산 변화량"] = work["일평균 변화량"] * 7.0
    work["조사월"] = work["조사일자"].dt.to_period("M").astype(str)
    first_date = work["조사일자"].min()
    work["경과일"] = (work["조사일자"] - first_date).dt.total_seconds() / 86400.0
    work["경과주"] = work["경과일"] / 7.0

    target_values = work[target_col].to_numpy(dtype=float)
    n = len(work)
    baseline_kind = "absolute"
    response_value_col = target_col
    response_value_label = f"{target_col} 관측값"

    # --------------------------------------------------------
    # 기준값 생성
    # --------------------------------------------------------
    if baseline_mode == "월평균 변화량 대비(신규 권장)":
        baseline_kind = "growth_change"
        response_value_col = "7일환산 변화량"
        response_value_label = "7일환산 변화량"
        baseline_series = pd.Series(np.nan, index=work.index, dtype=float)
        global_change_mean = float(pd.to_numeric(work[response_value_col], errors="coerce").mean())

        for month, idxs in work.groupby("조사월").groups.items():
            idxs = list(idxs)
            month_vals = pd.to_numeric(work.loc[idxs, response_value_col], errors="coerce")
            valid_month = month_vals.dropna()
            for idx in idxs:
                current_val = work.at[idx, response_value_col]
                if pd.isna(current_val):
                    continue
                others = valid_month.drop(index=idx, errors="ignore")
                if len(others) >= 2:
                    baseline_series.at[idx] = float(others.mean())
                elif len(valid_month) >= 1:
                    baseline_series.at[idx] = float(valid_month.mean())
                elif np.isfinite(global_change_mean):
                    baseline_series.at[idx] = global_change_mean

        baseline = baseline_series.to_numpy(dtype=float)
        baseline_label = "월별 7일환산 평균 변화량(가능 시 LOO)"

    elif baseline_mode == "생육단계 기대 변화량 대비(최종 권장)":
        baseline_kind = "growth_change"
        response_value_col = "7일환산 변화량"
        response_value_label = "7일환산 변화량"
        x_stage = work["경과일"].to_numpy(dtype=float)
        y_change = pd.to_numeric(work[response_value_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x_stage) & np.isfinite(y_change)
        baseline = np.full(n, np.nan, dtype=float)

        if valid.sum() >= 4:
            degree = 2 if valid.sum() >= 7 else 1
            try:
                coeff = np.polyfit(x_stage[valid], y_change[valid], deg=degree)
                baseline = np.polyval(coeff, x_stage)
                baseline_label = f"생육단계별 {degree}차 기대 7일 변화량"
            except Exception:
                mean_change = float(np.nanmean(y_change[valid]))
                baseline[:] = mean_change
                baseline_label = "전체 7일환산 평균 변화량"
        else:
            mean_change = float(np.nanmean(y_change[valid])) if valid.any() else np.nan
            baseline[:] = mean_change
            baseline_label = "전체 7일환산 평균 변화량"

    elif baseline_mode == "수동 기준값 대비":
        try:
            manual_value = float(manual_baseline)
        except (TypeError, ValueError):
            manual_value = float(np.nanmean(target_values))
        baseline = np.full(n, manual_value, dtype=float)
        baseline_label = f"수동 기준값({manual_value:.3f})"

    elif baseline_mode == "직전 조사 대비":
        baseline = work[target_col].shift(1).to_numpy(dtype=float)
        baseline_label = "직전 조사값"

    elif baseline_mode == "생육추세 기대값 대비(기존)":
        x_time = np.arange(n, dtype=float)
        degree = 2 if n >= 6 else 1
        try:
            coeff = np.polyfit(x_time, target_values, deg=degree)
            baseline = np.polyval(coeff, x_time)
            baseline_label = f"{degree}차 조사일 절대값 추세 기대값"
        except Exception:
            mean_value = float(np.nanmean(target_values))
            baseline = np.full(n, mean_value, dtype=float)
            baseline_label = "전체 조사 평균"

    else:
        mean_value = float(np.nanmean(target_values))
        baseline = np.full(n, mean_value, dtype=float)
        baseline_label = "전체 조사 평균"

    if response_value_col == target_col:
        response_values = target_values.copy()
    else:
        response_values = pd.to_numeric(work[response_value_col], errors="coerce").to_numpy(dtype=float)

    denominator = np.abs(baseline)
    # 기대 변화량이 0에 너무 가까우면 %가 폭발하므로 데이터 스케일 기반 안전 하한 적용
    finite_response = np.abs(response_values[np.isfinite(response_values)])
    scale_floor = max(1e-9, float(np.nanmedian(finite_response)) * 0.05) if finite_response.size else 1e-9
    denominator = np.where(
        np.isfinite(denominator) & (denominator > scale_floor),
        denominator,
        np.nan,
    )

    response_pct = (response_values - baseline) / denominator * 100.0

    work["반응대상값"] = response_values
    work["기준값"] = baseline
    work["반응률(%)"] = response_pct

    work = (
        work.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[gei_feature, target_col, "반응대상값", "기준값", "반응률(%)"])
        .reset_index(drop=True)
    )

    if work.empty:
        return pd.DataFrame(), {}

    stable_band_pct = abs(float(stable_band_pct))
    danger_pct = float(danger_pct)
    if danger_pct >= -stable_band_pct:
        danger_pct = -max(10.0, stable_band_pct * 2.0)

    conditions = [
        work["반응률(%)"] > stable_band_pct,
        work["반응률(%)"] >= -stable_band_pct,
        work["반응률(%)"] > danger_pct,
    ]
    choices = ["↑ 증가", "→ 유지", "↓ 감소"]
    work["생육반응 상태"] = np.select(conditions, choices, default="⚠ 위험")

    # --------------------------------------------------------
    # GEI 크기순 반응 추세
    # --------------------------------------------------------
    work = work.sort_values([gei_feature, "조사일자"]).reset_index(drop=True)
    smooth_window = max(3, min(7, int(round(len(work) * 0.20))))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth_window = min(smooth_window, len(work))

    if smooth_window >= 3:
        work["GEI 반응 추세(%)"] = (
            work["반응률(%)"].rolling(window=smooth_window, center=True, min_periods=2).mean()
        )
    else:
        work["GEI 반응 추세(%)"] = work["반응률(%)"]

    trend = work["GEI 반응 추세(%)"].to_numpy(dtype=float)
    gei_values = work[gei_feature].to_numpy(dtype=float)
    decrease_threshold = np.nan
    danger_threshold = np.nan

    for idx in range(len(work)):
        current = trend[idx]
        next_value = trend[idx + 1] if idx + 1 < len(work) else current
        if (
            not np.isfinite(decrease_threshold)
            and np.isfinite(current)
            and np.isfinite(next_value)
            and current < -stable_band_pct
            and next_value < -stable_band_pct
        ):
            decrease_threshold = float(gei_values[idx])
        if (
            not np.isfinite(danger_threshold)
            and np.isfinite(current)
            and np.isfinite(next_value)
            and current <= danger_pct
            and next_value <= danger_pct
        ):
            danger_threshold = float(gei_values[idx])

    finite_trend = np.isfinite(trend)
    best_gei = np.nan
    best_response = np.nan
    if np.any(finite_trend):
        valid_idx = np.where(finite_trend)[0]
        best_local_idx = int(valid_idx[np.nanargmax(trend[valid_idx])])
        best_gei = float(gei_values[best_local_idx])
        best_response = float(trend[best_local_idx])

    unique_gei = work[gei_feature].nunique()
    if unique_gei >= 4:
        q = min(6, max(4, len(work) // 5))
        q = min(q, unique_gei)
        try:
            work["GEI 반응구간"] = pd.qcut(work[gei_feature], q=q, duplicates="drop")
        except Exception:
            work["GEI 반응구간"] = pd.cut(work[gei_feature], bins=min(5, unique_gei), duplicates="drop")
    else:
        work["GEI 반응구간"] = "전체"

    summary_rows = []
    grouped = work.groupby("GEI 반응구간", observed=True, dropna=False)
    for interval, group in grouped:
        if group.empty:
            continue
        summary_rows.append({
            "GEI 구간": str(interval),
            "GEI 최소": float(group[gei_feature].min()),
            "GEI 최대": float(group[gei_feature].max()),
            "표본수": int(len(group)),
            "평균 반응률(%)": float(group["반응률(%)"].mean()),
            "중앙 반응률(%)": float(group["반응률(%)"].median()),
            "증가 비율(%)": float((group["생육반응 상태"] == "↑ 증가").mean() * 100.0),
            "감소+위험 비율(%)": float(group["생육반응 상태"].isin(["↓ 감소", "⚠ 위험"]).mean() * 100.0),
        })
    interval_summary = pd.DataFrame(summary_rows)

    metadata = {
        "baseline_mode": baseline_mode,
        "baseline_label": baseline_label,
        "baseline_kind": baseline_kind,
        "response_value_col": response_value_col,
        "response_value_label": response_value_label,
        "stable_band_pct": stable_band_pct,
        "danger_pct": danger_pct,
        "smooth_window": int(smooth_window),
        "decrease_threshold": float(decrease_threshold) if np.isfinite(decrease_threshold) else np.nan,
        "danger_threshold": float(danger_threshold) if np.isfinite(danger_threshold) else np.nan,
        "best_gei": best_gei,
        "best_response": best_response,
        "target_mean": float(np.nanmean(target_values)),
        "n": int(len(work)),
        "interval_summary": interval_summary,
    }
    return work, metadata

def explain_gei_growth_response_curve(
    response_df,
    metadata,
    gei_feature,
    target_col,
    ale_threshold=None,
):
    """GEI 기반 생육 반응 곡선의 자동 해석 문구."""
    if (
        response_df is None
        or response_df.empty
        or not metadata
    ):
        return (
            "GEI 기반 생육 반응 곡선을 "
            "해석할 유효 데이터가 부족합니다."
        )

    n = int(metadata.get("n", len(response_df)))
    baseline_label = metadata.get(
        "baseline_label",
        "기준값",
    )
    stable_band = float(
        metadata.get(
            "stable_band_pct",
            2.0,
        )
    )
    danger_pct = float(
        metadata.get(
            "danger_pct",
            -10.0,
        )
    )
    decrease_threshold = metadata.get(
        "decrease_threshold",
        np.nan,
    )
    danger_threshold = metadata.get(
        "danger_threshold",
        np.nan,
    )
    best_gei = metadata.get(
        "best_gei",
        np.nan,
    )
    best_response = metadata.get(
        "best_response",
        np.nan,
    )

    state_counts = (
        response_df[
            "생육반응 상태"
        ]
        .value_counts()
        .to_dict()
    )

    parts = [
        (
            f"총 {n}개 조사일에서 "
            f"{gei_feature}와 {target_col}을 연결했습니다."
        ),
        (
            f"반응률은 '{baseline_label}'을 기준으로 계산했으며, "
            f"±{stable_band:.1f}%를 유지영역, "
            f"{danger_pct:.1f}% 이하를 위험영역으로 설정했습니다."
        ),
    ]

    if metadata.get("baseline_kind") == "growth_change":
        parts.append(
            "이 모드의 0%는 해당 시기의 기대 성장량과 동일함을 뜻합니다. "
            "음의 반응률은 초장 자체가 줄었다는 뜻이 아니라, 실제 성장속도가 기대 성장속도보다 낮았다는 의미입니다."
        )

    if np.isfinite(best_gei):
        parts.append(
            f"GEI 정렬 추세에서 가장 우호적인 반응은 "
            f"{gei_feature}≈{best_gei:.2f} 부근 "
            f"({best_response:+.2f}%)에서 관찰되었습니다."
        )

    if np.isfinite(
        decrease_threshold
    ):
        parts.append(
            f"이동평균 추세가 연속적으로 감소영역에 들어가는 "
            f"첫 후보는 {gei_feature}≈"
            f"{decrease_threshold:.2f}입니다."
        )
    else:
        parts.append(
            "현재 표본에서는 연속적인 감소 시작점이 "
            "뚜렷하게 탐지되지 않았습니다."
        )

    if np.isfinite(
        danger_threshold
    ):
        parts.append(
            f"위험 반응({danger_pct:.1f}% 이하)이 연속되는 "
            f"후보는 {gei_feature}≈"
            f"{danger_threshold:.2f}입니다."
        )

    if (
        ale_threshold is not None
        and np.isfinite(
            float(ale_threshold)
        )
    ):
        parts.append(
            f"별도의 RandomForest Centered ALE 감소 후보는 "
            f"{float(ale_threshold):.2f}입니다. "
            "두 임계점이 비슷하면 관찰적 반응과 모델 기반 반응이 "
            "같은 방향을 지지하는 근거가 되지만, "
            "차이가 크면 표본수·비선형성·다른 GEI의 상관구조를 점검해야 합니다."
        )

    parts.append(
        "상태별 조사일 수는 "
        + ", ".join(
            [
                f"{key} {value}개"
                for key, value
                in state_counts.items()
            ]
        )
        + "입니다."
    )

    parts.append(
        "이 그래프는 25개 안팎의 조사일 원자료를 모두 표시하는 "
        "기술적 반응곡선이므로, ALE의 5~10개 bin 중심점보다 "
        "실제 조사일별 GEI-생육 변화를 훨씬 직접적으로 확인할 수 있습니다."
    )

    return " ".join(parts)


def make_environment_zone_reference_table(zone_config):
    rows = []
    for env_name, cfg in zone_config.items():
        for idx, label in enumerate(cfg["labels"]):
            rows.append(
                {
                    "환경요인": env_name,
                    "환경구간": label,
                    "범위": _finite_zone_text(cfg["edges"][idx], cfg["edges"][idx + 1], cfg["unit"]),
                    "위험가중치": cfg["weights"][idx],
                    "GEI분모포함": (
                        "제외"
                        if label in set(cfg.get("exclude_from_gei_denominator", []))
                        else "포함"
                    ),
                }
            )
    return pd.DataFrame(rows)


def render_zone_hours_chart(selected_row, env_name, zone_config):
    cfg = zone_config[env_name]
    x = [float(selected_row.get(f"{label} 누적시간(h)", 0.0) or 0.0) for label in cfg["labels"]]

    if env_name == "일사량":
        # L0는 야간/무일사이므로 회색으로 분리해 스트레스 구간과 시각적으로 구분
        zone_colors = ["#94a3b8", "#60a5fa", "#22c55e", "#facc15", "#fb923c", "#dc2626", "#111827"]
    else:
        zone_colors = ["#ef4444", "#60a5fa", "#22c55e", "#facc15", "#fb923c", "#dc2626", "#7c3aed", "#111827"]

    zone_colors = zone_colors[:len(cfg["labels"])]
    fig = go.Figure(
        go.Bar(
            x=x,
            y=cfg["labels"],
            orientation="h",
            text=[f"{v:.1f}h" for v in x],
            textposition="outside",
            marker=dict(
                color=zone_colors,
                line=dict(color="rgba(15,23,42,0.15)", width=1),
            ),
            hovertemplate="%{y}<br>누적시간=%{x:.2f}h<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{env_name} 환경구간 누적시간",
        height=330,
        margin=dict(l=10, r=30, t=55, b=35),
        xaxis_title="누적시간(h)",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
        showlegend=False,
    )
    return fig




def build_gei_window_datasets(
    sensor_df,
    yield_df,
    date_col_sensor,
    date_col_yield,
    env_column_map,
    growth_column_map,
    zone_config,
    start_date=None,
    end_date=None,
    weeks=tuple(range(1, 8)),
):
    """조사일 직전 1~7주 환경만 이용하여 주차별 GEI 데이터셋을 생성합니다."""
    result = {}
    for week in weeks:
        result[int(week)] = build_weekly_gei_dataset(
            sensor_df=sensor_df,
            yield_df=yield_df,
            date_col_sensor=date_col_sensor,
            date_col_yield=date_col_yield,
            env_column_map=env_column_map,
            growth_column_map=growth_column_map,
            zone_config=zone_config,
            window_days=int(week) * 7,
            start_date=start_date,
            end_date=end_date,
        )
    return result


def make_gei_window_wide_table(gei_week_dfs, gei_feature="통합 GEI"):
    """행=조사일, 열=1~7주인 GEI 비교표/Heatmap용 데이터셋을 만듭니다."""
    merged = None
    for week in sorted(gei_week_dfs):
        wk = gei_week_dfs[week]
        if wk is None or wk.empty or gei_feature not in wk.columns:
            continue
        part = wk[["조사일자", gei_feature]].copy()
        part["조사일자"] = pd.to_datetime(part["조사일자"], errors="coerce")
        part = part.rename(columns={gei_feature: f"{week}주"})
        merged = part if merged is None else merged.merge(part, on="조사일자", how="outer")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("조사일자").reset_index(drop=True)


def evaluate_gei_windows(
    gei_week_dfs,
    target_col,
    model_name="RandomForest",
    test_size=0.25,
):
    """
    1~7주 GEI의 생육·수확 예측성능을 동일 조사일과 동일 시간순 분할로 비교합니다.

    입력 Feature: 온도/습도/CO₂/일사량 GEI + 통합 GEI(존재하는 컬럼만)
    추천 기준: R² 최대, 동률이면 MAE 최소
    """
    if not gei_week_dfs or not target_col:
        return pd.DataFrame()

    gei_candidates = ["온도 GEI", "습도 GEI", "CO₂ GEI", "일사량 GEI", "통합 GEI"]
    prepared = {}
    common_dates = None

    for week, wk in gei_week_dfs.items():
        if wk is None or wk.empty or target_col not in wk.columns:
            continue
        features = [c for c in gei_candidates if c in wk.columns]
        if not features:
            continue
        part = wk[["조사일자"] + features + [target_col]].copy()
        part["조사일자"] = pd.to_datetime(part["조사일자"], errors="coerce")
        part[features + [target_col]] = part[features + [target_col]].apply(pd.to_numeric, errors="coerce")
        part = part.replace([np.inf, -np.inf], np.nan).dropna().sort_values("조사일자")
        if part.empty:
            continue
        prepared[int(week)] = (part, features)
        dates = set(part["조사일자"].tolist())
        common_dates = dates if common_dates is None else common_dates.intersection(dates)

    if not prepared or not common_dates:
        return pd.DataFrame()

    common_dates = sorted(common_dates)
    if len(common_dates) < 8:
        return pd.DataFrame()

    n_test = max(2, int(np.ceil(len(common_dates) * float(test_size))))
    if len(common_dates) - n_test < 4:
        n_test = max(2, len(common_dates) - 4)
    train_dates = set(common_dates[:-n_test])
    test_dates = set(common_dates[-n_test:])
    rows = []

    for week in sorted(prepared):
        part, features = prepared[week]
        part = part[part["조사일자"].isin(common_dates)].sort_values("조사일자")
        train = part[part["조사일자"].isin(train_dates)]
        test = part[part["조사일자"].isin(test_dates)]
        try:
            # v27.7: 일반 모델 선택과 동일한 factory를 사용하여
            # RF/GBR/XGB/LGBM + ANN/BPM/SVM을 GEI 1~7주 비교에도 지원합니다.
            model = make_model(model_name)
            model.fit(train[features], train[target_col])
            pred = model.predict(test[features])
            mse = mean_squared_error(test[target_col], pred)
            mae = mean_absolute_error(test[target_col], pred)
            r2 = r2_score(test[target_col], pred) if len(test) >= 2 and test[target_col].nunique() > 1 else np.nan
            corr = part["통합 GEI"].corr(part[target_col]) if "통합 GEI" in part.columns else np.nan
            rows.append({
                "환경 누적기간(주)": int(week),
                "예측대상": target_col,
                "모델": model_name,
                "MSE": float(mse),
                "MAE": float(mae),
                "R2": float(r2) if np.isfinite(r2) else np.nan,
                "GEI-대상 상관계수": float(corr) if np.isfinite(corr) else np.nan,
                "Feature수": len(features),
                "공통 데이터수": len(common_dates),
                "학습 데이터수": len(train),
                "평가 데이터수": len(test),
                "오류": "",
            })
        except Exception as exc:
            rows.append({
                "환경 누적기간(주)": int(week), "예측대상": target_col,
                "모델": model_name, "MSE": np.nan, "MAE": np.nan,
                "R2": np.nan, "GEI-대상 상관계수": np.nan,
                "Feature수": len(features), "공통 데이터수": len(common_dates),
                "학습 데이터수": len(train), "평가 데이터수": len(test),
                "오류": str(exc),
            })
    return pd.DataFrame(rows).sort_values("환경 누적기간(주)").reset_index(drop=True)


def apply_best_gei_window(best_week):
    """추천된 GEI 기간을 환경 누적기간 위젯에 적용합니다."""
    st.session_state["gei_window_days"] = int(best_week) * 7


def render_gei_window_optimizer(
    gei_week_dfs,
    growth_targets,
    selected_window_days,
):
    """GEI 1~7주 비교표, 조사일×주차 Heatmap, 최고기간 자동추천 UI."""
    render_stylish_section(
        "🏆 GEI 1~7주 비교·Heatmap·최고기간 자동추천",
        "조사일 직전까지만 센서데이터를 사용하여 1~7주 GEI를 각각 계산하고, 동일 조사일·동일 시간순 평가구간에서 생육·수확 예측성능을 공정하게 비교합니다.",
        kicker="GEI X-WINDOW OPTIMIZER",
    )

    control_a, control_b, control_c = st.columns([1.25, 1.0, 1.0])
    with control_a:
        target = st.selectbox(
            "최고 GEI 기간을 찾을 생육·수확 항목",
            list(growth_targets),
            key="gei_optimizer_target",
        )
    with control_b:
        gei_feature = st.selectbox(
            "Heatmap에 표시할 GEI",
            ["통합 GEI", "온도 GEI", "습도 GEI", "CO₂ GEI", "일사량 GEI"],
            key="gei_heatmap_feature",
        )
    with control_c:
        optimizer_model = st.selectbox(
            "비교 모델",
            [
                "RandomForest",
                "GradientBoosting",
                "XGBoost",
                "LGBM",
                "ANN(인공신경망)",
                "BPM(베이지안 확률 모델)",
                "SVM(서포트벡터머신)",
            ],
            key="gei_optimizer_model",
        )

    wide = make_gei_window_wide_table(gei_week_dfs, gei_feature=gei_feature)
    if wide.empty:
        st.warning(f"{gei_feature}의 1~7주 비교데이터를 만들 수 없습니다.")
        return pd.DataFrame(), None

    graph_col, table_col = st.columns([1.15, 0.85], gap="large")
    with graph_col:
        render_panel_label(f"조사일 × 누적기간 {gei_feature} Heatmap")
        matrix = wide.set_index("조사일자")[[c for c in wide.columns if c != "조사일자"]]
        heat_text = np.where(
            np.isfinite(matrix.to_numpy(dtype=float)),
            np.vectorize(lambda v: f"{v:.1f}")(matrix.to_numpy(dtype=float)),
            "",
        )
        fig_heat = go.Figure(
            go.Heatmap(
                z=matrix.to_numpy(dtype=float),
                x=matrix.columns.tolist(),
                y=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in matrix.index],
                text=heat_text,
                texttemplate="%{text}",
                colorscale="RdYlGn_r",
                zmin=0,
                zmax=100,
                colorbar=dict(title="GEI"),
                hovertemplate="조사일=%{y}<br>기간=%{x}<br>GEI=%{z:.2f}<extra></extra>",
            )
        )
        fig_heat.update_layout(
            height=max(430, min(850, 190 + len(matrix) * 18)),
            xaxis_title="환경 누적기간",
            yaxis_title="조사일",
            margin=dict(l=85, r=25, t=45, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.82)",
        )
        st.plotly_chart(fig_heat, use_container_width=True, key="gei_1_7_heatmap")

    with table_col:
        render_panel_label(f"{gei_feature} 1~7주 조사일별 비교표")
        table_show = wide.copy()
        table_show["조사일자"] = pd.to_datetime(table_show["조사일자"]).dt.strftime("%Y-%m-%d")
        st.dataframe(table_show.round(2), use_container_width=True, hide_index=True, height=520)
        st.download_button(
            "⬇️ GEI 1~7주 비교표 CSV 다운로드",
            data=wide.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"GEI_1to7weeks_{gei_feature.replace(' ', '_')}.csv",
            mime="text/csv",
            key="download_gei_1to7_table",
        )

    with st.spinner("GEI 1~7주 예측성능을 동일 조사일과 시간순 분할로 비교하고 있습니다..."):
        performance = evaluate_gei_windows(
            gei_week_dfs=gei_week_dfs,
            target_col=target,
            model_name=optimizer_model,
            test_size=0.25,
        )

    valid = performance.dropna(subset=["R2"]).copy() if not performance.empty else pd.DataFrame()
    if valid.empty:
        st.warning("최고기간 추천을 계산할 공통 유효자료가 부족합니다. 1~7주 모두에서 공통으로 존재하는 조사일이 최소 8개 필요합니다.")
        if not performance.empty:
            st.dataframe(performance.round(4), use_container_width=True, hide_index=True)
        return performance, None

    valid = valid.sort_values(["R2", "MAE"], ascending=[False, True])
    best = valid.iloc[0]
    best_week = int(best["환경 누적기간(주)"])
    best_r2 = float(best["R2"])
    current_week = int(selected_window_days) // 7
    current_rows = valid[valid["환경 누적기간(주)"] == current_week]
    current_r2 = float(current_rows.iloc[0]["R2"]) if not current_rows.empty else np.nan

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("현재 GEI 누적기간", f"{current_week}주")
    with m2:
        st.metric("현재기간 R²", f"{current_r2:.4f}" if np.isfinite(current_r2) else "계산불가")
    with m3:
        st.metric("추천 GEI 누적기간", f"{best_week}주")
    with m4:
        st.metric("최고 R²", f"{best_r2:.4f}", delta=(f"{best_r2-current_r2:+.4f}" if np.isfinite(current_r2) else None))

    chart_col, perf_col = st.columns([1.05, 0.95], gap="large")
    with chart_col:
        render_panel_label("GEI 누적기간별 R² 비교")
        chart = performance.dropna(subset=["R2"]).sort_values("환경 누적기간(주)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart["환경 누적기간(주)"], y=chart["R2"],
            mode="lines+markers+text",
            text=[f"{v:.3f}" for v in chart["R2"]],
            textposition="top center",
            line=dict(width=4, color="#2563eb", shape="spline"),
            marker=dict(size=11, color="#ffffff", line=dict(width=3, color="#2563eb")),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.10)", name="R²",
        ))
        fig.add_trace(go.Scatter(
            x=[best_week], y=[best_r2], mode="markers",
            marker=dict(size=21, symbol="star", color="#f59e0b", line=dict(width=2, color="#ffffff")),
            name="추천기간",
        ))
        fig.add_vline(x=current_week, line_dash="dot", line_color="#0f766e", annotation_text=f"현재 {current_week}주")
        fig.update_layout(
            height=410, title=f"{target} 예측 · {optimizer_model}",
            xaxis=dict(title="GEI 환경 누적기간", tickmode="array", tickvals=list(range(1, 8)), ticktext=[f"{w}주" for w in range(1, 8)]),
            yaxis_title="R²", hovermode="x unified",
            margin=dict(l=50, r=25, t=65, b=50),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.82)",
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        )
        st.plotly_chart(fig, use_container_width=True, key="gei_window_r2_chart")

    with perf_col:
        render_panel_label("GEI 1~7주 모델 성능표")
        cols = ["환경 누적기간(주)", "MSE", "MAE", "R2", "GEI-대상 상관계수", "공통 데이터수", "평가 데이터수"]
        st.dataframe(performance[cols].round(4), use_container_width=True, hide_index=True, height=410)

    if best_week == current_week:
        st.success(f"현재 선택한 {current_week}주가 {target} 예측에서 최고 R²={best_r2:.4f}를 보였습니다.")
    else:
        st.success(f"{target}의 추천 GEI 누적기간은 {best_week}주이며 R²={best_r2:.4f}입니다. 현재 {current_week}주보다 더 높은 설명력을 보였습니다.")
        st.button(
            f"✅ 추천기간 {best_week}주를 GEI 분석에 적용",
            key="apply_best_gei_window_button",
            on_click=apply_best_gei_window,
            args=(best_week,),
        )

    st.caption("공정 비교를 위해 1~7주 모두에서 유효한 동일 조사일만 사용하고, 과거 조사일을 학습·최근 조사일을 평가하는 시간순 분할을 적용했습니다. 조사일 당일 및 이후 센서값은 포함하지 않습니다.")
    return performance, best_week


# =============================================================
# v27.1 GEI 이후 환경패턴 클러스터링 기반 생육·수확 반응 분석
# =============================================================
CLUSTER_ARCHETYPES = {
    "Cluster 1 · 흐린 겨울형": np.array([-1.0,  1.0, -1.0, -1.0]),
    "Cluster 2 · 안정 생육형": np.array([ 0.0,  0.0,  0.0,  0.0]),
    "Cluster 3 · 고온·건조 스트레스형": np.array([ 1.0, -1.0,  0.0,  1.0]),
    "Cluster 4 · 고습·저광형": np.array([ 0.0,  1.0,  0.0, -1.0]),
    "Cluster 5 · 고온·고습 복합 스트레스형": np.array([ 1.0,  1.0,  1.0,  0.0]),
}

CLUSTER_DESCRIPTIONS = {
    "Cluster 1 · 흐린 겨울형": "저온 + 고습 + 저CO₂ + 저일사",
    "Cluster 2 · 안정 생육형": "적온 + 적습 + 적정CO₂ + 중일사",
    "Cluster 3 · 고온·건조 스트레스형": "고온 + 저습 + 고일사",
    "Cluster 4 · 고습·저광형": "적온 + 고습 + 저일사",
    "Cluster 5 · 고온·고습 복합 스트레스형": "고온 + 고습 + 고CO₂",
}


def _assign_cluster_archetypes(centers_z):
    """KMeans 중심을 5개 사전 정의 환경유형에 1:1로 자동 매칭합니다."""
    names = list(CLUSTER_ARCHETYPES.keys())
    refs = np.vstack([CLUSTER_ARCHETYPES[n] for n in names])
    n = min(len(centers_z), len(names))
    best_perm, best_cost = None, np.inf
    for perm in permutations(range(len(names)), n):
        cost = sum(float(np.sum((centers_z[i] - refs[j]) ** 2)) for i, j in enumerate(perm))
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    return {i: names[j] for i, j in enumerate(best_perm or range(n))}


def _describe_cluster_from_z(center_z, selected_names):
    """선택된 환경변수의 표준화 중심값으로 자동 환경특성을 생성합니다."""
    parts = []
    for name, z in zip(selected_names, center_z):
        if name == "온도":
            parts.append("고온" if z >= 0.45 else ("저온" if z <= -0.45 else "적온"))
        elif name == "습도":
            parts.append("고습" if z >= 0.45 else ("저습" if z <= -0.45 else "적습"))
        elif name == "CO₂":
            parts.append("고CO₂" if z >= 0.45 else ("저CO₂" if z <= -0.45 else "적정CO₂"))
        elif name == "일사량":
            parts.append("고일사" if z >= 0.45 else ("저일사" if z <= -0.45 else "중일사"))
    return " + ".join(parts) if parts else "데이터 기반 환경유형"


def build_environment_cluster_analysis(sensor_df, date_col, env_cols, n_clusters=5):
    """선택한 2~4개 환경센서 변수를 표준화한 뒤 KMeans 환경패턴을 생성합니다."""
    selected_names = [k for k in ["온도", "습도", "CO₂", "일사량"] if env_cols.get(k) is not None]
    required = [env_cols[k] for k in selected_names if env_cols.get(k) in sensor_df.columns]
    if len(required) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}, np.nan, None

    work = sensor_df[[date_col] + required].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    for c in required:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna().sort_values(date_col).reset_index(drop=True)
    if len(work) < max(30, n_clusters * 5):
        return pd.DataFrame(), pd.DataFrame(), {}, np.nan, None

    X = work[required].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    raw_labels = km.fit_predict(Xz)

    # 4개 변수를 모두 사용하면 기존 5개 archetype을 유지하고, 2~3개 선택 시 데이터 중심형 라벨을 사용합니다.
    if len(selected_names) == 4:
        mapping = _assign_cluster_archetypes(km.cluster_centers_)
        descriptions = {mapping[i]: CLUSTER_DESCRIPTIONS.get(mapping[i], _describe_cluster_from_z(km.cluster_centers_[i], selected_names)) for i in range(n_clusters)}
    else:
        order = np.argsort(km.cluster_centers_[:, 0])
        rank = {int(raw_id): rank_i + 1 for rank_i, raw_id in enumerate(order)}
        mapping = {i: f"Cluster {rank[i]} · 데이터 기반 환경형" for i in range(n_clusters)}
        descriptions = {mapping[i]: _describe_cluster_from_z(km.cluster_centers_[i], selected_names) for i in range(n_clusters)}

    work["환경Cluster"] = [mapping[int(v)] for v in raw_labels]
    centers_original = scaler.inverse_transform(km.cluster_centers_)
    profile_rows = []
    for raw_id in range(n_clusters):
        name = mapping.get(raw_id, f"Cluster {raw_id + 1}")
        mask = raw_labels == raw_id
        row = {"환경Cluster": name, "환경특성": descriptions.get(name, "데이터 기반 환경유형"), "관측수": int(mask.sum()), "비율(%)": float(mask.mean() * 100)}
        for j, env_name in enumerate(selected_names):
            label = {"온도":"온도(℃)", "습도":"습도(%)", "CO₂":"CO₂(ppm)", "일사량":"일사량(W/m²)"}[env_name]
            row[label] = centers_original[raw_id, j]
        profile_rows.append(row)
    profile = pd.DataFrame(profile_rows).sort_values("환경Cluster").reset_index(drop=True)
    profile.attrs["descriptions"] = descriptions
    profile.attrs["selected_names"] = selected_names

    sil = np.nan
    if len(np.unique(raw_labels)) > 1 and len(work) > n_clusters:
        sample_n = min(5000, len(work)); idx = np.linspace(0, len(work)-1, sample_n, dtype=int)
        try: sil = float(silhouette_score(Xz[idx], raw_labels[idx]))
        except Exception: sil = np.nan
    pca_df = None
    try:
        pca = PCA(n_components=2, random_state=42); coords = pca.fit_transform(Xz)
        pca_df = pd.DataFrame({"PC1":coords[:,0], "PC2":coords[:,1], "환경Cluster":work["환경Cluster"].values})
        pca_df.attrs["explained_variance"] = pca.explained_variance_ratio_.tolist()
    except Exception: pca_df = None
    return work, profile, mapping, sil, pca_df



def _gei_rule_score(zone_index, env_name):
    """GEI 구간 번호를 Rule-based 환경상태 점수(-1~1)로 변환합니다."""
    maps = {
        "온도":   [-1.0, -0.75, -0.35, 0.0, 0.55, 1.0],
        "습도":   [-1.0, -0.65, 0.0, 0.45, 0.75, 1.0],
        "CO₂":    [-1.0, -0.65, -0.30, 0.0, 0.55, 1.0],
        # L0는 야간/무일사이므로 생리적 저일사와 분리하여 중립에 가깝게 둡니다.
        "일사량": [0.0, -1.0, -0.65, -0.25, 0.0, 0.55, 1.0],
    }
    vals = maps.get(env_name, [])
    if zone_index is None or zone_index < 0 or zone_index >= len(vals):
        return np.nan
    return float(vals[zone_index])


def build_gei_rule_cluster_analysis(sensor_df, date_col, env_cols, n_clusters=5):
    """
    기존 GEI 환경구간(T/H/C/L)을 이용한 Rule-based 환경유형 분류.
    KMeans로 경계를 새로 학습하지 않고, 각 10분 센서값을 기존 GEI 구간으로 먼저 분류한 뒤
    구간 상태벡터와 5개 환경 archetype 간 거리가 가장 가까운 유형에 배정합니다.
    PCA는 원 센서값이 아니라 GEI 구간 기반 Rule score를 2차원으로 투영합니다.
    """
    cfg = get_default_gei_zone_config()
    selected_names = [k for k in ["온도", "습도", "CO₂", "일사량"] if env_cols.get(k) is not None]
    required = [env_cols[k] for k in selected_names if env_cols.get(k) in sensor_df.columns]
    if len(required) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}, np.nan, None

    work = sensor_df[[date_col] + required].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    for c in required:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna().sort_values(date_col).reset_index(drop=True)
    if len(work) < 20:
        return pd.DataFrame(), pd.DataFrame(), {}, np.nan, None

    rule_matrix, zone_text_cols = [], {}
    for env_name in selected_names:
        col = env_cols[env_name]
        conf = cfg[env_name]
        cats = classify_environment_series(work[col], conf)
        work[f"{env_name} GEI구간"] = cats.astype(str)
        zone_text_cols[env_name] = f"{env_name} GEI구간"
        codes = pd.cut(pd.to_numeric(work[col], errors="coerce"), bins=conf["edges"], labels=False, include_lowest=True, right=False)
        scores = pd.Series(codes).apply(lambda x: _gei_rule_score(int(x), env_name) if pd.notna(x) else np.nan)
        rule_matrix.append(scores.to_numpy(dtype=float))

    Xrule = np.column_stack(rule_matrix)
    valid_mask = np.isfinite(Xrule).all(axis=1)
    work = work.loc[valid_mask].reset_index(drop=True)
    Xrule = Xrule[valid_mask]
    if len(work) < 20:
        return pd.DataFrame(), pd.DataFrame(), {}, np.nan, None

    # 선택 변수에 맞춰 기존 5개 archetype을 투영하여 가장 가까운 규칙형 환경유형으로 배정합니다.
    full_order = ["온도", "습도", "CO₂", "일사량"]
    selected_idx = [full_order.index(k) for k in selected_names]
    archetype_names = list(CLUSTER_ARCHETYPES.keys())
    refs = np.vstack([CLUSTER_ARCHETYPES[n][selected_idx] for n in archetype_names])
    d2 = ((Xrule[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    work["환경Cluster"] = [archetype_names[int(i)] for i in labels]
    work["GEI 구간조합"] = work[[zone_text_cols[k] for k in selected_names]].astype(str).agg(" / ".join, axis=1)

    profile_rows = []
    descriptions = {n: CLUSTER_DESCRIPTIONS[n] for n in archetype_names}
    for cid, name in enumerate(archetype_names):
        mask = labels == cid
        row = {"환경Cluster": name, "환경특성": descriptions[name], "관측수": int(mask.sum()), "비율(%)": float(mask.mean()*100)}
        if mask.any():
            for env_name in selected_names:
                label = {"온도":"온도(℃)", "습도":"습도(%)", "CO₂":"CO₂(ppm)", "일사량":"일사량(W/m²)"}[env_name]
                row[label] = float(pd.to_numeric(work.loc[mask, env_cols[env_name]], errors="coerce").mean())
            combos = work.loc[mask, "GEI 구간조합"].value_counts()
            row["대표 GEI 구간조합"] = combos.index[0] if not combos.empty else "-"
        profile_rows.append(row)
    profile = pd.DataFrame(profile_rows)
    profile.attrs["descriptions"] = descriptions
    profile.attrs["selected_names"] = selected_names

    # Rule-based는 군집 경계를 학습하지 않으므로 Silhouette은 참고지표로만 계산합니다.
    sil = np.nan
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and len(work) > len(unique_labels):
        try:
            idx = np.linspace(0, len(work)-1, min(5000, len(work)), dtype=int)
            sil = float(silhouette_score(Xrule[idx], labels[idx]))
        except Exception:
            sil = np.nan

    pca_df = None
    try:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Xrule)
        pca_df = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "환경Cluster": work["환경Cluster"].values})
        pca_df.attrs["explained_variance"] = pca.explained_variance_ratio_.tolist()
    except Exception:
        pass
    mapping = {i: archetype_names[i] for i in range(len(archetype_names))}
    return work, profile, mapping, sil, pca_df


def _render_cluster_pca(pca_df, cluster_names, key, title):
    render_panel_label(title)
    if pca_df is None or pca_df.empty:
        st.info("PCA 2D를 계산할 수 없습니다.")
        return
    fig = go.Figure()
    for name in cluster_names:
        part = pca_df[pca_df["환경Cluster"] == name]
        if part.empty:
            continue
        if len(part) > 1200:
            part = part.iloc[np.linspace(0, len(part)-1, 1200, dtype=int)]
        fig.add_trace(go.Scattergl(x=part["PC1"], y=part["PC2"], mode="markers", name=name, marker=dict(size=5, opacity=0.55)))
    ev = pca_df.attrs.get("explained_variance", [np.nan, np.nan])
    fig.update_layout(height=420, xaxis_title=f"PC1 ({ev[0]*100:.1f}%)" if np.isfinite(ev[0]) else "PC1",
                      yaxis_title=f"PC2 ({ev[1]*100:.1f}%)" if np.isfinite(ev[1]) else "PC2",
                      margin=dict(l=45,r=20,t=30,b=45), legend=dict(font=dict(size=10)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.82)")
    st.plotly_chart(fig, use_container_width=True, key=key)

def build_cluster_exposure_by_survey(clustered_sensor, yield_df, sensor_date_col, yield_date_col,
                                     growth_map, window_days, interval_hours):
    """조사일 직전 N일 동안 5개 환경Cluster 노출시간/비율과 생육·수확값을 결합합니다."""
    if clustered_sensor.empty:
        return pd.DataFrame()
    survey = yield_df.copy()
    survey[yield_date_col] = pd.to_datetime(survey[yield_date_col], errors="coerce")
    survey = survey.dropna(subset=[yield_date_col]).sort_values(yield_date_col)
    cluster_names = sorted(clustered_sensor["환경Cluster"].dropna().unique().tolist())
    rows = []
    for _, sr in survey.iterrows():
        d = pd.Timestamp(sr[yield_date_col])
        start = d - pd.Timedelta(days=int(window_days))
        sub = clustered_sensor[(clustered_sensor[sensor_date_col] >= start) & (clustered_sensor[sensor_date_col] < d)]
        counts = sub["환경Cluster"].value_counts().reindex(cluster_names, fill_value=0)
        total = float(counts.sum())
        row = {"조사일자": d}
        for name in cluster_names:
            h = float(counts[name] * interval_hours)
            row[f"{name} 노출시간(h)"] = h
            row[f"{name} 노출비율(%)"] = float(counts[name] / total * 100) if total > 0 else np.nan
        for display_name, col in growth_map.items():
            if col in sr.index:
                row[display_name] = pd.to_numeric(pd.Series([sr[col]]), errors="coerce").iloc[0]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("조사일자").reset_index(drop=True)
    for target in growth_map:
        if target in out.columns:
            out[f"{target} 변화량"] = pd.to_numeric(out[target], errors="coerce").diff()
            prev = pd.to_numeric(out[target], errors="coerce").shift(1)
            out[f"{target} 변화율(%)"] = np.where(prev.abs() > 1e-12, out[f"{target} 변화량"] / prev.abs() * 100, np.nan)
    return out


def render_environment_clustering_after_gei(sensor_df, yield_df, date_col_sensor, date_col_yield,
                                             env_map, growth_map, window_days):
    render_stylish_section(
        "🧩 GEI 이후 · 환경패턴 클러스터링 기반 생육·수확 반응",
        "기존 Data-driven KMeans와 신규 GEI 환경구간 Rule-based 분류를 비교하고, 두 방식의 PCA 2D 환경패턴 및 조사일 직전 노출비율과 생육·수확 변화 관계를 분석합니다.",
        kicker="GEI → DUAL ENVIRONMENT CLUSTERING",
    )
    available_names = [k for k in ["온도", "습도", "CO₂", "일사량"] if env_map.get(k) is not None and env_map.get(k) in sensor_df.columns]
    if len(available_names) < 2:
        st.info("클러스터링에는 최소 2개의 환경변수가 필요합니다.")
        return

    selected_cluster_vars = st.multiselect("클러스터링 사용변수 선택 (2~4개)", options=available_names,
        default=available_names[:4], key="cluster_feature_selector_v274",
        help="두 방법 모두 동일한 선택변수를 사용하여 비교합니다.")
    if len(selected_cluster_vars) < 2:
        st.warning("클러스터링 사용변수를 2개 이상 선택하세요."); return
    selected_cluster_vars = selected_cluster_vars[:4]
    selected_env_map = {k: (env_map.get(k) if k in selected_cluster_vars else None) for k in ["온도", "습도", "CO₂", "일사량"]}

    method_view = st.radio("환경패턴 분류 방법", ["두 방식 비교", "Data-driven · StandardScaler + KMeans", "GEI Rule-based · 환경구간 기준"],
                           horizontal=True, key="cluster_method_view_v274")
    st.caption("Data-driven은 센서 분포에서 KMeans가 군집을 학습하고, GEI Rule-based는 기존 T/H/C/L 환경구간을 먼저 적용한 뒤 규칙형 환경유형으로 분류합니다.")

    dd = build_environment_cluster_analysis(sensor_df, date_col_sensor, selected_env_map, n_clusters=5)
    rb = build_gei_rule_cluster_analysis(sensor_df, date_col_sensor, selected_env_map, n_clusters=5)
    dd_clustered, dd_profile, _, dd_sil, dd_pca = dd
    rb_clustered, rb_profile, _, rb_sil, rb_pca = rb
    if dd_clustered.empty and rb_clustered.empty:
        st.warning("환경 클러스터링에 필요한 유효 센서자료가 부족합니다."); return

    # 두 방식 PCA를 동일 화면에서 비교
    if method_view == "두 방식 비교":
        pc1, pc2 = st.columns(2, gap="large")
        with pc1:
            _render_cluster_pca(dd_pca, dd_profile["환경Cluster"].tolist() if not dd_profile.empty else [], "cluster_pca_dd_v274", "PCA 2D · Data-driven 환경패턴")
        with pc2:
            _render_cluster_pca(rb_pca, rb_profile["환경Cluster"].tolist() if not rb_profile.empty else [], "cluster_pca_rb_v274", "PCA 2D · GEI Rule-based 환경패턴")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Data-driven 관측", f"{len(dd_clustered):,}")
        mc2.metric("Data-driven Silhouette", f"{dd_sil:.3f}" if np.isfinite(dd_sil) else "N/A")
        mc3.metric("Rule-based 관측", f"{len(rb_clustered):,}")
        mc4.metric("Rule-based 분리 참고값", f"{rb_sil:.3f}" if np.isfinite(rb_sil) else "N/A")
        st.caption("Rule-based의 Silhouette은 KMeans 최적화 점수가 아니라, 고정 GEI 규칙으로 분류된 집단의 기하학적 분리도를 사후 확인하는 참고값입니다.")
    elif method_view.startswith("Data-driven"):
        _render_cluster_pca(dd_pca, dd_profile["환경Cluster"].tolist(), "cluster_pca_dd_only_v274", "PCA 2D · Data-driven 환경패턴")
    else:
        _render_cluster_pca(rb_pca, rb_profile["환경Cluster"].tolist(), "cluster_pca_rb_only_v274", "PCA 2D · GEI Rule-based 환경패턴")

    # 중심값/GEI 구간 프로파일 비교
    st.markdown("#### 5개 환경 Cluster 중심값 · 두 방식 비교")
    t1, t2 = st.columns(2, gap="large")
    with t1:
        render_panel_label("Data-driven · KMeans 중심값")
        if not dd_profile.empty: st.dataframe(dd_profile.round(2), use_container_width=True, hide_index=True)
    with t2:
        render_panel_label("GEI Rule-based · 환경구간 기준 프로파일")
        if not rb_profile.empty:
            st.dataframe(
                rb_profile.round(2),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**대표 GEI 구간조합 수동 선택**")
            st.caption(
                "자동으로 가장 빈도가 높은 GEI 구간조합을 대표값으로 제시하며, "
                "연구자 판단에 따라 해당 Cluster에서 실제 관측된 다른 구간조합을 대표값으로 선택할 수 있습니다. "
                "이 선택은 Cluster 배정 자체를 바꾸지 않고 프로파일의 대표 조합 표시를 변경합니다."
            )

            rb_profile_manual = rb_profile.copy()
            for rb_idx, rb_row in rb_profile_manual.iterrows():
                cluster_name = rb_row["환경Cluster"]
                if (
                    "환경Cluster" not in rb_clustered.columns
                    or "GEI 구간조합" not in rb_clustered.columns
                ):
                    continue

                combo_counts = (
                    rb_clustered.loc[
                        rb_clustered["환경Cluster"] == cluster_name,
                        "GEI 구간조합",
                    ]
                    .dropna()
                    .astype(str)
                    .value_counts()
                )
                options = combo_counts.index.tolist()
                if not options:
                    continue

                auto_combo = str(
                    rb_row.get(
                        "대표 GEI 구간조합",
                        options[0],
                    )
                )
                default_idx = (
                    options.index(auto_combo)
                    if auto_combo in options
                    else 0
                )
                selected_combo = st.selectbox(
                    f"{cluster_name} · 대표 GEI 구간조합",
                    options=options,
                    index=default_idx,
                    key=f"rule_profile_combo_manual_{rb_idx}_v275",
                    format_func=lambda x, counts=combo_counts: (
                        f"{x} · 관측 {int(counts.get(x, 0))}회"
                    ),
                )
                rb_profile_manual.loc[
                    rb_idx,
                    "대표 GEI 구간조합",
                ] = selected_combo

            rb_profile = rb_profile_manual
            st.markdown("**수동 선택 반영 Rule-based 프로파일**")
            st.dataframe(
                rb_profile.round(2),
                use_container_width=True,
                hide_index=True,
            )

    # 반응분석에는 사용자가 어느 분류법을 연결할지 명시적으로 선택
    response_method = st.selectbox("클러스터 반응 분석에 연결할 방법", ["Data-driven", "GEI Rule-based"],
                                   index=0, key="cluster_response_method_v274")
    if response_method == "Data-driven":
        clustered, profile = dd_clustered, dd_profile
    else:
        clustered, profile = rb_clustered, rb_profile
    if clustered.empty or profile.empty:
        st.warning(f"{response_method} 결과가 없어 반응분석을 수행할 수 없습니다."); return

    interval_hours = infer_interval_hours(clustered[date_col_sensor])
    exposure = build_cluster_exposure_by_survey(clustered, yield_df, date_col_sensor, date_col_yield, growth_map,
                                                 window_days=window_days, interval_hours=interval_hours)
    st.markdown(f"#### 조사일 기준 Cluster 노출시간 → 생육·수확 변화 · {response_method}")
    if exposure.empty:
        st.warning("조사일 기준 Cluster 노출량을 만들 수 없습니다."); return
    target_options = [k for k in growth_map if k in exposure.columns]
    if not target_options:
        st.warning("클러스터 노출과 비교할 생육·수확 컬럼이 없습니다."); return
    cluster_target = st.selectbox("클러스터 반응 분석 대상", target_options, key="cluster_growth_target_v274")
    change_col = f"{cluster_target} 변화량"
    cluster_names = [n for n in profile["환경Cluster"].tolist() if f"{n} 노출비율(%)" in exposure.columns]
    corr_rows=[]
    for name in cluster_names:
        rcol=f"{name} 노출비율(%)"
        valid=exposure[[rcol,change_col]].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
        corr=valid[rcol].corr(valid[change_col]) if len(valid)>=3 else np.nan
        desc_row=profile.loc[profile["환경Cluster"]==name,"환경특성"] if "환경특성" in profile.columns else pd.Series(dtype=str)
        desc=desc_row.iloc[0] if not desc_row.empty else "환경유형"
        corr_rows.append({"환경Cluster":name,"환경특성":desc,"노출비율-변화량 Pearson r":corr,
                          "평균 노출비율(%)":float(pd.to_numeric(exposure[rcol],errors="coerce").mean())})
    corr_df=pd.DataFrame(corr_rows)
    if corr_df.empty:
        st.warning("반응분석용 Cluster 노출비율을 계산할 수 없습니다."); return

    best=corr_df.dropna(subset=["노출비율-변화량 Pearson r"])
    if not best.empty:
        top=best.iloc[best["노출비율-변화량 Pearson r"].abs().argmax()]
        direction="증가" if top["노출비율-변화량 Pearson r"]>0 else "감소"
        st.markdown(f'<div class="xai-insight-card"><b>{response_method} 기준 · {cluster_target} 변화와 가장 강한 환경유형</b><br>{top["환경Cluster"]} · r={top["노출비율-변화량 Pearson r"]:.3f} → 노출비율이 높을수록 {cluster_target} 변화량이 {direction}하는 경향입니다.<br><small>상관은 인과를 의미하지 않으며 생육단계·계절·조사일 표본수를 함께 검토해야 합니다.</small></div>', unsafe_allow_html=True)

    chart_col, table_col=st.columns([1.15,0.85],gap="large")
    with chart_col:
        render_panel_label(f"{response_method} Cluster 노출비율과 {cluster_target} 변화량")
        fig=go.Figure()
        for name in cluster_names:
            rcol=f"{name} 노출비율(%)"
            fig.add_trace(go.Scatter(x=pd.to_numeric(exposure[rcol],errors="coerce"), y=pd.to_numeric(exposure[change_col],errors="coerce"), mode="markers", name=name, marker=dict(size=8,opacity=0.7)))
        fig.add_hline(y=0,line_dash="dash",line_color="#64748b")
        fig.update_layout(height=430,xaxis_title="조사일 직전 Cluster 노출비율(%)",yaxis_title=f"{cluster_target} 변화량",
                          margin=dict(l=50,r=20,t=35,b=50),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,0.82)")
        st.plotly_chart(fig,use_container_width=True,key=f"cluster_response_{response_method}_v274")
    with table_col:
        render_panel_label("Cluster별 정량 반응 요약")
        st.dataframe(corr_df.round(3),use_container_width=True,hide_index=True,height=430)

    st.caption("확장 분석 흐름: 10분 센서 → [Data-driven: StandardScaler→KMeans] 또는 [Rule-based: GEI T/H/C/L 구간→규칙형 5개 환경유형] → PCA 2D → 조사일 직전 선택기간의 Cluster 노출시간·비율 → 생육·수확 변화량 → Pearson 반응분석. 두 방식의 결과가 일치하면 환경유형 해석의 강건성을 보조적으로 확인할 수 있습니다.")


def build_survey_growth_quantitative_table(gei_df, target_col):
    """
    조사일별 생육·수확 변화량을 조사간격으로 나누어 일평균 변화량을 계산합니다.

    변화상태 기준:
    - 급격증가: 전체 일평균 변화량 평균 + 1표준편차 이상
    - 증가: 평균 초과 ~ 평균+1표준편차 미만
    - 감소: 평균-1표준편차 초과 ~ 평균 이하
    - 급격감소: 평균-1표준편차 이하

    첫 조사일은 이전 조사값이 없으므로 변화량/일평균 변화량이 NaN입니다.
    """
    if (
        gei_df is None
        or gei_df.empty
        or target_col not in gei_df.columns
        or "조사일자" not in gei_df.columns
    ):
        return pd.DataFrame(), {}

    work = gei_df.copy()
    work["조사일자"] = pd.to_datetime(
        work["조사일자"],
        errors="coerce",
    )
    work[target_col] = pd.to_numeric(
        work[target_col],
        errors="coerce",
    )
    work = (
        work.dropna(subset=["조사일자"])
        .sort_values("조사일자")
        .reset_index(drop=True)
    )

    work["월"] = work["조사일자"].dt.strftime("%Y-%m")
    work["조사간격(일)"] = (
        work["조사일자"].diff().dt.total_seconds() / 86400.0
    )
    work["변화량"] = work[target_col].diff()

    valid_days = work["조사간격(일)"].where(
        work["조사간격(일)"] > 0
    )
    work["일평균 변화량"] = (
        work["변화량"] / valid_days
    )

    daily = pd.to_numeric(
        work["일평균 변화량"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)

    mean_daily = float(daily.mean()) if daily.notna().any() else np.nan
    std_daily = float(daily.std(ddof=1)) if daily.notna().sum() >= 2 else 0.0

    upper = mean_daily + std_daily if np.isfinite(mean_daily) else np.nan
    lower = mean_daily - std_daily if np.isfinite(mean_daily) else np.nan

    def classify(v):
        if not np.isfinite(v) or not np.isfinite(mean_daily):
            return "기준없음"
        if v >= upper:
            return "↑↑ 급격증가"
        if v > mean_daily:
            return "↑ 증가"
        if v <= lower:
            return "↓↓ 급격감소"
        return "↓ 감소"

    work["평균대비 변화상태"] = daily.apply(
        lambda v: classify(float(v)) if pd.notna(v) else "기준없음"
    )

    meta = {
        "평균 일변화량": mean_daily,
        "표준편차": std_daily,
        "급격증가 기준": upper,
        "급격감소 기준": lower,
        "유효 조사구간수": int(daily.notna().sum()),
    }
    return work, meta


def build_monthly_environment_growth_summary(
    sensor_df,
    gei_df,
    date_col_sensor,
    env_map,
    target_col,
):
    """
    월별 센서 환경 최저/평균/최고와 선택 생육·수확 항목의
    증가량·감소량·평균·최저·최고를 하나의 표로 결합합니다.
    """
    monthly_parts = []

    if (
        sensor_df is not None
        and not sensor_df.empty
        and date_col_sensor in sensor_df.columns
    ):
        env = sensor_df.copy()
        env[date_col_sensor] = pd.to_datetime(
            env[date_col_sensor],
            errors="coerce",
        )
        env = env.dropna(subset=[date_col_sensor])
        env["월"] = env[date_col_sensor].dt.strftime("%Y-%m")

        agg_dict = {}
        rename_map = {}
        irradiance_cols = set()
        for env_name, col in env_map.items():
            if col is None or col not in env.columns:
                continue
            env[col] = pd.to_numeric(
                env[col],
                errors="coerce",
            )
            if env_name == "일사량":
                # 월별 일사광 최저는 야간 0값을 제외하고 1 이상 유효광 값만 사용합니다.
                valid_light_col = f"__{col}_valid_light_ge1"
                env[valid_light_col] = env[col].where(env[col] >= 1)
                agg_dict[valid_light_col] = ["min"]
                agg_dict[col] = ["mean", "max"]
                irradiance_cols.add((col, valid_light_col))
            else:
                agg_dict[col] = ["min", "mean", "max"]

        if agg_dict:
            env_month = env.groupby("월").agg(agg_dict)
            env_month.columns = [
                f"{col}_{stat}"
                for col, stat in env_month.columns
            ]
            env_month = env_month.reset_index()

            for env_name, col in env_map.items():
                if col is None:
                    continue
                if env_name == "일사량":
                    valid_light_col = f"__{col}_valid_light_ge1"
                    rename_map.update(
                        {
                            f"{valid_light_col}_min": "일사량 최저(≥1)",
                            f"{col}_mean": "일사량 평균",
                            f"{col}_max": "일사량 최고",
                        }
                    )
                else:
                    rename_map.update(
                        {
                            f"{col}_min": f"{env_name} 최저",
                            f"{col}_mean": f"{env_name} 평균",
                            f"{col}_max": f"{env_name} 최고",
                        }
                    )
            env_month = env_month.rename(columns=rename_map)
            monthly_parts.append(env_month)

    growth_month = pd.DataFrame()
    if (
        gei_df is not None
        and not gei_df.empty
        and "조사일자" in gei_df.columns
        and target_col in gei_df.columns
    ):
        g = gei_df[["조사일자", target_col]].copy()
        g["조사일자"] = pd.to_datetime(
            g["조사일자"],
            errors="coerce",
        )
        g[target_col] = pd.to_numeric(
            g[target_col],
            errors="coerce",
        )
        g = (
            g.dropna(subset=["조사일자"])
            .sort_values("조사일자")
            .reset_index(drop=True)
        )
        g["월"] = g["조사일자"].dt.strftime("%Y-%m")
        g["변화량"] = g[target_col].diff()
        g["증가량"] = g["변화량"].clip(lower=0)
        g["감소량"] = g["변화량"].clip(upper=0)

        growth_month = (
            g.groupby("월")
            .agg(
                **{
                    f"{target_col} 증가량 합계": ("증가량", "sum"),
                    f"{target_col} 감소량 합계": ("감소량", "sum"),
                    f"{target_col} 평균": (target_col, "mean"),
                    f"{target_col} 최저": (target_col, "min"),
                    f"{target_col} 최고": (target_col, "max"),
                    "조사횟수": (target_col, "count"),
                }
            )
            .reset_index()
        )
        monthly_parts.append(growth_month)

    if not monthly_parts:
        return pd.DataFrame()

    result = monthly_parts[0]
    for part in monthly_parts[1:]:
        result = result.merge(
            part,
            on="월",
            how="outer",
        )

    return result.sort_values("월").reset_index(drop=True)


def observed_gei_response_scan(model, X, feature):
    """
    조사일에서 실제 관측된 GEI 고유값 각각에 대해,
    다른 GEI Feature는 각 조사일 값으로 유지한 채 선택 Feature만
    해당 관측값으로 치환하여 평균 예측반응을 계산합니다.

    이 곡선은 ALE 자체가 아니라 'Observed-grid model response'이며,
    ALE bin 중심점 수가 적을 때 조사일별 GEI 해상도로 임계 후보를
    보조 확인하기 위한 기능입니다.
    """
    if (
        model is None
        or X is None
        or not isinstance(X, pd.DataFrame)
        or X.empty
        or feature not in X.columns
    ):
        return pd.DataFrame()

    x = pd.to_numeric(
        X[feature],
        errors="coerce",
    )
    values = np.sort(
        x.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .unique()
    )
    if len(values) < 3:
        return pd.DataFrame()

    rows = []
    for value in values:
        temp = X.copy()
        temp[feature] = float(value)
        try:
            pred = np.asarray(
                model.predict(temp),
                dtype=float,
            ).reshape(-1)
            pred = pred[np.isfinite(pred)]
            mean_pred = float(np.mean(pred)) if pred.size else np.nan
        except Exception:
            mean_pred = np.nan
        rows.append(
            {
                "관측 GEI": float(value),
                "평균 예측반응": mean_pred,
            }
        )

    out = pd.DataFrame(rows).dropna()
    if out.empty:
        return out

    out["Centered 관측반응"] = (
        out["평균 예측반응"]
        - out["평균 예측반응"].mean()
    )

    if len(out) >= 3:
        slopes = np.gradient(
            out["Centered 관측반응"].to_numpy(dtype=float),
            out["관측 GEI"].to_numpy(dtype=float),
        )
        out["기울기"] = slopes
    else:
        out["기울기"] = np.nan

    return out


def detect_observed_gei_threshold(scan_df):
    """관측 GEI 상세 스캔에서 첫 음의 반응+음의 기울기 지점을 임계 후보로 반환."""
    if scan_df is None or scan_df.empty or len(scan_df) < 3:
        return None

    y = scan_df["Centered 관측반응"].to_numpy(dtype=float)
    slope = scan_df["기울기"].to_numpy(dtype=float)
    x = scan_df["관측 GEI"].to_numpy(dtype=float)

    idxs = np.where(
        (y < 0) & (slope < 0)
    )[0]
    idx = int(idxs[0]) if len(idxs) else int(np.argmin(y))

    return {
        "threshold": float(x[idx]),
        "response": float(y[idx]),
        "best": float(x[int(np.argmax(y))]),
        "worst": float(x[int(np.argmin(y))]),
    }


def _interp_curve_value(x_value, curve_df, x_col, y_col):
    """곡선의 x-y 점을 선형보간하여 관측 GEI 위치의 효과값을 반환합니다."""
    if curve_df is None or curve_df.empty or not np.isfinite(x_value):
        return np.nan
    temp = curve_df[[x_col, y_col]].copy()
    temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
    temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna().sort_values(x_col)
    temp = temp.drop_duplicates(x_col, keep="last")
    if temp.empty:
        return np.nan
    xs = temp[x_col].to_numpy(dtype=float)
    ys = temp[y_col].to_numpy(dtype=float)
    if len(xs) == 1:
        return float(ys[0])
    return float(np.interp(float(x_value), xs, ys, left=ys[0], right=ys[-1]))


def build_gei_date_threshold_mapping(
    gei_df,
    feature,
    target_col,
    ale_df=None,
    observed_scan=None,
    ale_threshold=None,
    observed_threshold=None,
):
    """
    GEI-domain 효과곡선을 조사일(time-domain)에 다시 매핑합니다.

    - 관측 GEI마다 ALE 중심곡선을 선형보간해 '보간 Centered ALE' 계산
    - observed-grid response도 동일 방식으로 보간
    - 기본 생육단계 기대 변화량 대비 NGR을 함께 계산
    - 임계 접근/초과 상태를 자동 분류
    """
    if (
        gei_df is None
        or gei_df.empty
        or "조사일자" not in gei_df.columns
        or feature not in gei_df.columns
        or target_col not in gei_df.columns
    ):
        return pd.DataFrame(), np.nan

    cols = ["조사일자", feature, target_col]
    delta_col = f"{target_col} 변화량"
    if delta_col in gei_df.columns:
        cols.append(delta_col)
    for extra in ["통합 GEI", "온도 GEI", "습도 GEI", "CO2 GEI", "CO₂ GEI", "일사량 GEI"]:
        if extra in gei_df.columns and extra not in cols:
            cols.append(extra)

    out = gei_df[cols].copy()
    out["조사일자"] = pd.to_datetime(out["조사일자"], errors="coerce")
    out[feature] = pd.to_numeric(out[feature], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    if delta_col not in out.columns:
        out[delta_col] = out[target_col].diff()
    else:
        out[delta_col] = pd.to_numeric(out[delta_col], errors="coerce")
    out = out.dropna(subset=["조사일자", feature]).sort_values("조사일자").reset_index(drop=True)
    if out.empty:
        return out, np.nan

    # v28.0의 최종 권장 NGR을 날짜 기준으로 결합합니다.
    try:
        ngr_df, _ = build_gei_growth_response_curve(
            gei_df=gei_df,
            gei_feature=feature,
            target_col=target_col,
            baseline_mode="생육단계 기대 변화량 대비(최종 권장)",
            stable_band_pct=2.0,
            danger_pct=-10.0,
        )
        if ngr_df is not None and not ngr_df.empty and "반응률(%)" in ngr_df.columns:
            ngr_map = ngr_df[["조사일자", "반응률(%)"]].copy()
            ngr_map["조사일자"] = pd.to_datetime(ngr_map["조사일자"], errors="coerce")
            ngr_map = ngr_map.rename(columns={"반응률(%)": "NGR(%)"}).drop_duplicates("조사일자", keep="last")
            out = out.merge(ngr_map, on="조사일자", how="left")
        else:
            out["NGR(%)"] = np.nan
    except Exception:
        out["NGR(%)"] = np.nan

    out["보간 Centered ALE"] = [
        _interp_curve_value(v, ale_df, "구간중심", "Centered ALE")
        for v in out[feature]
    ]
    out["관측-grid Centered 반응"] = [
        _interp_curve_value(v, observed_scan, "관측 GEI", "Centered 관측반응")
        for v in out[feature]
    ]

    threshold_candidates = []
    if observed_threshold and np.isfinite(observed_threshold.get("threshold", np.nan)):
        threshold_candidates.append(float(observed_threshold["threshold"]))
    if ale_threshold and np.isfinite(ale_threshold.get("threshold", np.nan)):
        threshold_candidates.append(float(ale_threshold["threshold"]))
    threshold_value = float(np.mean(threshold_candidates)) if threshold_candidates else np.nan

    if np.isfinite(threshold_value):
        approach_margin = max(1.0, abs(threshold_value) * 0.08)
        values = pd.to_numeric(out[feature], errors="coerce")
        out["임계거리"] = threshold_value - values
        out["상태"] = np.select(
            [
                values >= threshold_value,
                values >= (threshold_value - approach_margin),
            ],
            ["위험(임계초과)", "주의(임계접근)"],
            default="정상",
        )
    else:
        out["임계거리"] = np.nan
        out["상태"] = "임계값 미정"

    return out, threshold_value


def build_lagged_crop_response_tracking(mapping_df, feature, target_col, threshold_value, lags=(1, 2, 3)):
    """임계 접근/초과 이벤트 이후 1~3주 뒤 실제 생육반응을 날짜 기준으로 추적합니다."""
    if mapping_df is None or mapping_df.empty or not np.isfinite(threshold_value):
        return pd.DataFrame()

    work = mapping_df.copy().sort_values("조사일자").reset_index(drop=True)
    events = work[work[feature] >= threshold_value].copy()
    if events.empty:
        return pd.DataFrame()

    rows = []
    for _, event in events.iterrows():
        event_date = pd.to_datetime(event["조사일자"])
        for lag in lags:
            target_date = event_date + pd.Timedelta(days=int(lag) * 7)
            candidates = work[work["조사일자"] > event_date].copy()
            if candidates.empty:
                continue
            candidates["_gap"] = (candidates["조사일자"] - target_date).abs().dt.total_seconds() / 86400.0
            nearest = candidates.sort_values("_gap").iloc[0]
            # 주간 조사자료를 전제로 ±4일 이내만 같은 lag로 인정합니다.
            if float(nearest["_gap"]) > 4.0:
                continue
            rows.append({
                "임계 이벤트일": event_date,
                "이벤트 GEI": float(event[feature]),
                "이벤트 상태": event.get("상태", ""),
                "Lag(주)": int(lag),
                "반응 조사일": pd.to_datetime(nearest["조사일자"]),
                "실제 간격(일)": int((pd.to_datetime(nearest["조사일자"]) - event_date).days),
                f"{target_col} 변화량": float(nearest.get(f"{target_col} 변화량", np.nan)),
                "NGR(%)": float(nearest.get("NGR(%)", np.nan)),
                "반응시 GEI": float(nearest.get(feature, np.nan)),
            })
    return pd.DataFrame(rows)


def build_environment_context_for_survey_dates(
    sensor_df,
    date_col_sensor,
    value_col,
    survey_dates,
    window_days,
    display_mode,
):
    """조사일 기준 GEI 누적기간과 동일한 lookback에서 원환경 요약값을 계산합니다.

    display_mode
    - '일평균·최고·최저': 일별 mean/min/max를 만든 뒤 window 전체의 평균 일평균,
      최고 일최고, 최저 일최저를 반환합니다.
    - '주간평균·야간평균': 기존 DIF 정의와 동일하게 08~18시를 주간,
      19~07시를 야간으로 하여 window 전체 평균을 반환합니다.
    """
    if (
        sensor_df is None or not isinstance(sensor_df, pd.DataFrame) or sensor_df.empty
        or date_col_sensor not in sensor_df.columns or value_col not in sensor_df.columns
    ):
        return pd.DataFrame()

    s = sensor_df[[date_col_sensor, value_col]].copy()
    s[date_col_sensor] = pd.to_datetime(s[date_col_sensor], errors="coerce")
    s[value_col] = pd.to_numeric(s[value_col], errors="coerce")
    s = s.dropna(subset=[date_col_sensor, value_col]).sort_values(date_col_sensor)
    if s.empty:
        return pd.DataFrame()

    window_days = max(1, int(window_days or 7))
    rows = []
    for survey_date in pd.to_datetime(pd.Series(survey_dates), errors="coerce").dropna():
        # build_weekly_gei_dataset과 같은 의미로 조사일 직전 window_days일을 사용합니다.
        end_ts = pd.Timestamp(survey_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        start_ts = pd.Timestamp(survey_date) - pd.Timedelta(days=window_days - 1)
        w = s[(s[date_col_sensor] >= start_ts) & (s[date_col_sensor] <= end_ts)].copy()
        row = {"조사일자": pd.Timestamp(survey_date)}
        if w.empty:
            rows.append(row)
            continue

        if display_mode == "일평균·최고·최저":
            daily = (
                w.set_index(date_col_sensor)[value_col]
                .resample("D")
                .agg(["mean", "min", "max"])
                .dropna(how="all")
            )
            row["일평균"] = float(daily["mean"].mean()) if daily["mean"].notna().any() else np.nan
            row["최저"] = float(daily["min"].min()) if daily["min"].notna().any() else np.nan
            row["최고"] = float(daily["max"].max()) if daily["max"].notna().any() else np.nan
        elif display_mode == "주간평균·야간평균":
            hours = w[date_col_sensor].dt.hour
            day_mask = (hours >= 8) & (hours <= 18)
            night_mask = ~day_mask
            row["주간평균"] = float(w.loc[day_mask, value_col].mean()) if day_mask.any() else np.nan
            row["야간평균"] = float(w.loc[night_mask, value_col].mean()) if night_mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def get_environment_context_meta(feature, temp_col, hum_col, co2_col, solar_col):
    """선택 GEI와 원환경 센서 컬럼/단위/표시명을 연결합니다."""
    meta = {
        "온도 GEI": (temp_col, "온도", "°C"),
        "습도 GEI": (hum_col, "습도", "%"),
        "CO2 GEI": (co2_col, "CO₂", "ppm"),
        "CO₂ GEI": (co2_col, "CO₂", "ppm"),
        "일사량 GEI": (solar_col, "일사량", "W/m²"),
    }
    return meta.get(str(feature), (None, None, None))


def render_gei_time_domain_mapping(
    mapping_df,
    feature,
    target_col,
    threshold_value,
    key_prefix,
    sensor_df=None,
    date_col_sensor=None,
    temp_col=None,
    hum_col=None,
    co2_col=None,
    solar_col=None,
    window_days=7,
):
    """GEI 임계값을 날짜축에 재투영한 시계열, Event table, Lag tracking을 렌더링합니다."""
    if mapping_df is None or mapping_df.empty:
        st.info(f"{feature}: 조사일자 매핑에 사용할 자료가 없습니다.")
        return

    st.markdown(f"#### 🗓️ GEI 임계값 → 조사일자 시계열 매핑 · {feature}")

    # v29.0: Knowledge Base 저장용으로 현재 분석에서 탐지된 임계값을 세션에 보존합니다.
    if np.isfinite(pd.to_numeric(pd.Series([threshold_value]), errors="coerce").iloc[0]):
        _threshold_store = st.session_state.get("gei_thresholds", {})
        if not isinstance(_threshold_store, dict):
            _threshold_store = {}
        _threshold_store[str(feature)] = float(threshold_value)
        st.session_state["gei_thresholds"] = _threshold_store

    raw_col, raw_label, raw_unit = get_environment_context_meta(
        feature, temp_col, hum_col, co2_col, solar_col
    )
    context_options = ["일평균·최고·최저", "주간평균·야간평균", "없음"]
    if raw_col is None:
        context_mode = "없음"
        st.caption("통합 GEI는 여러 환경변수의 가중 결합값이므로 이 그래프에서는 단일 원환경 시계열을 겹치지 않습니다.")
    else:
        context_mode = st.radio(
            "원환경 시계열 표시",
            context_options,
            horizontal=True,
            index=0,
            key=f"{key_prefix}_raw_environment_mode",
            help=(
                "① 일평균·최고·최저: 선택한 GEI 누적기간에서 일별 평균/최저/최고를 요약합니다. "
                "② 주간평균·야간평균: 동일 누적기간에서 08~18시와 19~07시 평균을 비교합니다. "
                "③ 없음: GEI와 NGR만 표시합니다."
            ),
        )

    env_context_df = pd.DataFrame()
    if context_mode != "없음" and raw_col is not None:
        env_context_df = build_environment_context_for_survey_dates(
            sensor_df=sensor_df,
            date_col_sensor=date_col_sensor,
            value_col=raw_col,
            survey_dates=mapping_df["조사일자"],
            window_days=window_days,
            display_mode=context_mode,
        )

    if context_mode != "없음" and not env_context_df.empty:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
            row_heights=[0.38, 0.62],
        )
        gei_row = 2
        env_row = 1
        if context_mode == "일평균·최고·최저":
            env_specs = [("일평균", "solid"), ("최저", "dot"), ("최고", "dash")]
        else:
            env_specs = [("주간평균", "solid"), ("야간평균", "dash")]
        for col_name, dash_style in env_specs:
            if col_name not in env_context_df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=env_context_df["조사일자"],
                    y=env_context_df[col_name],
                    mode="lines+markers",
                    name=f"{raw_label} {col_name}",
                    line=dict(width=2.2, dash=dash_style),
                    marker=dict(size=6),
                    hovertemplate=(
                        "조사일=%{x|%Y-%m-%d}<br>" + raw_label + " " + col_name + "=%{y:.2f} " + raw_unit + "<extra></extra>"
                    ),
                ),
                row=env_row, col=1, secondary_y=False,
            )
        fig.update_yaxes(title_text=f"{raw_label} ({raw_unit})", row=env_row, col=1)
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        gei_row = 1
        env_row = None
    custom = np.column_stack([
        mapping_df["조사일자"].dt.strftime("%Y-%m-%d"),
        pd.to_numeric(mapping_df.get("보간 Centered ALE", np.nan), errors="coerce").fillna(np.nan),
        pd.to_numeric(mapping_df.get("관측-grid Centered 반응", np.nan), errors="coerce").fillna(np.nan),
        pd.to_numeric(mapping_df.get(f"{target_col} 변화량", np.nan), errors="coerce").fillna(np.nan),
        pd.to_numeric(mapping_df.get("NGR(%)", np.nan), errors="coerce").fillna(np.nan),
        mapping_df.get("상태", pd.Series("", index=mapping_df.index)).astype(str),
    ])
    fig.add_trace(
        go.Scatter(
            x=mapping_df["조사일자"],
            y=mapping_df[feature],
            mode="lines+markers",
            name=feature,
            line=dict(width=3, color="#0f766e"),
            marker=dict(size=8),
            customdata=custom,
            hovertemplate=(
                "조사일=%{customdata[0]}<br>"
                + feature + "=%{y:.3f}<br>"
                "보간 ALE=%{customdata[1]:.4f}<br>"
                "Observed-grid=%{customdata[2]:.4f}<br>"
                + target_col + " 변화량=%{customdata[3]:.3f}<br>"
                "NGR=%{customdata[4]:.2f}%<br>"
                "상태=%{customdata[5]}<extra></extra>"
            ),
        ),
        row=gei_row, col=1, secondary_y=False,
    )
    if "NGR(%)" in mapping_df.columns and pd.to_numeric(mapping_df["NGR(%)"], errors="coerce").notna().any():
        fig.add_trace(
            go.Scatter(
                x=mapping_df["조사일자"],
                y=mapping_df["NGR(%)"],
                mode="lines+markers",
                name="NGR(%)",
                line=dict(width=2.5, color="#2563eb", dash="dot"),
                marker=dict(size=7),
                hovertemplate="조사일=%{x|%Y-%m-%d}<br>NGR=%{y:.2f}%<extra></extra>",
            ),
            row=gei_row, col=1, secondary_y=True,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#64748b", row=gei_row, col=1, secondary_y=True)

    if np.isfinite(threshold_value):
        fig.add_hline(
            y=threshold_value,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text=f"임계 GEI {threshold_value:.2f}",
            annotation_position="top left",
            row=gei_row, col=1, secondary_y=False,
        )
        # 위험 관측일을 반투명 구간으로 표시
        dates = list(mapping_df["조사일자"])
        for i, row in mapping_df.iterrows():
            if float(row[feature]) < threshold_value:
                continue
            date = pd.to_datetime(row["조사일자"])
            if len(dates) > 1:
                prev_gap = abs((date - pd.to_datetime(dates[max(i - 1, 0)])).days) if i > 0 else 7
                next_gap = abs((pd.to_datetime(dates[min(i + 1, len(dates) - 1)]) - date).days) if i < len(dates)-1 else 7
                half = max(1, min(5, int(round((prev_gap + next_gap) / 4))))
            else:
                half = 3
            fig.add_vrect(
                x0=date - pd.Timedelta(days=half),
                x1=date + pd.Timedelta(days=half),
                fillcolor="rgba(220,38,38,0.10)",
                line_width=0,
                layer="below",
                row=gei_row, col=1,
            )

    fig.update_layout(
        height=620 if env_row is not None else 430,
        title=(
            f"조사일별 {raw_label} 원환경 · {feature} · 임계상태 · {target_col} NGR"
            if env_row is not None else
            f"조사일별 {feature} · 임계상태 · {target_col} NGR"
        ),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=60, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
        legend=dict(orientation="h", y=1.12, x=1.0, xanchor="right"),
    )
    fig.update_xaxes(title_text="조사일자", row=gei_row, col=1)
    fig.update_yaxes(title_text=feature, row=gei_row, col=1, secondary_y=False)
    fig.update_yaxes(title_text="NGR(%)", row=gei_row, col=1, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_time_series")
    if env_row is not None:
        st.caption(
            f"원환경 패널은 {feature} 산정에 사용한 조사일 직전 {int(window_days)}일과 동일한 기간을 요약합니다. "
            "따라서 원환경 변화 → 환경구간 누적 → GEI → 임계상태 → NGR의 연결을 같은 날짜축에서 해석할 수 있습니다."
        )

    event_df = mapping_df[mapping_df["상태"].isin(["주의(임계접근)", "위험(임계초과)"])].copy()
    st.markdown("**임계 접근·초과 Event table**")
    if event_df.empty:
        st.success("현재 관측기간에는 임계 접근 또는 초과 이벤트가 없습니다.")
    else:
        show_cols = ["조사일자", feature, "보간 Centered ALE", "관측-grid Centered 반응", f"{target_col} 변화량", "NGR(%)", "임계거리", "상태"]
        show_cols = [c for c in show_cols if c in event_df.columns]
        show = event_df[show_cols].copy()
        show["조사일자"] = pd.to_datetime(show["조사일자"]).dt.strftime("%Y-%m-%d")
        st.dataframe(show.round(4), use_container_width=True, hide_index=True, height=min(360, 38 * (len(show) + 1)))

    lag_df = build_lagged_crop_response_tracking(mapping_df, feature, target_col, threshold_value, lags=(1, 2, 3))
    st.markdown("**Lagged crop-response tracking · 임계 이벤트 이후 1~3주**")
    if lag_df.empty:
        st.info("임계초과 이벤트 또는 이후 1~3주 매칭 조사자료가 부족하여 Lag 추적표를 생성하지 못했습니다.")
    else:
        lag_show = lag_df.copy()
        lag_show["임계 이벤트일"] = pd.to_datetime(lag_show["임계 이벤트일"]).dt.strftime("%Y-%m-%d")
        lag_show["반응 조사일"] = pd.to_datetime(lag_show["반응 조사일"]).dt.strftime("%Y-%m-%d")
        st.dataframe(lag_show.round(4), use_container_width=True, hide_index=True, height=min(420, 38 * (len(lag_show) + 1)))
        lag_summary = lag_df.groupby("Lag(주)").agg(
            이벤트수=("임계 이벤트일", "count"),
            평균_NGR=("NGR(%)", "mean"),
            평균_변화량=(f"{target_col} 변화량", "mean"),
        ).reset_index()
        st.caption("임계초과 시점과 이후 생육반응을 날짜로 직접 연결한 기술적 추적 결과입니다. 인과효과가 아니라 시간적 연관성 확인용이며 Lag SHAP 결과와 함께 해석하는 것을 권장합니다.")
        st.dataframe(lag_summary.round(4), use_container_width=True, hide_index=True)

def render_gei_growth_module(
    sensor_df,
    yield_df,
    date_col_sensor,
    date_col_yield,
    temp_col,
    hum_col,
    co2_col,
    solar_col,
    growth_cols,
    harvest_count_col,
    fruit_set_col,
):
    render_stylish_section(
        "🌿 주간 환경구간 누적시간(GEI)과 생육·수확 증감 분석",
        "조사일 직전 환경노출을 구간별 누적시간으로 변환하고, GEI 상승에 따른 초장·엽장·엽폭·생장길이·화방높이·개화수·착과수·수확수 변화를 분석합니다.",
        kicker="GEI × GROWTH RESPONSE",
    )

    env_map = {"온도": temp_col, "습도": hum_col, "CO₂": co2_col, "일사량": solar_col}
    env_map = {k: v for k, v in env_map.items() if v is not None}
    if not env_map:
        st.warning("GEI 계산을 위해 온도·습도·CO₂·일사량 중 하나 이상의 센서 컬럼을 선택하세요.")
        return

    growth_map = {name: col for name, col in growth_cols.items() if col is not None}
    if fruit_set_col is not None:
        growth_map["착과수"] = fruit_set_col
    if harvest_count_col is not None:
        growth_map["수확수"] = harvest_count_col
    # 표준 생육항목은 우선순위대로 표시하고,
    # 수확수/착과수/개화수/평균과중1/평균과중2처럼 사용자가 실제 선택한
    # 원본 컬럼명은 누락시키지 않고 뒤에 그대로 유지합니다.
    preferred_order = [
        "초장",
        "엽장",
        "엽폭",
        "생장길이",
        "화방높이",
        "줄기굵기",
        "엽수",
    ]
    ordered_growth_map = {
        k: growth_map[k]
        for k in preferred_order
        if k in growth_map
    }
    for k, v in growth_map.items():
        if k not in ordered_growth_map:
            ordered_growth_map[k] = v
    growth_map = ordered_growth_map
    if not growth_map:
        st.warning("생육·수확 증감 분석을 위해 생육 또는 수확 컬럼을 하나 이상 선택하세요.")
        return

    zone_config = get_default_gei_zone_config()
    survey_min = pd.to_datetime(yield_df[date_col_yield], errors="coerce").min()
    survey_max = pd.to_datetime(yield_df[date_col_yield], errors="coerce").max()
    default_start = max(pd.Timestamp("2025-09-01"), survey_min) if pd.notna(survey_min) else pd.Timestamp("2025-09-01")
    default_end = min(pd.Timestamp("2026-06-30"), survey_max) if pd.notna(survey_max) else pd.Timestamp("2026-06-30")
    if default_start > default_end:
        default_start, default_end = survey_min, survey_max

    control0, control1, control2, control3 = st.columns([0.9, 1.15, 1.25, 1.25])
    with control0:
        gei_window_unit = st.radio(
            "환경 누적기간 단위",
            ["주 단위", "일 단위"],
            horizontal=True,
            key="gei_window_unit_v276",
        )
    with control1:
        if gei_window_unit == "주 단위":
            gei_window_value = st.selectbox(
                "환경 누적기간",
                options=list(range(1, 8)),
                index=0,
                format_func=lambda x: f"{x}주 ({x * 7}일)",
                key="gei_window_week_v276",
            )
            window_days = int(gei_window_value) * 7
        else:
            gei_window_value = st.selectbox(
                "환경 누적기간",
                options=[2, 3, 4, 5, 6],
                index=1,
                format_func=lambda x: f"{x}일",
                key="gei_window_day_v276",
            )
            window_days = int(gei_window_value)

        # 기존 코드/세션과의 호환을 위해 실제 계산 일수를 보존합니다.
        st.session_state["gei_window_days"] = int(window_days)
        st.caption(f"실제 GEI 계산기간: 조사일 직전 {int(window_days)}일")
    with control2:
        gei_start = st.date_input("GEI 분석 시작일", value=default_start.date(), key="gei_start_date")
    with control3:
        gei_end = st.date_input("GEI 분석 종료일", value=default_end.date(), key="gei_end_date")

    with st.expander("⚙️ 환경구간 및 GEI 위험가중치 기준", expanded=False):
        st.caption("환경구간 내부 위험가중치(구간별 0~4점)와 환경 간 통합가중치(온도·습도·CO₂·일사량 비중)는 서로 다른 개념입니다. 아래 표는 구간 내부 위험가중치이며, 환경 간 통합가중치는 데이터 기반 최적화 섹션에서 별도로 산출합니다.")
        st.dataframe(make_environment_zone_reference_table(zone_config), use_container_width=True, hide_index=True)
        st.markdown("**GEI 위험단계**")
        st.dataframe(GEI_STAGE_TABLE[["하한", "상한", "위험단계"]], use_container_width=True, hide_index=True)

    gei_df = build_weekly_gei_dataset(
        sensor_df=sensor_df,
        yield_df=yield_df,
        date_col_sensor=date_col_sensor,
        date_col_yield=date_col_yield,
        env_column_map=env_map,
        growth_column_map=growth_map,
        zone_config=zone_config,
        window_days=window_days,
        start_date=gei_start,
        end_date=gei_end,
    )
    if gei_df.empty:
        st.warning("선택한 기간에서 GEI 통합데이터를 만들 수 없습니다.")
        return

    st.session_state["gei_growth_dataset_equal_weight_baseline"] = gei_df.copy()

    # --------------------------------------------------------
    # v28.1 데이터 기반 통합 GEI 가중치 최적화
    # --------------------------------------------------------
    gei_df, selected_gei_weights, selected_gei_weight_method, gei_weight_comparison = render_gei_weight_optimizer(
        gei_df=gei_df,
        growth_targets=list(growth_map.keys()),
    )
    st.session_state["gei_growth_dataset"] = gei_df
    st.session_state["gei_selected_weights"] = selected_gei_weights
    st.session_state["gei_weight_method"] = selected_gei_weight_method
    st.session_state["gei_weight_comparison"] = gei_weight_comparison

    # GEI 자체의 1~7주 비교표·Heatmap·최고기간 자동추천
    gei_week_dfs = build_gei_window_datasets(
        sensor_df=sensor_df,
        yield_df=yield_df,
        date_col_sensor=date_col_sensor,
        date_col_yield=date_col_yield,
        env_column_map=env_map,
        growth_column_map=growth_map,
        zone_config=zone_config,
        start_date=gei_start,
        end_date=gei_end,
        weeks=tuple(range(1, 8)),
    )
    # 현재 선택한 데이터 기반 가중치를 1~7주 GEI 데이터셋에도 동일 적용하여
    # window 비교가 서로 다른 통합 GEI 정의 때문에 왜곡되지 않도록 합니다.
    if isinstance(gei_week_dfs, dict):
        for _week_key, _week_df in list(gei_week_dfs.items()):
            if isinstance(_week_df, pd.DataFrame) and not _week_df.empty:
                _method_map = {selected_gei_weight_method: selected_gei_weights}
                if "equal" not in _method_map:
                    _features = [f"{env} GEI" for env in GEI_ENV_ORDER if f"{env} GEI" in _week_df.columns]
                    _method_map["equal"] = _normalize_nonnegative_weights(np.ones(len(_features)), _features) if _features else {}
                gei_week_dfs[_week_key] = apply_gei_weight_methods(
                    _week_df,
                    methods=_method_map,
                    selected_method_key=selected_gei_weight_method,
                )

    st.session_state["gei_1to7_week_datasets"] = gei_week_dfs
    gei_window_performance, best_gei_week = render_gei_window_optimizer(
        gei_week_dfs=gei_week_dfs,
        growth_targets=list(growth_map.keys()),
        selected_window_days=window_days,
    )
    st.session_state["gei_1to7_performance"] = gei_window_performance
    st.session_state["best_gei_week"] = best_gei_week

    latest = gei_df.iloc[-1]
    gei_cols = [f"{env} GEI" for env in env_map if f"{env} GEI" in gei_df.columns]
    metric_cols = st.columns(len(gei_cols) + 1)
    for idx, col in enumerate(gei_cols):
        score = latest[col]
        stage, _ = gei_stage(score)
        with metric_cols[idx]:
            st.metric(col, f"{score:.1f}" if pd.notna(score) else "N/A", stage)
    with metric_cols[-1]:
        score = latest["통합 GEI"]
        stage, _ = gei_stage(score)
        st.metric("통합 GEI", f"{score:.1f}" if pd.notna(score) else "N/A", f"{stage} · {GEI_WEIGHT_METHOD_LABELS.get(selected_gei_weight_method, selected_gei_weight_method)}")

    st.markdown("### 1. 조사일별 환경구간 누적시간")
    selected_date = st.selectbox(
        "환경구간을 확인할 조사일",
        gei_df["조사일자"].tolist(),
        index=len(gei_df) - 1,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        key="gei_selected_survey_date",
    )
    selected_row = gei_df.loc[gei_df["조사일자"] == selected_date].iloc[0]
    env_names = list(env_map.keys())
    for start in range(0, len(env_names), 2):
        chart_cols = st.columns(2)
        for offset, env_name in enumerate(env_names[start:start + 2]):
            with chart_cols[offset]:
                st.plotly_chart(
                    render_zone_hours_chart(selected_row, env_name, zone_config),
                    use_container_width=True,
                    key=f"zone_hours_{env_name}_{start}",
                )

    # 일사량은 야간/무일사 L0를 GEI 분모에서 제외하여 유효광시간만 사용합니다.
    if "일사량" in env_map and "일사량 유효광시간(h)" in gei_df.columns:
        solar_c1, solar_c2, solar_c3, solar_c4 = st.columns(4)
        selected_expected = float(selected_row.get("기대누적시간(h)", np.nan))
        selected_light = float(selected_row.get("일사량 유효광시간(h)", np.nan))
        selected_dark = float(selected_row.get("일사량 야간·무일사 제외시간(h)", np.nan))
        selected_solar_gei = float(selected_row.get("일사량 GEI", np.nan))

        with solar_c1:
            st.metric(
                "선택 누적기간 전체시간",
                f"{selected_expected:.1f} h" if np.isfinite(selected_expected) else "N/A",
            )
        with solar_c2:
            st.metric(
                "일사량 유효광시간(GEI 분모)",
                f"{selected_light:.1f} h" if np.isfinite(selected_light) else "N/A",
            )
        with solar_c3:
            st.metric(
                "L0 야간·무일사 제외시간",
                f"{selected_dark:.1f} h" if np.isfinite(selected_dark) else "N/A",
            )
        with solar_c4:
            st.metric(
                "일사량 GEI",
                f"{selected_solar_gei:.1f}" if np.isfinite(selected_solar_gei) else "N/A",
            )

        st.markdown(
            """
            <div class="xai-insight-card">
                <b>☀️ 일사량 GEI 분모 처리</b><br>
                L0 일사없음(0~1 W/m² 미만)은 야간·무일사 시간으로 누적시간 그래프에는 그대로 표시하지만,
                일사량 GEI의 분자와 분모에서는 제외합니다.<br><br>
                <b>일사량 GEI = Σ(L1~L6 누적시간 × 위험가중치)
                ÷ [L1~L6 유효광시간 합계 × 최대위험가중치] × 100</b><br><br>
                따라서 환경 누적기간을 1주, 2주, …, 7주로 변경하면 각 조사일 직전
                7~49일 센서자료에서 L1~L6 시간이 다시 합산되어 <b>유효광시간 분모가 자동 변경</b>됩니다.
                온도·습도·CO₂의 GEI 계산방식은 기존과 동일하게 전체 유효 센서시간을 사용합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 2. 조사일 단위 GEI·생육·수확 통합데이터")
    base_display = (
        ["조사일자"]
        + gei_cols
        + [
            "통합 GEI",
            "통합 GEI 가중방식",
            "GEI 위험단계",
            "통합 데이터충족률(%)",
            "일사량 유효광시간(h)",
            "일사량 야간·무일사 제외시간(h)",
            "일사량 유효광시간비율(%)",
        ]
    )
    growth_display = []
    for target in growth_map:
        growth_display.extend([target, f"{target} 변화량", f"{target} 증감", f"GEI상승시 {target} 반응"])
    display_cols = [c for c in base_display + growth_display if c in gei_df.columns]
    st.dataframe(gei_df[display_cols].round(3), use_container_width=True, hide_index=True, height=430)
    st.download_button(
        "⬇️ GEI·생육 통합데이터 CSV 다운로드",
        data=gei_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"GEI_growth_{pd.Timestamp(gei_start):%Y%m%d}_{pd.Timestamp(gei_end):%Y%m%d}.csv",
        mime="text/csv",
        key="download_gei_growth_csv",
    )

    st.markdown("### 3. GEI 상승에 따른 생육량·수확량 증가/감소")
    selected_target = st.selectbox("분석할 생육·수확 항목", list(growth_map.keys()), key="gei_growth_target")
    time_fig = make_subplots(specs=[[{"secondary_y": True}]])
    time_fig.add_trace(
        go.Scatter(x=gei_df["조사일자"], y=gei_df["통합 GEI"], mode="lines+markers", name="통합 GEI", line=dict(width=3, color="#ef4444")),
        secondary_y=False,
    )
    time_fig.add_trace(
        go.Scatter(x=gei_df["조사일자"], y=gei_df[selected_target], mode="lines+markers+text", text=gei_df[f"{selected_target} 증감"], textposition="top center", name=selected_target, line=dict(width=3, color="#2563eb")),
        secondary_y=True,
    )
    time_fig.update_yaxes(title_text="GEI (0~100)", range=[0, 100], secondary_y=False)
    time_fig.update_yaxes(title_text=selected_target, secondary_y=True)
    time_fig.update_layout(height=470, hovermode="x unified", title=f"조사일별 통합 GEI와 {selected_target} 변화", margin=dict(l=45, r=45, t=65, b=45), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.82)")
    st.plotly_chart(time_fig, use_container_width=True, key="gei_growth_timeseries")


    # --------------------------------------------------------
    # 조사일별 정량 결과: 변화량 / 조사간격 / 일평균 변화량 / 월 / 상태
    # --------------------------------------------------------
    st.markdown("**조사일별 정량 결과**")
    survey_quant_df, survey_quant_meta = build_survey_growth_quantitative_table(
        gei_df=gei_df,
        target_col=selected_target,
    )

    if not survey_quant_df.empty:
        quant_cols = [
            c for c in [
                "조사일자",
                "월",
                "통합 GEI",
                selected_target,
                "변화량",
                "조사간격(일)",
                "일평균 변화량",
                "평균대비 변화상태",
            ]
            if c in survey_quant_df.columns
        ]
        quant_show = survey_quant_df[quant_cols].copy()
        quant_show["조사일자"] = pd.to_datetime(
            quant_show["조사일자"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric(
                "전체 평균 일변화량",
                (
                    f"{survey_quant_meta.get('평균 일변화량', np.nan):+.3f}"
                    if np.isfinite(
                        survey_quant_meta.get("평균 일변화량", np.nan)
                    )
                    else "N/A"
                ),
            )
        with q2:
            st.metric(
                "급격증가 기준",
                (
                    f"≥ {survey_quant_meta.get('급격증가 기준', np.nan):+.3f}"
                    if np.isfinite(
                        survey_quant_meta.get("급격증가 기준", np.nan)
                    )
                    else "N/A"
                ),
            )
        with q3:
            st.metric(
                "급격감소 기준",
                (
                    f"≤ {survey_quant_meta.get('급격감소 기준', np.nan):+.3f}"
                    if np.isfinite(
                        survey_quant_meta.get("급격감소 기준", np.nan)
                    )
                    else "N/A"
                ),
            )
        with q4:
            st.metric(
                "유효 조사구간",
                f"{survey_quant_meta.get('유효 조사구간수', 0)}개",
            )

        st.dataframe(
            quant_show.round(3),
            use_container_width=True,
            hide_index=True,
            height=430,
        )
        st.caption(
            "일평균 변화량 = 조사일 간 변화량 ÷ 조사간격(일). "
            "급격증가/급격감소는 전체 일평균 변화량의 평균 ± 1표준편차를 기준으로 자동 분류합니다."
        )

    # --------------------------------------------------------
    # 월별 환경 + 생육·수확 정량 요약
    # --------------------------------------------------------
    st.markdown("**월별 환경·생육(수확) 정량 요약**")
    monthly_summary = build_monthly_environment_growth_summary(
        sensor_df=sensor_df,
        gei_df=gei_df,
        date_col_sensor=date_col_sensor,
        env_map=env_map,
        target_col=selected_target,
    )
    if not monthly_summary.empty:
        st.dataframe(
            monthly_summary.round(3),
            use_container_width=True,
            hide_index=True,
            height=400,
        )
        st.caption(
            "환경은 각 월의 센서 원자료 기준 최저·평균·최고를 표시합니다. 일사량 최저는 야간 0을 제외한 1 이상 유효값만 사용하며, "
            f"{selected_target}은 월별 증가량 합계·감소량 합계·평균·최저·최고를 표시합니다."
        )
    else:
        st.info("월별 환경·생육 요약을 생성할 유효 데이터가 부족합니다.")

    rising = gei_df[gei_df["GEI 변화량"] > 0].copy()
    response_rows = []
    for target in growth_map:
        delta_col = f"{target} 변화량"
        valid = rising.dropna(subset=[delta_col])
        response_rows.append(
            {
                "생육·수확 항목": target,
                "GEI 상승 조사구간수": len(valid),
                "증가 횟수": int((valid[delta_col] > 0).sum()),
                "감소 횟수": int((valid[delta_col] < 0).sum()),
                "유지 횟수": int((valid[delta_col] == 0).sum()),
                "GEI 상승 시 평균 변화량": float(valid[delta_col].mean()) if len(valid) else np.nan,
                "감소 비율(%)": float((valid[delta_col] < 0).mean() * 100) if len(valid) else np.nan,
            }
        )
    response_df = pd.DataFrame(response_rows)
    st.dataframe(response_df.round(3), use_container_width=True, hide_index=True)

    selected_valid = rising.dropna(subset=[f"{selected_target} 변화량"])
    if not selected_valid.empty:
        decrease_ratio = (selected_valid[f"{selected_target} 변화량"] < 0).mean() * 100
        mean_change = selected_valid[f"{selected_target} 변화량"].mean()
        relation_text = "감소 경향" if mean_change < 0 else "증가 경향" if mean_change > 0 else "뚜렷한 변화 없음"
        st.markdown(
            f'<div class="xai-insight-card"><b>{selected_target} 해석</b><br>통합 GEI가 상승한 {len(selected_valid)}개 조사구간 중 {decrease_ratio:.1f}%에서 {selected_target}이 감소했습니다. GEI 상승 시 평균 변화량은 {mean_change:+.3f}으로, 현재 데이터에서는 <b>{relation_text}</b>이 관찰됩니다. 이는 관찰적 관계이며 생육단계·계절 추세·관리작업을 함께 검토해야 합니다.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 4. GEI 기반 Centered ALE 임계구간 분석")
    gei_ale_thresholds = {}
    ale_target_mode = st.radio("ALE 예측대상 형태", ["주간 변화량", "조사값 원자료"], horizontal=True, key="gei_ale_target_mode")
    y_col = f"{selected_target} 변화량" if ale_target_mode == "주간 변화량" else selected_target
    available_gei = gei_cols + ["통합 GEI"]
    ale_features = st.multiselect(
        "ALE로 분석할 GEI",
        available_gei,
        default=available_gei,
        key="gei_ale_features",
    )
    model_cols = list(dict.fromkeys(available_gei))
    model_df = (
        gei_df[model_cols + [y_col]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    max_ale_bins = max(
        5,
        min(
            30,
            max(5, len(model_df) - 1),
        ),
    )
    default_ale_bins = min(
        max_ale_bins,
        max(5, min(20, len(model_df) - 1)),
    )
    gei_ale_bins = st.slider(
        "GEI ALE bins 수",
        min_value=5,
        max_value=max_ale_bins,
        value=default_ale_bins,
        step=1,
        key="gei_ale_bins_detail_v275",
        help=(
            "조사일이 25개라면 최대 24개 구간까지 요청할 수 있습니다. "
            "단, 동일 GEI 값이나 중복 분위수 경계가 있으면 실제 ALE 중심점 수는 더 적어질 수 있습니다."
        ),
    )

    if len(model_df) < 10 or len(ale_features) == 0:
        st.warning("Centered ALE 계산에는 GEI와 예측대상이 모두 존재하는 조사일이 최소 10개 필요합니다.")
    else:
        X_ale = model_df[model_cols]
        y_ale = model_df[y_col]
        ale_model = RandomForestRegressor(n_estimators=500, min_samples_leaf=max(1, len(model_df) // 20), random_state=42)
        ale_model.fit(X_ale, y_ale)
        train_r2 = r2_score(y_ale, ale_model.predict(X_ale)) if y_ale.nunique() > 1 else np.nan
        st.caption(f"ALE 모델: RandomForest · 분석자료 {len(model_df)}개 · 참고용 학습 R²={train_r2:.3f}. 소표본에서는 교차검증 결과와 함께 해석해야 합니다.")
        for start in range(0, len(ale_features), 2):
            cols = st.columns(2)
            for offset, feature in enumerate(ale_features[start:start + 2]):
                with cols[offset]:
                    ale_df = centered_ale_1d(ale_model, X_ale, feature, bins=gei_ale_bins)
                    if ale_df.empty:
                        st.info(f"{feature}: 고유값이 부족하여 ALE를 계산할 수 없습니다.")
                        continue
                    threshold = detect_ale_threshold(ale_df)
                    if threshold:
                        gei_ale_thresholds[feature] = threshold
                    fig_ale = go.Figure()
                    for low, high, stage, color in GEI_STAGE_TABLE.itertuples(index=False, name=None):
                        fig_ale.add_vrect(x0=low, x1=min(high, 100), fillcolor=color, opacity=0.28, line_width=0)
                    fig_ale.add_trace(go.Scatter(x=ale_df["구간중심"], y=ale_df["Centered ALE"], mode="lines+markers", line=dict(width=4, color="#1d4ed8"), marker=dict(size=8), name="Centered ALE"))
                    fig_ale.add_hline(y=0, line_dash="dash", line_color="#475569")
                    if threshold:
                        fig_ale.add_vline(x=threshold["threshold"], line_dash="dot", line_color="#dc2626", annotation_text=f"감소 후보 {threshold['threshold']:.1f}", annotation_position="top right")
                    fig_ale.update_layout(title=f"{feature} → {y_col}", height=390, xaxis_title=feature, yaxis_title="Centered ALE", margin=dict(l=50, r=25, t=65, b=45), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.82)")
                    st.plotly_chart(fig_ale, use_container_width=True, key=f"gei_ale_{feature}_{selected_target}_{ale_target_mode}")

                    st.caption(
                        f"ALE 요청 bins={gei_ale_bins} · 실제 ALE 중심점={len(ale_df)}개 · "
                        f"분석 조사일={len(model_df)}개 · {feature} 고유값={model_df[feature].nunique()}개"
                    )

                    # 조사일별 실제 GEI 고유값을 모두 이용한 상세 임계 보조 스캔
                    observed_scan = observed_gei_response_scan(
                        ale_model,
                        X_ale,
                        feature,
                    )
                    observed_threshold = detect_observed_gei_threshold(
                        observed_scan
                    )

                    if not observed_scan.empty:
                        st.markdown(
                            f"**조사일 관측 GEI 상세 임계 스캔 · {feature}**"
                        )
                        obs_fig = go.Figure()
                        obs_fig.add_trace(
                            go.Scatter(
                                x=observed_scan["관측 GEI"],
                                y=observed_scan["Centered 관측반응"],
                                mode="lines+markers",
                                line=dict(width=3, color="#0f766e"),
                                marker=dict(size=7),
                                name="Observed-grid response",
                            )
                        )
                        obs_fig.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color="#64748b",
                        )
                        if observed_threshold:
                            obs_fig.add_vline(
                                x=observed_threshold["threshold"],
                                line_dash="dot",
                                line_color="#dc2626",
                                annotation_text=(
                                    "상세 감소 후보 "
                                    f"{observed_threshold['threshold']:.2f}"
                                ),
                                annotation_position="top right",
                            )
                        obs_fig.update_layout(
                            height=320,
                            xaxis_title=feature,
                            yaxis_title="Centered 관측 GEI 예측반응",
                            margin=dict(l=48, r=20, t=35, b=45),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.82)",
                        )
                        st.plotly_chart(
                            obs_fig,
                            use_container_width=True,
                            key=f"gei_observed_grid_{feature}_{selected_target}_{ale_target_mode}",
                        )
                        st.dataframe(
                            observed_scan.round(4),
                            use_container_width=True,
                            hide_index=True,
                            height=min(340, 38 * (len(observed_scan) + 1)),
                        )
                        st.caption(
                            "위 상세 스캔은 ALE 자체가 아니라, 조사일에서 실제 관측된 GEI 고유값을 "
                            "모델 입력값으로 순차 대입한 보조 반응곡선입니다. "
                            "따라서 25개 조사일에 25개 서로 다른 GEI가 있으면 최대 25개 관측 GEI 점을 표시할 수 있습니다."
                        )

                    # v28.2: GEI-domain 임계값을 실제 조사일(time-domain)에 재투영
                    date_mapping_df, mapped_threshold = build_gei_date_threshold_mapping(
                        gei_df=gei_df,
                        feature=feature,
                        target_col=selected_target,
                        ale_df=ale_df,
                        observed_scan=observed_scan,
                        ale_threshold=threshold,
                        observed_threshold=observed_threshold,
                    )
                    render_gei_time_domain_mapping(
                        mapping_df=date_mapping_df,
                        feature=feature,
                        target_col=selected_target,
                        threshold_value=mapped_threshold,
                        key_prefix=f"gei_time_map_{feature}_{selected_target}_{ale_target_mode}",
                        sensor_df=sensor_df,
                        date_col_sensor=date_col_sensor,
                        temp_col=temp_col,
                        hum_col=hum_col,
                        co2_col=co2_col,
                        solar_col=solar_col,
                        window_days=window_days,
                    )

                    if threshold:
                        st.markdown(
                            f'<div class="xai-insight-card"><b>{feature}</b>의 ALE가 가장 우호적인 중심은 약 <b>{threshold["best"]:.1f}</b>, 가장 불리한 중심은 약 <b>{threshold["worst"]:.1f}</b>입니다. 약 <b>{threshold["threshold"]:.1f}</b> 부근부터 음의 ALE와 하락 기울기가 함께 나타나 {selected_target} {"변화량" if ale_target_mode == "주간 변화량" else "예측값"} 감소 후보 임계점으로 해석할 수 있습니다.</div>',
                            unsafe_allow_html=True,
                        )

    # =========================================================
    # 5. GEI 기반 생육 반응 곡선
    # =========================================================
    st.markdown(
        "### 5. GEI 기반 생육 반응 곡선 · 기대 생육 변화량 대비 증가/유지/감소"
    )
    st.markdown(
        """
        <div class="xai-insight-card">
            <b>왜 이 그래프를 추가하는가?</b><br>
            Centered ALE는 조사일 자료를 여러 bin으로 나누기 때문에
            25개 조사일이 있어도 곡선 중심점은 보통 5~10개 정도만 보입니다.
            아래 반응곡선은 <b>각 조사일의 GEI와 실제 생육·수확 관측값을 모두 점으로 표시</b>하여,
            GEI가 변할 때 관측값이 기준 생육보다 커졌는지 또는 작아졌는지를 직접 확인합니다.<br><br>
            <b>중요:</b> 이 그래프는 관찰적 기술통계이고 Centered ALE는 모델 기반 조건부 효과입니다.
            따라서 두 결과가 같은 임계영역을 가리킬 때 해석의 신뢰성이 더 높아집니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    response_c1, response_c2, response_c3, response_c4 = st.columns(
        [1.05, 1.55, 1.0, 1.0]
    )

    with response_c1:
        gei_response_feature = st.selectbox(
            "반응곡선 GEI",
            available_gei,
            index=(
                available_gei.index(
                    "통합 GEI"
                )
                if "통합 GEI"
                in available_gei
                else 0
            ),
            key="gei_growth_response_feature",
        )

    with response_c2:
        gei_response_baseline = st.selectbox(
            "생육 반응률 기준",
            [
                "생육단계 기대 변화량 대비(최종 권장)",
                "월평균 변화량 대비(신규 권장)",
                "전체 조사 평균 대비",
                "생육추세 기대값 대비(기존)",
                "직전 조사 대비",
                "수동 기준값 대비",
            ],
            index=0,
            key="gei_growth_response_baseline",
            help=(
                "최종 권장: 조사 간 변화량을 7일 기준으로 환산한 뒤 생육단계별 기대 변화량과 비교합니다. "
                "월평균 변화량 대비는 같은 달의 정상 성장속도와 비교합니다. "
                "두 신규 방식은 초장처럼 시간이 지나며 누적 증가하는 지표의 시간·생육단계 효과를 줄이는 데 유리합니다."
            ),
        )

    with response_c3:
        stable_band_pct = st.number_input(
            "유지 허용범위 ±(%)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="gei_growth_response_stable_band",
        )

    with response_c4:
        danger_abs_pct = st.number_input(
            "위험 감소기준 (%)",
            min_value=5.0,
            max_value=30.0,
            value=10.0,
            step=1.0,
            key="gei_growth_response_danger",
            help=(
                "예: 10이면 기준값 대비 -10% 이하를 위험으로 분류합니다."
            ),
        )

    manual_baseline_value = None
    if gei_response_baseline == "수동 기준값 대비":
        default_manual_baseline = float(
            pd.to_numeric(
                gei_df[selected_target],
                errors="coerce",
            ).mean()
        )
        manual_baseline_value = st.number_input(
            f"{selected_target} 수동 기준값 직접 입력",
            value=(
                default_manual_baseline
                if np.isfinite(default_manual_baseline)
                else 0.0
            ),
            step=0.1,
            key="gei_growth_manual_baseline_v275",
            help=(
                "모든 조사일의 반응률을 이 기준값과 비교합니다. "
                "예: 표준 초장, 목표 생체중, 목표 수확수 등 연구자 기준값을 직접 입력할 수 있습니다."
            ),
        )

    response_curve_df, response_meta = (
        build_gei_growth_response_curve(
            gei_df=gei_df,
            gei_feature=gei_response_feature,
            target_col=selected_target,
            baseline_mode=gei_response_baseline,
            stable_band_pct=float(
                stable_band_pct
            ),
            danger_pct=-float(
                danger_abs_pct
            ),
            manual_baseline=manual_baseline_value,
        )
    )

    if response_curve_df.empty:
        st.warning(
            "GEI 기반 생육 반응 곡선을 만들 유효 조사일이 부족합니다."
        )
    else:
        response_value_label = response_meta.get(
            "response_value_label",
            f"{selected_target} 관측값",
        )
        baseline_label_ui = response_meta.get(
            "baseline_label",
            "기준값",
        )
        response_left, response_right = st.columns(
            [1.25, 0.75],
            gap="large",
        )

        with response_left:
            render_panel_label(
                f"{gei_response_feature} → {selected_target} 정규화 반응률(%) · {response_value_label}"
            )

            response_fig = go.Figure()

            # ------------------------------------------------
            # Y축 반응영역 배경:
            # 증가(녹색) / 유지(노랑) / 감소(연한 빨강) / 위험(빨강)
            # ------------------------------------------------
            y_min_data = float(
                np.nanmin(
                    response_curve_df[
                        "반응률(%)"
                    ].to_numpy(dtype=float)
                )
            )
            y_max_data = float(
                np.nanmax(
                    response_curve_df[
                        "반응률(%)"
                    ].to_numpy(dtype=float)
                )
            )
            stable = float(
                stable_band_pct
            )
            danger = -float(
                danger_abs_pct
            )

            y_span = max(
                10.0,
                y_max_data - y_min_data,
            )
            plot_y_min = min(
                danger - 0.20 * y_span,
                y_min_data - 0.12 * y_span,
            )
            plot_y_max = max(
                stable + 0.25 * y_span,
                y_max_data + 0.12 * y_span,
            )

            response_fig.add_hrect(
                y0=stable,
                y1=plot_y_max,
                fillcolor="rgba(34,197,94,0.12)",
                line_width=0,
                annotation_text="↑ 증가영역",
                annotation_position="top left",
            )
            response_fig.add_hrect(
                y0=-stable,
                y1=stable,
                fillcolor="rgba(250,204,21,0.16)",
                line_width=0,
                annotation_text="→ 유지영역",
                annotation_position="top left",
            )
            response_fig.add_hrect(
                y0=danger,
                y1=-stable,
                fillcolor="rgba(248,113,113,0.12)",
                line_width=0,
                annotation_text="↓ 감소영역",
                annotation_position="top left",
            )
            response_fig.add_hrect(
                y0=plot_y_min,
                y1=danger,
                fillcolor="rgba(220,38,38,0.15)",
                line_width=0,
                annotation_text="⚠ 위험영역",
                annotation_position="bottom left",
            )

            state_color_map = {
                "↑ 증가": "#16a34a",
                "→ 유지": "#eab308",
                "↓ 감소": "#f97316",
                "⚠ 위험": "#dc2626",
            }

            # 모든 조사일을 상태별 point로 표시
            for state_name in [
                "↑ 증가",
                "→ 유지",
                "↓ 감소",
                "⚠ 위험",
            ]:
                state_df = response_curve_df[
                    response_curve_df[
                        "생육반응 상태"
                    ]
                    == state_name
                ]
                if state_df.empty:
                    continue

                response_fig.add_trace(
                    go.Scatter(
                        x=state_df[
                            gei_response_feature
                        ],
                        y=state_df[
                            "반응률(%)"
                        ],
                        mode="markers",
                        marker=dict(
                            size=11,
                            color=state_color_map[
                                state_name
                            ],
                            line=dict(
                                width=1.5,
                                color="#ffffff",
                            ),
                        ),
                        name=state_name,
                        customdata=np.column_stack(
                            [
                                state_df["조사일자"].dt.strftime("%Y-%m-%d"),
                                state_df[selected_target],
                                state_df["변화량"],
                                state_df["7일환산 변화량"],
                                state_df["반응대상값"],
                                state_df["기준값"],
                            ]
                        ),
                        hovertemplate=(
                            "조사일=%{customdata[0]}<br>"
                            + gei_response_feature + "=%{x:.3f}<br>"
                            + selected_target + "=%{customdata[1]:.3f}<br>"
                            + "실제 변화량=%{customdata[2]:+.3f}<br>"
                            + "7일환산 변화량=%{customdata[3]:+.3f}<br>"
                            + response_value_label + "=%{customdata[4]:+.3f}<br>"
                            + baseline_label_ui + "=%{customdata[5]:+.3f}<br>"
                            + "정규화 반응률=%{y:+.2f}%"
                            + "<extra></extra>"
                        ),
                    )
                )

            # GEI 정렬 이동평균 추세선
            trend_df = (
                response_curve_df[
                    [
                        gei_response_feature,
                        "GEI 반응 추세(%)",
                    ]
                ]
                .dropna()
                .sort_values(
                    gei_response_feature
                )
            )
            if not trend_df.empty:
                response_fig.add_trace(
                    go.Scatter(
                        x=trend_df[
                            gei_response_feature
                        ],
                        y=trend_df[
                            "GEI 반응 추세(%)"
                        ],
                        mode="lines",
                        line=dict(
                            width=4,
                            color="#1d4ed8",
                            shape="spline",
                        ),
                        name=(
                            "GEI 정렬 반응 추세"
                        ),
                        hovertemplate=(
                            gei_response_feature
                            + "=%{x:.3f}<br>"
                            "추세 반응=%{y:+.2f}%"
                            "<extra></extra>"
                        ),
                    )
                )

            response_fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="#475569",
                annotation_text="기대 생육 변화량 = 0% 반응",
                annotation_position="bottom right",
            )

            # 반응곡선에서 자동 탐지한 감소 시작 후보
            response_decrease_threshold = (
                response_meta.get(
                    "decrease_threshold",
                    np.nan,
                )
            )
            if np.isfinite(
                response_decrease_threshold
            ):
                response_fig.add_vline(
                    x=float(
                        response_decrease_threshold
                    ),
                    line_dash="dot",
                    line_width=3,
                    line_color="#f97316",
                    annotation_text=(
                        "반응곡선 감소 후보 "
                        f"{response_decrease_threshold:.1f}"
                    ),
                    annotation_position="top right",
                )

            # 위험 후보
            response_danger_threshold = (
                response_meta.get(
                    "danger_threshold",
                    np.nan,
                )
            )
            if np.isfinite(
                response_danger_threshold
            ):
                response_fig.add_vline(
                    x=float(
                        response_danger_threshold
                    ),
                    line_dash="dash",
                    line_width=3,
                    line_color="#b91c1c",
                    annotation_text=(
                        "위험 후보 "
                        f"{response_danger_threshold:.1f}"
                    ),
                    annotation_position="bottom right",
                )

            # 동일 GEI의 ALE 임계점을 함께 표시
            ale_threshold_value = np.nan
            if (
                gei_response_feature
                in gei_ale_thresholds
            ):
                ale_threshold_value = float(
                    gei_ale_thresholds[
                        gei_response_feature
                    ]["threshold"]
                )
                response_fig.add_vline(
                    x=ale_threshold_value,
                    line_dash="dot",
                    line_width=2,
                    line_color="#7c3aed",
                    annotation_text=(
                        "ALE 임계 "
                        f"{ale_threshold_value:.1f}"
                    ),
                    annotation_position="top left",
                )

            response_fig.update_layout(
                title=(
                    f"{gei_response_feature}에 따른 "
                    f"{selected_target} 생육 반응곡선"
                ),
                xaxis_title=gei_response_feature,
                yaxis_title=(
                    f"{selected_target} 정규화 반응률 (%)"
                ),
                yaxis=dict(
                    range=[
                        plot_y_min,
                        plot_y_max,
                    ],
                    zeroline=False,
                ),
                height=540,
                hovermode="closest",
                template=plotly_template,
                margin=dict(
                    l=60,
                    r=40,
                    t=80,
                    b=55,
                ),
                legend=dict(
                    orientation="h",
                    y=1.13,
                    x=1,
                    xanchor="right",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=(
                    "rgba(255,255,255,0.85)"
                ),
            )
            st.plotly_chart(
                response_fig,
                use_container_width=True,
                key=(
                    "gei_growth_response_curve_"
                    f"{gei_response_feature}_"
                    f"{selected_target}_"
                    f"{gei_response_baseline}"
                ),
            )

        with response_right:
            render_panel_label(
                "조사일별 정량 결과"
            )

            response_display = (
                response_curve_df[
                    [
                        "조사일자",
                        gei_response_feature,
                        selected_target,
                        "변화량",
                        "조사간격(일)",
                        "일평균 변화량",
                        "7일환산 변화량",
                        "조사월",
                        "경과주",
                        "반응대상값",
                        "기준값",
                        "반응률(%)",
                        "GEI 반응 추세(%)",
                        "생육반응 상태",
                    ]
                ]
                .sort_values(
                    "조사일자"
                )
                .copy()
            )
            response_display[
                "조사일자"
            ] = response_display[
                "조사일자"
            ].dt.strftime(
                "%Y-%m-%d"
            )

            st.dataframe(
                response_display.round(3),
                use_container_width=True,
                hide_index=True,
                height=540,
            )

        # ----------------------------------------------------
        # 핵심 Metric
        # ----------------------------------------------------
        gr_m1, gr_m2, gr_m3, gr_m4 = st.columns(
            4
        )

        with gr_m1:
            st.metric(
                "반응 분석 조사일",
                f"{len(response_curve_df)}개",
            )

        with gr_m2:
            best_gei_value = response_meta.get(
                "best_gei",
                np.nan,
            )
            st.metric(
                "우호 GEI 중심 후보",
                (
                    f"{best_gei_value:.2f}"
                    if np.isfinite(
                        best_gei_value
                    )
                    else "N/A"
                ),
            )

        with gr_m3:
            dec_thr = response_meta.get(
                "decrease_threshold",
                np.nan,
            )
            st.metric(
                "감소 시작 후보",
                (
                    f"{dec_thr:.2f}"
                    if np.isfinite(dec_thr)
                    else "탐지 불가"
                ),
            )

        with gr_m4:
            risk_thr = response_meta.get(
                "danger_threshold",
                np.nan,
            )
            st.metric(
                "위험 GEI 후보",
                (
                    f"{risk_thr:.2f}"
                    if np.isfinite(risk_thr)
                    else "탐지 불가"
                ),
            )

        # ----------------------------------------------------
        # GEI 구간별 정량 요약
        # ----------------------------------------------------
        interval_summary = response_meta.get(
            "interval_summary",
            pd.DataFrame(),
        )

        if (
            interval_summary is not None
            and not interval_summary.empty
        ):
            st.markdown(
                "**GEI 구간별 생육 반응 요약**"
            )
            st.dataframe(
                interval_summary.round(3),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # 자동 해석
        # ----------------------------------------------------
        ale_thr_for_text = (
            gei_ale_thresholds.get(
                gei_response_feature,
                {},
            ).get(
                "threshold",
                np.nan,
            )
        )

        st.markdown(
            "**GEI 기반 생육 반응 자동 해석**"
        )
        st.markdown(
            (
                '<div class="xai-insight-card">'
                + explain_gei_growth_response_curve(
                    response_df=response_curve_df,
                    metadata=response_meta,
                    gei_feature=gei_response_feature,
                    target_col=selected_target,
                    ale_threshold=(
                        ale_thr_for_text
                        if np.isfinite(
                            ale_thr_for_text
                        )
                        else None
                    ),
                )
                + "</div>"
            ),
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # 방법론 설명
        # ----------------------------------------------------
        st.caption(
            "권장 해석: 누적형 생육지표(초장·생장길이 등)는 절대값보다 조사 간 변화량을 사용해야 시간경과 효과를 줄일 수 있습니다. "
            "'생육단계 기대 변화량 대비'는 조사간격을 7일 기준으로 표준화한 성장량을 생육단계별 기대 성장량과 비교하고, "
            "'월평균 변화량 대비'는 동일 월의 평균 성장속도와 비교합니다. "
            "0%는 기대 성장량과 동일, 음수는 실제 초장이 줄었다는 뜻이 아니라 기대 성장속도보다 낮았다는 뜻입니다. "
            "최종 임계값은 반응곡선 후보를 Centered ALE 및 Bootstrap CI와 교차확인해 제시하는 것을 권장합니다."
        )

    st.markdown("### 6. GEI 처리과정 요약")
    st.markdown(
        """
        <div class="xai-card">
        <b>Sensor CSV</b> → 조사일 직전까지만 1~7주(7~49일) 각각 계산 → 온도·습도·CO₂·일사량 환경구간 분류 → 구간별 누적시간 계산 → <b>일사량은 L0 야간·무일사 시간을 별도 누적하되 GEI 분모에서 제외하고 L1~L6 유효광시간만 분모로 사용</b> → 위험가중치 기반 개별 환경 GEI(0~100) 계산 → <b>동일가중(25% 기준선)·SHAP·표준화회귀·상관·제약최적화·합의가중 비교</b> → 선택 데이터 기반 가중치로 통합 GEI 재계산 → 같은 가중치를 1~7주 window에 일관 적용 → 생육·수확 조사값과 조사일 기준 결합 → 주간 변화량과 증가·감소 화살표 생성 → GEI 상승 시 반응표 작성 → RandomForest와 Centered ALE로 영향방향 및 감소 후보 임계점 분석 → <b>조사일별 GEI × 월평균/생육단계 기대 변화량 대비 정규화 생육 반응률(%) 곡선</b> 생성 → 증가·유지·감소·위험 영역 자동 분류 → 반응곡선 감소 후보와 ALE 임계점 비교
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # 7. GEI 이후 환경패턴 클러스터링 기반 생육·수확 반응
    # =========================================================
    render_environment_clustering_after_gei(
        sensor_df=sensor_df,
        yield_df=yield_df,
        date_col_sensor=date_col_sensor,
        date_col_yield=date_col_yield,
        env_map=env_map,
        growth_map=growth_map,
        window_days=window_days,
    )


# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
crop_name = st.selectbox("작물 선택", ["토마토", "오이"])
sensor_file = st.file_uploader("환경센서 데이터 업로드 (CSV)", type=["csv"])
yield_file = st.file_uploader("수확/생육 데이터 업로드 (CSV)", type=["csv"])

if sensor_file and yield_file:
    sensor_df = pd.read_csv(sensor_file)
    yield_df = pd.read_csv(yield_file)

    st.subheader("환경센서 데이터")
    st.dataframe(sensor_df.head())
    st.subheader("수확/생육 데이터")
    st.dataframe(yield_df.head())

    st.subheader("컬럼 선택")
    st.markdown("**환경 센서 데이터 컬럼 선택**")
    st.caption(
        "각 선택 목록에서 None을 선택하면 해당 변수는 매핑데이터·환경그래프·"
        "모델 학습·XAI 분석에서 제외됩니다. 날짜시간은 데이터 통합을 위해 필수입니다."
    )

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    with c1:
        date_col_sensor = select_optional_column(
            "날짜시간",
            sensor_df.columns,
            key="sensor_datetime_col",
            preferred_names=(
                "날짜시간", "datetime", "date_time", "timestamp",
                "측정일시", "수집일시", "시간",
            ),
            help_text="환경센서 측정 시각입니다. 데이터 통합을 위해 반드시 선택해야 합니다.",
        )
    with c2:
        temp_col = select_optional_column(
            "온도",
            sensor_df.columns,
            key="sensor_temperature_col",
            preferred_names=("온도", "temperature", "temp", "기온"),
        )
    with c3:
        hum_col = select_optional_column(
            "습도",
            sensor_df.columns,
            key="sensor_humidity_col",
            preferred_names=("습도", "humidity", "rh", "상대습도"),
        )
    with c4:
        co2_col = select_optional_column(
            "CO₂",
            sensor_df.columns,
            key="sensor_co2_col",
            preferred_names=("CO₂", "CO2", "co2", "이산화탄소"),
        )
    with c5:
        solar_col = select_optional_column(
            "일사량",
            sensor_df.columns,
            key="sensor_solar_col",
            preferred_names=(
                "일사량", "누적일사량", "solar", "solar_radiation",
                "radiation", "광량",
            ),
        )
    with c6:
        moisture_deficit_col = select_optional_column(
            "수분부족분",
            sensor_df.columns,
            key="sensor_moisture_deficit_col",
            preferred_names=(
                "수분부족분", "HD", "hd", "humidity_deficit",
                "moisture_deficit", "absolute_humidity_deficit",
            ),
            help_text=(
                "온실 공기의 수분부족분(일반적으로 g/m³)을 선택합니다. "
                "None이면 수분부족분 관련 분석을 제외합니다."
            ),
        )

    st.markdown("---")
    st.markdown("**수확량 데이터 컬럼 선택**")
    st.caption(
        "조사일자는 통합 기준일이므로 필수이며, 수확수·착과수·개화수·평균과중은 "
        "해당 컬럼이 없으면 None을 선택할 수 있습니다."
    )
    c7, c8, c9 = st.columns(3)
    c10, c11, c12 = st.columns(3)
    with c7:
        date_col_yield = select_optional_column(
            "조사일자",
            yield_df.columns,
            key="yield_date_col",
            preferred_names=(
                "조사일자", "조사일", "date", "Date",
                "수확일자", "측정일자",
            ),
            help_text="환경데이터와 통합할 기준일입니다. 반드시 선택해야 합니다.",
        )
    with c8:
        harvest_count_col = select_optional_column(
            "수확수",
            yield_df.columns,
            key="yield_harvest_count_col",
            preferred_names=("수확수", "harvest_count", "HarvestCount"),
        )
    with c9:
        harvest_weight_col = select_optional_column(
            "착과수",
            yield_df.columns,
            key="yield_fruit_set_col",
            preferred_names=("착과수", "fruit_set", "fruit_count", "착과개수"),
        )
    with c10:
        flower_count_col = select_optional_column(
            "개화수",
            yield_df.columns,
            key="yield_flower_count_col",
            preferred_names=("개화수", "flower_count", "flowers", "개화개수"),
            help_text="개화수 컬럼이 없으면 None을 선택합니다.",
        )
    with c11:
        avg_fruit_weight_col = select_optional_column(
            "평균과중1",
            yield_df.columns,
            key="yield_avg_fruit_weight_col",
            preferred_names=(
                "평균과중", "평균과실중", "평균과실무게",
                "avg_fruit_weight", "average_fruit_weight",
            ),
            help_text="선택한 실제 Feature명이 이후 표·그래프·모델 Target에 표시됩니다. 예: 생체중 선택 → 생체중",
        )
    with c12:
        avg_fruit_weight_col2 = select_optional_column(
            "평균과중2",
            yield_df.columns,
            key="yield_avg_fruit_weight_col2",
            preferred_names=("건물중", "건조중량", "dry_weight", "평균과중2"),
            help_text="선택한 실제 Feature명이 이후 표시명으로 사용됩니다. 예: 건물중 선택 → 건물중",
        )

    avg_fruit_weight_label = str(avg_fruit_weight_col) if avg_fruit_weight_col is not None else "평균과중1"
    avg_fruit_weight_label2 = str(avg_fruit_weight_col2) if avg_fruit_weight_col2 is not None else "평균과중2"

    st.markdown("---")
    st.markdown("**추가 생육 컬럼 선택**")

    if crop_name == "토마토":
        growth_features = ["초장", "엽장", "엽폭", "생장길이", "줄기굵기", "화방높이", "엽수"]
    else:
        growth_features = ["초장", "엽장", "엽폭", "생장길이", "줄기굵기", "화방높이", "엽수"]

    growth_cols = {}
    for i in range(0, len(growth_features), 3):
        cols = st.columns(3)
        for j, gf in enumerate(growth_features[i:i + 3]):
            with cols[j]:
                options = [None] + yield_df.columns.tolist()
                default_idx = yield_df.columns.get_loc(gf) + 1 if gf in yield_df.columns else 0
                growth_cols[gf] = st.selectbox(gf, options, index=default_idx, key=f"growth_{gf}")

    # 날짜 컬럼은 데이터 통합에 반드시 필요합니다.
    missing_required_dates = []
    if date_col_sensor is None:
        missing_required_dates.append("환경센서 날짜시간")
    if date_col_yield is None:
        missing_required_dates.append("수확/생육 조사일자")

    if missing_required_dates:
        st.warning(
            "다음 필수 날짜 컬럼을 선택해야 분석을 시작할 수 있습니다: "
            + ", ".join(missing_required_dates)
        )
        st.stop()

    # 날짜 처리
    sensor_df[date_col_sensor] = pd.to_datetime(
        sensor_df[date_col_sensor],
        errors="coerce",
    )
    yield_df[date_col_yield] = pd.to_datetime(
        yield_df[date_col_yield],
        errors="coerce",
    )
    sensor_df = sensor_df.dropna(subset=[date_col_sensor]).copy()
    yield_df = yield_df.dropna(subset=[date_col_yield]).copy()

    if sensor_df.empty or yield_df.empty:
        st.error(
            "날짜 변환 후 사용할 수 있는 행이 없습니다. "
            "선택한 날짜 컬럼의 형식을 확인하세요."
        )
        st.stop()

    sensor_df["date"] = sensor_df[date_col_sensor].dt.date
    sensor_df["hour"] = sensor_df[date_col_sensor].dt.hour
    sensor_df["time"] = sensor_df[date_col_sensor].dt.time

    # 선택한 컬럼만 수치형으로 변환합니다.
    selected_sensor_numeric_cols = [
        col
        for col in [
            temp_col,
            hum_col,
            co2_col,
            solar_col,
            moisture_deficit_col,
        ]
        if col is not None and col in sensor_df.columns
    ]
    for col in selected_sensor_numeric_cols:
        sensor_df[col] = pd.to_numeric(
            sensor_df[col],
            errors="coerce",
        )

    selected_yield_numeric_cols = [
        col
        for col in [
            harvest_count_col,
            harvest_weight_col,
            flower_count_col,
            avg_fruit_weight_col,
            avg_fruit_weight_col2,
        ]
        + [c for c in growth_cols.values() if c is not None]
        if col is not None and col in yield_df.columns
    ]
    for col in selected_yield_numeric_cols:
        yield_df[col] = pd.to_numeric(
            yield_df[col],
            errors="coerce",
        )

    # v29.3: Calendar × Phenology 이중정렬 모듈에서 재사용할 원자료/컬럼 매핑을 세션에 보존합니다.
    # 데스크탑에서 여러 작기를 순차 테스트할 때 현재 업로드 작기의 원환경과 생육조사 날짜를
    # DAP/WAP/GDD/계절축에 재정렬하는 데 사용합니다.
    st.session_state["v293_sensor_df"] = sensor_df.copy()
    st.session_state["v293_yield_df"] = yield_df.copy()
    st.session_state["v293_sensor_date_col"] = date_col_sensor
    st.session_state["v293_yield_date_col"] = date_col_yield
    st.session_state["v293_indoor_temp_col"] = temp_col
    st.session_state["v293_solar_col"] = solar_col
    st.session_state["v293_growth_cols"] = dict(growth_cols)
    st.session_state["v293_harvest_count_col"] = harvest_count_col
    st.session_state["v293_avg_fruit_weight_col"] = avg_fruit_weight_col

    if not selected_sensor_numeric_cols:
        st.info(
            "현재 환경변수 선택이 모두 None입니다. 매핑데이터의 날짜·수확·생육 항목은 "
            "확인할 수 있지만, 환경그래프·모델·XAI 분석을 수행하려면 환경변수를 하나 이상 선택해야 합니다."
        )

    if "weeks" not in st.session_state:
        st.session_state.weeks = 7

    def update_weeks_1():
        st.session_state.weeks = st.session_state.weeks_slider_1

    period_unit = st.radio(
        "평균 계산 기간 단위 - 센서 평균용",
        ["주 단위", "일 단위"],
        horizontal=True,
        key="sensor_avg_period_unit",
    )
    if period_unit == "주 단위":
        weeks1 = st.slider("평균 계산 기간 (주 단위) - 센서 평균용", 1, 7, st.session_state.weeks, key="weeks_slider_1", on_change=update_weeks_1)
        selected_week = st.session_state.weeks
        selected_day_window = None
    else:
        selected_day_window = st.selectbox(
            "평균 계산 기간 (일 단위) - 센서 평균용",
            [2, 3, 4, 5, 6],
            index=1,
            format_func=lambda x: f"{x}일",
            key="sensor_avg_days",
        )
        selected_week = st.session_state.weeks

    # 선택 주차 데이터 + 전체 1~7주 데이터 생성
    week_dfs = {}
    for week in range(1, 8):
        week_dfs[week] = compute_rolling_summary(
            sensor_df=sensor_df,
            yield_df=yield_df,
            date_col_sensor=date_col_sensor,
            date_col_yield=date_col_yield,
            temp_col=temp_col,
            hum_col=hum_col,
            co2_col=co2_col,
            solar_col=solar_col,
            moisture_deficit_col=moisture_deficit_col,
            harvest_count_col=harvest_count_col,
            harvest_weight_col=harvest_weight_col,
            flower_count_col=flower_count_col,
            avg_fruit_weight_col=avg_fruit_weight_col,
            avg_fruit_weight_col2=avg_fruit_weight_col2,
            avg_fruit_weight_label=avg_fruit_weight_label,
            avg_fruit_weight_label2=avg_fruit_weight_label2,
            growth_cols=growth_cols,
            week=week,
        )

    if selected_day_window is None:
        df = week_dfs[selected_week].copy()
    else:
        df = compute_rolling_summary(
            sensor_df=sensor_df, yield_df=yield_df, date_col_sensor=date_col_sensor, date_col_yield=date_col_yield,
            temp_col=temp_col, hum_col=hum_col, co2_col=co2_col, solar_col=solar_col, moisture_deficit_col=moisture_deficit_col,
            harvest_count_col=harvest_count_col, harvest_weight_col=harvest_weight_col, flower_count_col=flower_count_col,
            avg_fruit_weight_col=avg_fruit_weight_col, avg_fruit_weight_col2=avg_fruit_weight_col2,
            avg_fruit_weight_label=avg_fruit_weight_label, avg_fruit_weight_label2=avg_fruit_weight_label2,
            growth_cols=growth_cols, week=selected_week, window_days=int(selected_day_window),
        )
    # 수확수 안정화용 파생변수 생성:
    # 1~4주 이동평균수확수, 누적수확수, 누적착과수, 착과잔량
    df = add_harvest_enhancement_features(df)

    st.subheader("매핑 데이터")
    derived_cols = [
        c for c in df.columns
        if any(token in str(c) for token in ["ADT(", "DIF(", "GDD(", "VPD("])
    ]
    if derived_cols:
        current_sensor_period_text = (
            f"{int(selected_day_window)}일"
            if selected_day_window is not None
            else f"{int(selected_week)}주"
        )
        st.caption(
            f"v26.1 파생 Feature 생성 완료 ({current_sensor_period_text} 센서 평균): "
            + ", ".join(derived_cols)
            + " · ADT=24시간 평균온도, DIF=주간08~18-야간19~07, "
              "GDD=Base10℃ 누적, VPD=24시간 평균 · GDD 기준온도는 10℃입니다."
        )
    st.dataframe(df)

    # ---------------------------------------------------------
    # GEI 기반 주간 환경구간 누적시간·생육/수확 증감·Centered ALE
    # ---------------------------------------------------------
    # GEI/Cluster Target 표시명은 수확량 데이터에서 실제 선택한 원본 컬럼명을 사용합니다.
    # None인 항목은 추가하지 않습니다.
    gei_growth_cols = dict(growth_cols)

    dynamic_yield_target_pairs = [
        (harvest_count_col, harvest_count_col),
        (harvest_weight_col, harvest_weight_col),
        (flower_count_col, flower_count_col),
        (avg_fruit_weight_col, avg_fruit_weight_col),
        (avg_fruit_weight_col2, avg_fruit_weight_col2),
    ]
    for display_name, source_col in dynamic_yield_target_pairs:
        if (
            display_name is not None
            and source_col is not None
            and source_col in yield_df.columns
        ):
            gei_growth_cols[str(display_name)] = source_col

    render_gei_growth_module(
        sensor_df=sensor_df,
        yield_df=yield_df,
        date_col_sensor=date_col_sensor,
        date_col_yield=date_col_yield,
        temp_col=temp_col,
        hum_col=hum_col,
        co2_col=co2_col,
        solar_col=solar_col,
        growth_cols=gei_growth_cols,
        # 이미 gei_growth_cols에 실제 선택 컬럼명으로 포함했으므로
        # canonical "수확수"/"착과수" 중복 추가는 비활성화합니다.
        harvest_count_col=None,
        fruit_set_col=None,
    )

    # 환경 시계열 표시용 컬럼명:
    # 주 단위면 N주..., 일 단위면 N일... 로 매핑데이터의 실제 생성 컬럼명과 일치시킵니다.
    sensor_window_prefix = (
        f"{int(selected_day_window)}일"
        if selected_day_window is not None
        else f"{int(selected_week)}주"
    )
    temp_day_col_name = f"{sensor_window_prefix}평균주간온도(08~18시)"
    temp_night_col_name = f"{sensor_window_prefix}평균야간온도(19~07시)"
    hum_day_col_name = f"{sensor_window_prefix}평균주간습도(08~18시)"
    hum_night_col_name = f"{sensor_window_prefix}평균야간습도(19~07시)"
    co2_day_col_name = f"{sensor_window_prefix}평균주간CO₂(08~18시)"
    co2_night_col_name = f"{sensor_window_prefix}평균야간CO₂(19~07시)"
    solar_col_name = f"{sensor_window_prefix}평균누적일사량(1일최대값기준)"
    moisture_deficit_col_name = f"{sensor_window_prefix}평균수분부족분(24시간)"

    # 모델 학습은 선택 주차 기준
    st.subheader("🧩 머신러닝 입력 환경변수 선택")
    st.caption(
        "여기서 선택한 환경변수만 모델 학습과 SHAP·FI·Permutation Importance·LIME·"
        "ICE/PDP·Centered ALE 등 XAI 입력 Feature로 사용합니다."
    )

    all_model_feature_columns = get_environment_feature_columns(
        df, apply_user_selection=False
    )
    available_feature_groups = []
    for _feature_col in all_model_feature_columns:
        _group = _model_feature_group_name(_feature_col)
        if _group is not None and _group not in available_feature_groups:
            available_feature_groups.append(_group)

    preferred_group_order = [
        "주간온도", "야간온도", "주간습도", "야간습도",
        "주간CO₂", "야간CO₂", "누적일사량", "수분부족분",
        "ADT", "DIF", "GDD", "VPD",
    ]
    available_feature_groups = [
        g for g in preferred_group_order if g in available_feature_groups
    ]
    basic_7_groups = [
        g for g in [
            "주간온도", "야간온도", "주간습도", "야간습도",
            "주간CO₂", "야간CO₂", "누적일사량",
        ] if g in available_feature_groups
    ]
    derived_groups = [g for g in ["ADT", "DIF", "GDD", "VPD"] if g in available_feature_groups]

    selector_key = "selected_model_feature_groups"
    if selector_key not in st.session_state:
        st.session_state[selector_key] = basic_7_groups if basic_7_groups else available_feature_groups
    else:
        # 업로드/컬럼 매핑 변경 후 존재하지 않는 선택값은 자동 제거합니다.
        st.session_state[selector_key] = [
            g for g in st.session_state[selector_key] if g in available_feature_groups
        ]

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("전체 선택", key="feature_select_all", use_container_width=True):
        st.session_state[selector_key] = list(available_feature_groups)
    if b2.button("기본 7개", key="feature_select_basic7", use_container_width=True):
        st.session_state[selector_key] = list(basic_7_groups)
    if b3.button("파생변수 포함", key="feature_select_with_derived", use_container_width=True):
        st.session_state[selector_key] = list(dict.fromkeys(basic_7_groups + derived_groups))
    if b4.button("선택 초기화", key="feature_select_clear", use_container_width=True):
        st.session_state[selector_key] = []

    selected_model_feature_groups = st.multiselect(
        "모델/XAI에 사용할 환경변수",
        options=available_feature_groups,
        key=selector_key,
        help=(
            "기본 7개는 주·야 온도/습도/CO₂와 누적일사량입니다. "
            "파생변수 포함은 기본 7개에 ADT·DIF·GDD·VPD를 추가합니다. "
            "수분부족분은 전체 선택 또는 직접 선택으로 포함할 수 있습니다."
        ),
    )
    st.info(
        f"선택된 Feature : **{len(selected_model_feature_groups)} / {len(available_feature_groups)}개** · "
        + (", ".join(selected_model_feature_groups) if selected_model_feature_groups else "선택 없음")
    )

    # ---------------------------------------------------------
    # 머신러닝 입력 환경변수 선택 아래: 환경/생육·수확 그래프 선택
    # ---------------------------------------------------------
    st.markdown("#### 환경 그래프로 표시할 항목 선택")

    # None으로 선택한 변수는 환경그래프·기준표·월별 분류표에서 제외합니다.
    env_mapping = {}
    optional_environment_items = [
        (temp_col, temp_day_col_name),
        (temp_col, temp_night_col_name),
        (hum_col, hum_day_col_name),
        (hum_col, hum_night_col_name),
        (co2_col, co2_day_col_name),
        (co2_col, co2_night_col_name),
        (solar_col, solar_col_name),
        (moisture_deficit_col, moisture_deficit_col_name),
    ]
    for source_column, mapped_column in optional_environment_items:
        if (
            source_column is not None
            and mapped_column in df.columns
        ):
            env_mapping[mapped_column] = mapped_column

    env_cols = st.multiselect(
        "환경 그래프로 표시할 항목 선택",
        list(env_mapping.keys()),
        default=list(env_mapping.keys()),
        key="environment_graph_columns_v277",
        help=(
            "컬럼 선택 단계에서 None으로 지정한 환경변수는 이 목록에 표시되지 않습니다. "
            "일 단위 센서 평균을 선택하면 2일~6일 접두사가 적용된 실제 매핑 Feature가 표시됩니다."
        ),
    )

    if env_cols:
        for i in range(0, len(env_cols), 2):
            cols = st.columns(2)
            for j, label in enumerate(env_cols[i:i + 2]):
                with cols[j]:
                    true_col = env_mapping[label]
                    if true_col in df.columns:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(
                            df["조사일자"],
                            df[true_col],
                            marker="o",
                            linestyle="-",
                        )
                        ax.set_title(f"{label} 시계열")
                        ax.set_xlabel("조사일자")
                        ax.set_ylabel(label)
                        ax.tick_params(axis="x", rotation=45)
                        ax.grid(True, linestyle="--", alpha=0.5)
                        display_matplotlib(fig)
                        plt.close(fig)

                        env_desc = explain_environment_timeseries(
                            label,
                            df[true_col],
                        )
                        st.markdown(
                            f"""<div style="background:linear-gradient(135deg,#ffffff,#eef5ff);
                            box-shadow:0 6px 20px rgba(0,0,0,0.05);padding:12px;border-radius:10px;
                            line-height:1.8;font-size:15px;margin-bottom:20px">{env_desc}</div>""",
                            unsafe_allow_html=True,
                        )

                        st.markdown("**환경구간 기준표**")
                        st.dataframe(
                            environment_zone_reference_table(label),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.markdown("**월별 환경구간 분류표**")
                        monthly_zone_df = build_monthly_environment_zone_table(
                            df,
                            "조사일자",
                            label,
                            true_col,
                        )
                        st.dataframe(
                            monthly_zone_df,
                            use_container_width=True,
                            hide_index=True,
                        )

    st.markdown("#### 그래프로 표시할 항목 선택")

    df = df.sort_values("조사일자")

    harvest_average_targets = [
        f"{window}주평균수확수"
        for window in range(2, 5)
        if f"{window}주평균수확수" in df.columns
    ]
    extra_graph_targets = harvest_average_targets + ["누적수확수"]

    # 화면표시명 → 내부 매핑데이터 컬럼명.
    # 수확량 데이터에서 None이 아니면 실제 선택 원본 컬럼명을 표시합니다.
    graph_target_map = {}

    dynamic_graph_pairs = [
        (harvest_count_col, "수확수"),
        (harvest_weight_col, "착과수"),
        (flower_count_col, "개화수"),
        (avg_fruit_weight_col, avg_fruit_weight_label),
        (avg_fruit_weight_col2, avg_fruit_weight_label2),
    ]
    for source_name, internal_name in dynamic_graph_pairs:
        if (
            source_name is not None
            and internal_name in df.columns
            and pd.to_numeric(df[internal_name], errors="coerce").notna().any()
        ):
            graph_target_map[str(source_name)] = internal_name

    for internal_name in extra_graph_targets:
        if (
            internal_name in df.columns
            and pd.to_numeric(df[internal_name], errors="coerce").notna().any()
        ):
            graph_target_map[internal_name] = internal_name

    for growth_name in growth_features:
        if (
            growth_name in df.columns
            and pd.to_numeric(df[growth_name], errors="coerce").notna().any()
        ):
            graph_target_map[growth_name] = growth_name

    excluded_timeseries = {
        "1주평균수확수",
        "누적착과수",
        "착과잔량(Fruit Load)",
    }
    graph_target_map = {
        display_name: internal_name
        for display_name, internal_name in graph_target_map.items()
        if internal_name not in excluded_timeseries
    }

    default_graph_labels = []
    for source_name in [harvest_count_col, harvest_weight_col]:
        if source_name is not None and str(source_name) in graph_target_map:
            default_graph_labels.append(str(source_name))

    plot_labels = st.multiselect(
        "그래프로 표시할 항목 선택",
        options=list(graph_target_map.keys()),
        default=default_graph_labels,
        key="growth_graph_columns_v277",
        help=(
            "수확수·착과수·개화수·평균과중1·평균과중2는 None이면 표시하지 않습니다. "
            "선택된 경우 실제 원본 컬럼명(예: 생체중, 건물중)을 그대로 표시합니다."
        ),
    )

    auto_plot_internal = [
        graph_target_map[label]
        for label in plot_labels
        if label in graph_target_map
    ]

    # 원본 수확수 컬럼이 선택되었으면 기존 수확수 파생 시계열도 자동 표시
    harvest_display_name = str(harvest_count_col) if harvest_count_col is not None else None
    if (
        harvest_display_name is not None
        and harvest_display_name in plot_labels
    ):
        auto_targets = ["누적수확수"]
        if int(harvest_avg_weeks) >= 2:
            auto_targets.insert(0, selected_harvest_target)
        for c in auto_targets:
            if (
                c in df.columns
                and c not in excluded_timeseries
                and c not in auto_plot_internal
            ):
                auto_plot_internal.append(c)

    if auto_plot_internal:
        # 내부명 → 실제 표시명 역매핑
        reverse_display = {
            internal: display
            for display, internal in graph_target_map.items()
        }
        for i in range(0, len(auto_plot_internal), 3):
            cols = st.columns(3)
            for j, col_name in enumerate(auto_plot_internal[i:i + 3]):
                with cols[j]:
                    if col_name in df.columns:
                        display_name = reverse_display.get(col_name, col_name)
                        fig, ax = plt.subplots(figsize=(4.5, 3))
                        ax.plot(
                            df["조사일자"],
                            df[col_name],
                            marker="o",
                            linestyle="-",
                        )
                        ax.set_title(f"{display_name} 시계열")
                        ax.set_xlabel("조사일자")
                        ax.set_ylabel(display_name)
                        ax.tick_params(axis="x", rotation=45)
                        ax.grid(True, linestyle="--", alpha=0.5)
                        display_matplotlib(fig)
                        plt.close(fig)

    if not selected_model_feature_groups:
        st.warning("머신러닝과 XAI 분석을 위해 환경변수를 하나 이상 선택하세요.")
        st.stop()

    st.subheader("모델 선택")
    model_options = ["RandomForest", "GradientBoosting", "XGBoost", "LGBM", "ANN(인공신경망)", "BPM(베이지안 확률 모델)", "SVM(서포트벡터머신)", "GaussianNB"]
    model_choice = st.selectbox("모델 선택", model_options)
    target_options = (
        ["수확수"]
        + [
            f"{window}주평균수확수"
            for window in range(1, 5)
        ]
        + [
            "착과수",
            "개화수",
            avg_fruit_weight_label,
            avg_fruit_weight_label2,
            "착과잔량(Fruit Load)",
        ]
        + growth_features
    )
    target_options = list(
        dict.fromkeys(
            [
                col for col in target_options
                if (
                    col in df.columns
                    and pd.to_numeric(
                        df[col],
                        errors="coerce",
                    ).notna().any()
                )
            ]
        )
    )

    if not target_options:
        st.warning(
            "예측대상으로 사용할 수 있는 수확·착과·개화·중량·생육 데이터가 없습니다. "
            "수확량 데이터 컬럼 선택에서 수확수·착과수·개화수·평균과중1·평균과중2 또는 추가 생육 컬럼을 하나 이상 선택하세요."
        )
        st.stop()

    # 슬라이더 변경 시 해당 평균기간을 예측대상으로 자동 연결합니다.
    if (
        st.session_state.get("_last_harvest_avg_weeks")
        != harvest_avg_weeks
        or st.session_state.get("target_col_select") not in target_options
    ):
        if selected_harvest_target in target_options:
            st.session_state["target_col_select"] = selected_harvest_target
        elif target_options:
            st.session_state["target_col_select"] = target_options[0]
        st.session_state["_last_harvest_avg_weeks"] = harvest_avg_weeks

    model_target_display_map = {
        "수확수": str(harvest_count_col) if harvest_count_col is not None else "수확수",
        "착과수": str(harvest_weight_col) if harvest_weight_col is not None else "착과수",
        "개화수": str(flower_count_col) if flower_count_col is not None else "개화수",
        avg_fruit_weight_label: str(avg_fruit_weight_col) if avg_fruit_weight_col is not None else avg_fruit_weight_label,
        avg_fruit_weight_label2: str(avg_fruit_weight_col2) if avg_fruit_weight_col2 is not None else avg_fruit_weight_label2,
    }
    target_col = st.selectbox(
        "예측 대상 컬럼 선택",
        target_options,
        key="target_col_select",
        format_func=lambda x: model_target_display_map.get(x, x),
    )
    report_target = get_report_target_name(target_col)

    show_harvest_window_optimizers = is_harvest_window_optimizer_target(target_col)

    if show_harvest_window_optimizers:
        st.info(
            f"현재 선택한 모델은 **{model_choice}**이고, 예측 대상은 **{target_col}**입니다. "
            "모델 학습과 SHAP, Feature Importance, Temporal SHAP, Feature × Week Heatmap, "
            "ICE+PDP, Centered ALE, 1D Bootstrap 95% CI, Threshold Detection, Counterfactual Target Control을 수행합니다. "
            f"현재 환경 입력기간(X-window)은 **{selected_week}주**, "
            f"수확수 평균기간(Y-window)은 **{harvest_avg_weeks}주**입니다. "
            "수확수 계열을 선택했으므로 Y-window 최적화와 X×Y 28개 조합 최적화를 함께 표시합니다."
        )

        st.markdown(
            """
            <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:12px 0 18px 0;">
                <div class="xai-insight-card" style="margin:0;">
                    <b>환경 X-window</b><br>
                    과거 환경데이터를 몇 주까지 입력으로 사용할지를 의미합니다.<br>
                    X가 7주이면 과거 7주 온도·습도·CO₂·일사량·수분부족분 중 선택한 변수를 사용합니다.
                </div>
                <div class="xai-insight-card" style="margin:0;">
                    <b>수확수 Y-window</b><br>
                    예측할 수확수를 몇 주 이동평균으로 만들지를 의미합니다.<br>
                    Y가 3주이면 3개 수확수 자료를 한 칸씩 이동해 평균합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            f"현재 선택한 모델은 **{model_choice}**이고, 예측 대상은 **{target_col}**입니다. "
            "모델 평가와 SHAP, Feature Importance, Temporal SHAP, Feature × Week Heatmap, "
            "ICE+PDP, Centered ALE, 1D Bootstrap 95% CI, Threshold Detection, Counterfactual Target Control을 수행합니다. "
            "Y-window 및 X×Y 28개 조합 최적화는 수확수 또는 1~4주평균수확수를 "
            "예측대상으로 선택한 경우에만 표시됩니다."
        )

    # 모델 학습 및 전체 XAI 분석 Feature는 1~7주 환경변수 Feature 사용
    features = get_environment_feature_columns(df)
    if len(features) == 0:
        st.error("선택한 머신러닝 입력 환경변수에 해당하는 Feature를 찾지 못했습니다. 입력 환경변수 선택과 컬럼 매핑을 확인하세요.")
        st.stop()

    # 수확수 계열을 선택한 경우에만 수확수 전용 최적화 화면을 표시합니다.
    # 화면 순서: ① Y-window optimizer → ② Joint X–Y window optimizer
    harvest_window_comparison_df = pd.DataFrame()
    best_harvest_avg_weeks = None
    xy_window_result_df = pd.DataFrame()
    best_xy_window = None

    if show_harvest_window_optimizers:
        harvest_window_comparison_df, best_harvest_avg_weeks = (
            render_harvest_average_window_optimizer(
                df=df,
                features=features,
                model_choice=model_choice,
                selected_window=harvest_avg_weeks,
                selected_x_window=selected_week,
            )
        )

        xy_window_result_df, best_xy_window = (
            render_xy_window_joint_optimizer(
                week_dfs=week_dfs,
                model_choice=model_choice,
                selected_x_window=selected_week,
                selected_y_window=harvest_avg_weeks,
            )
        )
    X = df[features].copy().apply(pd.to_numeric, errors="coerce").fillna(df[features].mean(numeric_only=True)).fillna(0)
    y = df[target_col].copy()
    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    if len(X) < 6:
        st.error(
            f"선택한 예측대상 '{target_col}'의 유효 데이터가 {len(X)}개입니다. "
            "R² 계산을 위해 최소 6개 이상의 유효 데이터가 필요합니다. "
            "평균기간을 줄이거나 수확 데이터를 추가하세요."
        )
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = make_model(model_choice)
    model.fit(X_train, y_train)
    model_features = get_model_feature_names(model, features)
    features = model_features
    X_train = align_xai_input(X_train, features, model)
    X_test = align_xai_input(X_test, features, model)
    y_pred = safe_predict(model, X_test, features)

    st.subheader("선택한 X-window·Y-window 조합의 모델 평가 지표")
    metrics = compute_metrics(y_test, y_pred)

    if (
        str(target_col) == "수확수"
        and isinstance(metrics, dict)
        and metrics.get("R2", 0) < 0.3
    ):
        st.warning(
            f"현재 수확수 R²={metrics['R2']:.3f}입니다. "
            "위 1~4주 평균기간 R² 비교에서 가장 높은 기간을 선택하면 "
            "수확수 변동성을 완화할 수 있습니다."
        )

    st.markdown(
        f"""
        <div class="xai-insight-card">
            이 지표는 현재 선택한 <b>환경 X-window {selected_week}주</b>와
            <b>예측대상 Y '{target_col}'</b> 조합 하나의 성능입니다.
            위 Y-window 표 및 아래 X-window 표와 역할이 다르며,
            최종 동시 비교는 28개 조합표에서 확인합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div style="background:linear-gradient(135deg,#ffffff,#eaf3ff);
            box-shadow:0 8px 24px rgba(0,0,0,0.08);
            padding:18px;
            border-radius:16px;
            border-left:6px solid #2563eb;
            margin-bottom:14px;">
    <div style="font-size:18px; font-weight:800; color:#183b56;">
        (환경 X-window: {selected_week}주 · 예측대상 Y: {target_col})
    </div>
    <div style="display:flex; gap:18px; margin-top:12px; flex-wrap:wrap;">
        <div style="flex:1; min-width:180px; background:#ffffff; border-radius:14px; padding:14px; box-shadow:0 3px 10px rgba(0,0,0,0.05);">
            <div style="font-size:14px; color:#64748b;">MSE</div>
            <div style="font-size:26px; font-weight:900; color:#1d4ed8;">{metrics['MSE']:.3f}</div>
        </div>
        <div style="flex:1; min-width:180px; background:#ffffff; border-radius:14px; padding:14px; box-shadow:0 3px 10px rgba(0,0,0,0.05);">
            <div style="font-size:14px; color:#64748b;">MAE</div>
            <div style="font-size:26px; font-weight:900; color:#0f766e;">{metrics['MAE']:.3f}</div>
        </div>
        <div style="flex:1; min-width:180px; background:#ffffff; border-radius:14px; padding:14px; box-shadow:0 3px 10px rgba(0,0,0,0.05);">
            <div style="font-size:14px; color:#64748b;">R²</div>
            <div style="font-size:26px; font-weight:900; color:#b45309;">{metrics['R2']:.3f}</div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------
    # 1~7주 모델 성능 비교
    # -------------------------------------------------------------
    render_stylish_section(
        "📊 환경 입력변수 누적기간 성능 비교 (X-window, 1~7주)",
        (
            f"예측대상(Y)은 '{target_col}'로 고정하고, "
            "환경 입력기간(X-window)만 1~7주로 바꾸어 성능을 비교합니다."
        ),
        kicker="X-WINDOW PERFORMANCE",
    )

    st.markdown(
        f"""
        <div class="xai-insight-card">
            <b>이 표에서 바뀌는 것은 환경 입력기간(X)입니다.</b><br>
            예측대상(Y)은 <b>{target_col}</b>로 동일하게 고정됩니다.<br><br>
            예를 들어 <b>환경 7주</b> 행은 “과거 7주 환경정보를 사용해
            {target_col}을 예측한 결과”입니다.
            따라서 이 표는 <b>과거 환경을 몇 주까지 봐야 예측이 가장 잘 되는가?</b>를 찾습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    weekly_metrics = []

    try:

        for wk in range(1, 8):

            wk_df = add_harvest_enhancement_features(week_dfs[wk].copy())

            # 주차별 모델 성능 비교는 현재 선택된 환경변수(수분부족분 포함)를 사용
            wk_features = get_environment_feature_columns(wk_df)

            X_wk = wk_df[wk_features].copy()
            X_wk = X_wk.fillna(X_wk.mean(numeric_only=True))

            if target_col not in wk_df.columns:
                continue
            y_wk = wk_df[target_col].copy()

            valid_mask_wk = y_wk.notna()

            X_wk = X_wk.loc[valid_mask_wk].copy()
            y_wk = y_wk.loc[valid_mask_wk].copy()

            if len(X_wk) < 5:
                continue

            X_train_wk, X_test_wk, y_train_wk, y_test_wk = train_test_split(
                X_wk,
                y_wk,
                test_size=0.2,
                random_state=42
            )

            wk_model = make_model(model_choice)

            wk_model.fit(X_train_wk, y_train_wk)

            preds_wk = safe_predict(
                wk_model,
                X_test_wk,
                wk_features
            )

            mse_wk = mean_squared_error(y_test_wk, preds_wk)
            mae_wk = mean_absolute_error(y_test_wk, preds_wk)
            r2_wk = r2_score(y_test_wk, preds_wk)

            weekly_metrics.append({
                "Week": wk,
                "MSE": mse_wk,
                "MAE": mae_wk,
                "R2": r2_wk
            })

        if len(weekly_metrics) > 0:

            weekly_metrics_df = pd.DataFrame(weekly_metrics)

            # -------------------------------------------------
            # MSE / MAE / R² 그래프 가로 배치
            # -------------------------------------------------
            st.markdown(
                f"### 📈 환경 X-window별 모델 성능 변화 "
                f"(Y 고정: {target_col})"
            )

            col_mse, col_mae, col_r2 = st.columns(3)

            # -------------------------------------------------
            # MSE
            # -------------------------------------------------
            with col_mse:

                fig_mse, ax_mse = plt.subplots(figsize=(4, 3))

                ax_mse.plot(
                    weekly_metrics_df["Week"],
                    weekly_metrics_df["MSE"],
                    marker="o",
                    linewidth=2
                )

                ax_mse.set_xlabel("주차")
                ax_mse.set_ylabel("MSE (평균제곱오차)")
                # ax_mse.set_title("MSE (평균제곱오차)")
                ax_mse.grid(True, linestyle="--", alpha=0.5)

                display_matplotlib(fig_mse)
                plt.close(fig_mse)

            # -------------------------------------------------
            # MAE
            # -------------------------------------------------
            with col_mae:

                fig_mae, ax_mae = plt.subplots(figsize=(4, 3))

                ax_mae.plot(
                    weekly_metrics_df["Week"],
                    weekly_metrics_df["MAE"],
                    marker="o",
                    linewidth=2
                )

                ax_mae.set_xlabel("주차")
                ax_mae.set_ylabel("MAE (평균절대오차)")
                # ax_mae.set_title("MAE (평균절대오차)")
                ax_mae.grid(True, linestyle="--", alpha=0.5)

                display_matplotlib(fig_mae)
                plt.close(fig_mae)

            # -------------------------------------------------
            # R²
            # -------------------------------------------------
            with col_r2:

                fig_r2, ax_r2 = plt.subplots(figsize=(4, 3))

                ax_r2.plot(
                    weekly_metrics_df["Week"],
                    weekly_metrics_df["R2"],
                    marker="o",
                    linewidth=2
                )

                ax_r2.set_xlabel("주차")
                ax_r2.set_ylabel("R² (결정계수)")
                # ax_r2.set_title("R² (결정계수)")
                ax_r2.grid(True, linestyle="--", alpha=0.5)

                display_matplotlib(fig_r2)
                plt.close(fig_r2)

            st.markdown(
                f"### 📋 환경 입력기간(X-window)별 평가지표 "
                f"(Y 고정: {target_col})"
            )
            weekly_display_df = weekly_metrics_df.rename(
                columns={
                    "Week": "환경 입력기간 X(주)",
                    "R2": "R²",
                }
            ).copy()
            st.dataframe(
                weekly_display_df.round(4),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                f"""
                <div class="xai-insight-card">
                    <b>표 읽는 방법</b><br>
                    • 환경 입력기간 X(주): 조사일 기준 과거 환경을 몇 주 누적·평균했는지 의미합니다.<br>
                    • 예측대상 Y: 모든 행에서 <b>{target_col}</b>로 고정되어 있습니다.<br>
                    • R²는 높을수록, MSE와 MAE는 낮을수록 좋습니다.<br><br>
                    예: 환경 입력기간 5주는 “과거 5주 환경정보로 {target_col}을 예측한 성능”입니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # -------------------------------------------------
            # 성능 자동 해석
            # -------------------------------------------------
            best_r2_row = weekly_metrics_df.sort_values(
                "R2",
                ascending=False
            ).iloc[0]

            best_mse_row = weekly_metrics_df.sort_values(
                "MSE",
                ascending=True
            ).iloc[0]

            best_mae_row = weekly_metrics_df.sort_values(
                "MAE",
                ascending=True
            ).iloc[0]

            st.markdown('<div class="pretty-box"><h3>🧠 주차별 성능 자동 해석</h3></div>', unsafe_allow_html=True)

            col_metric_left, col_metric_right = st.columns(2)

            with col_metric_left:

                st.markdown(
                    f"""
<div style="background:linear-gradient(135deg,#ffffff,#eef5ff); box-shadow:0 6px 20px rgba(0,0,0,0.05);
padding:12px;
border-radius:10px;
line-height:1.45;
font-size:15px">

<b>최적 R² 구간</b><br><br>

가장 높은 R² 성능은 <b>{int(best_r2_row['Week'])}주</b>에서 나타났으며,
R² 값은 <b>{best_r2_row['R2']:.4f}</b>입니다.<br><br>

이는 해당 기간의 환경 데이터를 사용할 때
예측 대상({report_target})을 가장 잘 설명할 수 있었음을 의미합니다.<br><br>

<b>최소 MSE 구간</b><br><br>

가장 낮은 MSE는 <b>{int(best_mse_row['Week'])}주</b>에서 나타났으며,
MSE 값은 <b>{best_mse_row['MSE']:.4f}</b>입니다.<br><br>

즉, 해당 기간의 환경 누적 정보가
큰 예측 오차를 가장 작게 만든 구간으로 해석할 수 있습니다.<br><br>

<b>최소 MAE 구간</b><br><br>

가장 낮은 MAE는 <b>{int(best_mae_row['Week'])}주</b>에서 나타났으며,
MAE 값은 <b>{best_mae_row['MAE']:.4f}</b>입니다.<br><br>

이는 실제값과 예측값의 평균적인 차이가 가장 작았던 구간으로,
현장 해석 관점에서 가장 직관적인 오차 최소 구간입니다.<br><br>

일반적으로 R²가 높고 MSE/MAE가 낮을수록
모델 성능이 우수한 것으로 해석합니다.

</div>
                    """,
                    unsafe_allow_html=True
                )

            with col_metric_right:

                st.markdown(
                    """
<div style="background:linear-gradient(135deg,#ffffff,#eef5ff); box-shadow:0 6px 20px rgba(0,0,0,0.05);
padding:12px;
border-radius:10px;
line-height:1.45;
font-size:15px">

<b>R² 특징 설명</b><br><br>

R²(결정계수)는 모델이 실제 데이터 변동성을 얼마나 설명할 수 있는지를 나타냅니다.<br>
1에 가까울수록 설명력이 높으며, 0에 가까우면 평균 예측 수준과 유사한 상태를 의미합니다.<br>
즉, R²가 높을수록 환경데이터와 생육·수확 데이터 간 관계를 잘 학습했다고 볼 수 있습니다.<br><br>

<b>MSE 특징 설명</b><br><br>

MSE(Mean Squared Error)는 실제값과 예측값 차이의 제곱 평균입니다.<br>
큰 오차에 더 민감하게 반응하므로, 이상치나 큰 예측 실패가 존재할 경우 값이 크게 증가합니다.<br>
따라서 MSE가 낮다는 것은 모델이 큰 오차 없이 안정적으로 예측했다는 의미입니다.<br><br>

<b>MAE 특징 설명</b><br><br>

MAE(Mean Absolute Error)는 실제값과 예측값 차이의 절대값 평균입니다.<br>
실제 평균적으로 얼마나 차이가 나는지를 직관적으로 보여주는 지표입니다.<br>
단위가 원래 목표변수와 동일하기 때문에 농업 현장에서는 해석이 비교적 쉬운 장점이 있습니다.

</div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"1~7주 모델 성능 비교 오류: {e}")

    features = model_features
    X_train = align_xai_input(X_train, features, model)
    X_test = align_xai_input(X_test, features, model)

    st.subheader("SHAP / Feature Importance / ICE / PDP / ALE — 자동 리포트 포함")

    shap_values = None
    shap_df = None
    fi_df = None
    pi_df = None
    temporal_df = None
    week_importance = None
    heatmap_df = None
    weekly_metrics_df = None
    cf_result = None
    ice_mean_slope = None
    ice_std_slope = None
    pdp_summary = None
    ale_summary = None
    bin_centers = None
    ale_vals = None


    def build_shap_dependence_quantitative(
        X_dependence,
        shap_array,
        shap_features,
        feature_name,
    ):
        """
        SHAP Dependence Plot의 정량지표를 계산합니다.
        - Pearson / Spearman: Feature 값과 해당 Feature SHAP의 관계
        - Linear slope: SHAP 반응의 1차 평균 기울기
        - Q1/Q4 Mean SHAP: 낮은 값/높은 값 구간에서 평균 기여 방향
        """
        if feature_name not in shap_features:
            return None, pd.DataFrame()

        idx = list(shap_features).index(feature_name)
        x = pd.to_numeric(
            X_dependence[feature_name],
            errors="coerce",
        ).to_numpy(dtype=float)
        s = np.asarray(shap_array, dtype=float)[:, idx]

        valid = np.isfinite(x) & np.isfinite(s)
        x = x[valid]
        s = s[valid]

        if len(x) < 3:
            return None, pd.DataFrame()

        pearson = (
            float(pd.Series(x).corr(pd.Series(s), method="pearson"))
            if np.nanstd(x) > 0 and np.nanstd(s) > 0
            else np.nan
        )
        spearman = (
            float(pd.Series(x).corr(pd.Series(s), method="spearman"))
            if np.nanstd(x) > 0 and np.nanstd(s) > 0
            else np.nan
        )

        try:
            lr_dep = LinearRegression()
            lr_dep.fit(x.reshape(-1, 1), s)
            slope = float(lr_dep.coef_[0])
            intercept = float(lr_dep.intercept_)
            dep_r2 = float(lr_dep.score(x.reshape(-1, 1), s))
        except Exception:
            slope = np.nan
            intercept = np.nan
            dep_r2 = np.nan

        q25 = float(np.nanpercentile(x, 25))
        q75 = float(np.nanpercentile(x, 75))
        low_mask = x <= q25
        high_mask = x >= q75

        low_mean = (
            float(np.nanmean(s[low_mask]))
            if np.any(low_mask)
            else np.nan
        )
        high_mean = (
            float(np.nanmean(s[high_mask]))
            if np.any(high_mask)
            else np.nan
        )

        metrics = {
            "feature": feature_name,
            "n": int(len(x)),
            "pearson": pearson,
            "spearman": spearman,
            "slope": slope,
            "intercept": intercept,
            "linear_r2": dep_r2,
            "q25": q25,
            "q75": q75,
            "low_mean_shap": low_mean,
            "high_mean_shap": high_mean,
            "delta_high_low": (
                float(high_mean - low_mean)
                if np.isfinite(high_mean) and np.isfinite(low_mean)
                else np.nan
            ),
        }

        table = pd.DataFrame([
            {"지표": "표본수", "값": int(len(x))},
            {"지표": "Pearson r (Feature ↔ SHAP)", "값": pearson},
            {"지표": "Spearman ρ (Feature ↔ SHAP)", "값": spearman},
            {"지표": "SHAP 선형 기울기", "값": slope},
            {"지표": "선형 R²", "값": dep_r2},
            {"지표": "Feature Q1(25%)", "값": q25},
            {"지표": "Feature Q3(75%)", "값": q75},
            {"지표": "낮은 25% Mean SHAP", "값": low_mean},
            {"지표": "높은 25% Mean SHAP", "값": high_mean},
            {"지표": "High-Low Mean SHAP 차이", "값": metrics["delta_high_low"]},
        ])

        return metrics, table


    def explain_shap_dependence_quantitative(metrics, target_name):
        if not metrics:
            return "SHAP Dependence 정량 해석을 생성할 수 없습니다."

        feature_name = pretty_time_text(metrics["feature"])
        pearson = metrics["pearson"]
        spearman = metrics["spearman"]
        slope = metrics["slope"]
        delta = metrics["delta_high_low"]

        if np.isfinite(pearson):
            abs_r = abs(pearson)
            if abs_r >= 0.7:
                strength = "매우 강한"
            elif abs_r >= 0.5:
                strength = "강한"
            elif abs_r >= 0.3:
                strength = "중간 수준의"
            elif abs_r >= 0.1:
                strength = "약한"
            else:
                strength = "매우 약한"
            direction = "증가" if pearson > 0 else "감소" if pearson < 0 else "중립"
        else:
            strength = "계산 불가능한"
            direction = "불명확"

        if np.isfinite(delta):
            quartile_text = (
                "Feature가 높은 구간에서 예측 기여가 더 증가"
                if delta > 0
                else "Feature가 높은 구간에서 예측 기여가 더 감소"
                if delta < 0
                else "낮은 구간과 높은 구간의 평균 기여 차이가 거의 없음"
            )
        else:
            quartile_text = "사분위 구간의 SHAP 차이를 계산하기 어렵습니다."

        slope_text = (
            f"SHAP 평균 선형 기울기는 {slope:.5f}"
            if np.isfinite(slope)
            else "SHAP 평균 선형 기울기는 계산되지 않았습니다"
        )

        spearman_text = (
            f"Spearman ρ={spearman:.3f}"
            if np.isfinite(spearman)
            else "Spearman ρ=N/A"
        )

        return (
            f"'{feature_name}' 값과 {target_name} 예측에 대한 해당 Feature의 SHAP 값은 "
            f"Pearson r={pearson:.3f}로 {strength} {direction} 방향 관계를 보입니다. "
            f"{spearman_text}이며, {slope_text}입니다. "
            f"낮은 25%와 높은 25% 구간을 비교하면 {quartile_text}하는 패턴입니다. "
            "SHAP Dependence는 모델 내부의 조건부 예측 기여 패턴이므로 인과효과로 단정하지 않고, "
            "ICE/PDP 및 Centered ALE와 함께 해석하는 것이 적절합니다."
        )


    def compute_shap_interaction_summary(
        model,
        X_input,
        feature_names,
    ):
        """
        Tree SHAP interaction values를 이용하여 변수 쌍 상호작용을 정량화합니다.

        반환값
        -------
        matrix_abs:
            Feature × Feature 평균 |interaction SHAP| 행렬.
            순수 변수상호작용을 보기 위해 대각선(main effect)은 0으로 표시합니다.
        pair_df:
            각 Feature pair의 Mean(|Interaction SHAP|), Mean Interaction SHAP,
            전체 pair interaction 중 비중(%)과 순위.
        """
        X_int = align_xai_input(
            X_input,
            feature_names,
            model,
        ).reset_index(drop=True)

        if X_int.empty or len(X_int.columns) < 2:
            return None, pd.DataFrame(), None

        try:
            interaction_explainer = shap.TreeExplainer(model)
            interaction_values = interaction_explainer.shap_interaction_values(
                X_int
            )
        except Exception:
            return None, pd.DataFrame(), None

        if isinstance(interaction_values, list):
            if not interaction_values:
                return None, pd.DataFrame(), None
            interaction_values = interaction_values[0]

        arr = np.asarray(interaction_values)

        # SHAP/model 버전에 따라 output 차원이 마지막 축에 붙을 수 있음
        if arr.ndim == 4:
            arr = arr[:, :, :, 0]

        if arr.ndim != 3:
            return None, pd.DataFrame(), None

        n_features = min(
            arr.shape[1],
            arr.shape[2],
            len(X_int.columns),
        )
        arr = arr[:, :n_features, :n_features]
        names = list(X_int.columns[:n_features])

        mean_abs = np.nanmean(np.abs(arr), axis=0)
        mean_signed = np.nanmean(arr, axis=0)

        # 대각선은 main effect이므로 interaction heatmap에서는 제거
        matrix_abs = mean_abs.copy()
        np.fill_diagonal(matrix_abs, 0.0)

        rows = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                rows.append(
                    {
                        "Feature 1": names[i],
                        "Feature 2": names[j],
                        "Mean(|Interaction SHAP|)": float(
                            mean_abs[i, j]
                        ),
                        "Mean Interaction SHAP": float(
                            mean_signed[i, j]
                        ),
                    }
                )

        pair_df = pd.DataFrame(rows)
        if pair_df.empty:
            return matrix_abs, pair_df, names

        total = float(
            pair_df["Mean(|Interaction SHAP|)"].sum()
        )
        pair_df["Interaction Share(%)"] = (
            pair_df["Mean(|Interaction SHAP|)"] / total * 100.0
            if total > 0
            else 0.0
        )
        pair_df = (
            pair_df
            .sort_values(
                "Mean(|Interaction SHAP|)",
                ascending=False,
            )
            .reset_index(drop=True)
        )
        pair_df["Rank"] = np.arange(1, len(pair_df) + 1)

        return matrix_abs, pair_df, names


    def explain_shap_interaction_result(
        pair_df,
        target_name,
    ):
        if pair_df is None or pair_df.empty:
            return (
                "SHAP Interaction을 계산할 수 없습니다. "
                "Tree 기반 회귀모델인지와 평가 데이터 수를 확인하세요."
            )

        top = pair_df.iloc[0]
        signed = float(
            top["Mean Interaction SHAP"]
        )
        direction = (
            "두 변수가 함께 나타날 때 평균적으로 예측값을 증가시키는 방향"
            if signed > 0
            else "두 변수가 함께 나타날 때 평균적으로 예측값을 감소시키는 방향"
            if signed < 0
            else "평균 방향성은 거의 중립"
        )

        top3_text = ", ".join(
            [
                (
                    f"{pretty_time_text(r['Feature 1'])} × "
                    f"{pretty_time_text(r['Feature 2'])}"
                    f"({r['Mean(|Interaction SHAP|)']:.4f})"
                )
                for _, r in pair_df.head(3).iterrows()
            ]
        )

        return (
            f"'{target_name}' 예측에서 가장 강한 변수 상호작용은 "
            f"'{pretty_time_text(top['Feature 1'])}' × "
            f"'{pretty_time_text(top['Feature 2'])}'이며 "
            f"Mean(|Interaction SHAP|)="
            f"{top['Mean(|Interaction SHAP|)']:.5f}, "
            f"전체 pair interaction 중 "
            f"{top['Interaction Share(%)']:.1f}%를 차지합니다. "
            f"Mean Interaction SHAP={signed:+.5f}로 {direction}입니다. "
            f"상위 조합은 {top3_text} 순입니다. "
            "SHAP interaction은 모델이 학습한 비선형 결합효과를 설명하며 "
            "생리학적 인과관계를 직접 증명하는 값은 아닙니다."
        )


    def get_local_shap_sample(
        shap_values_input,
        X_input,
        feature_names,
        sample_index,
    ):
        """Waterfall/Force Plot에 공통으로 사용할 1개 샘플 SHAP 정보를 정렬합니다."""
        if shap_values_input is None:
            return None

        X_local = X_input.copy().reset_index(drop=True)
        if X_local.empty:
            return None

        idx = int(
            np.clip(
                int(sample_index),
                0,
                len(X_local) - 1,
            )
        )

        values = np.asarray(
            getattr(
                shap_values_input,
                "values",
                shap_values_input,
            )
        )
        if values.ndim == 3:
            values = values[:, :, 0]
        if values.ndim == 1:
            values = values.reshape(1, -1)

        n_features = min(
            values.shape[1],
            len(feature_names),
            X_local.shape[1],
        )
        names = list(feature_names[:n_features])
        shap_vec = np.asarray(
            values[idx, :n_features],
            dtype=float,
        )
        sample_row = (
            X_local[names]
            .iloc[idx]
            .astype(float)
        )

        base_values = getattr(
            shap_values_input,
            "base_values",
            0.0,
        )
        base_arr = np.asarray(base_values)

        if base_arr.ndim == 0:
            base_value = float(base_arr)
        elif base_arr.ndim == 1:
            if len(base_arr) == len(X_local):
                base_value = float(base_arr[idx])
            else:
                base_value = float(
                    base_arr.reshape(-1)[0]
                )
        else:
            base_value = float(
                np.asarray(base_arr[idx]).reshape(-1)[0]
            )

        local_table = pd.DataFrame(
            {
                "Feature": names,
                "Feature Value": sample_row.values,
                "SHAP Value": shap_vec,
                "|SHAP|": np.abs(shap_vec),
            }
        )
        local_table["Direction"] = np.where(
            local_table["SHAP Value"] > 0,
            "예측 증가",
            np.where(
                local_table["SHAP Value"] < 0,
                "예측 감소",
                "중립",
            ),
        )
        local_table = (
            local_table
            .sort_values("|SHAP|", ascending=False)
            .reset_index(drop=True)
        )

        reconstructed = float(
            base_value + np.nansum(shap_vec)
        )

        return {
            "sample_index": idx,
            "feature_names": names,
            "sample_row": sample_row,
            "shap_values": shap_vec,
            "base_value": base_value,
            "reconstructed_prediction": reconstructed,
            "table": local_table,
        }


    def explain_local_shap_result(
        local_result,
        target_name,
        plot_name,
    ):
        if (
            local_result is None
            or local_result["table"].empty
        ):
            return f"{plot_name} 결과를 해석할 수 없습니다."

        table = local_result["table"]
        top = table.iloc[0]

        pos = (
            table[table["SHAP Value"] > 0]
            .head(3)
        )
        neg = (
            table[table["SHAP Value"] < 0]
            .head(3)
        )

        pos_text = ", ".join(
            [
                (
                    f"{pretty_time_text(r['Feature'])}"
                    f"({r['SHAP Value']:+.4f})"
                )
                for _, r in pos.iterrows()
            ]
        ) or "뚜렷한 양의 기여 없음"

        neg_text = ", ".join(
            [
                (
                    f"{pretty_time_text(r['Feature'])}"
                    f"({r['SHAP Value']:+.4f})"
                )
                for _, r in neg.iterrows()
            ]
        ) or "뚜렷한 음의 기여 없음"

        return (
            f"{plot_name}은 평가 샘플 #{local_result['sample_index']}의 "
            f"'{target_name}' 예측을 설명합니다. "
            f"기준값(Base value)은 {local_result['base_value']:.4f}, "
            f"SHAP 합으로 복원한 예측값은 "
            f"{local_result['reconstructed_prediction']:.4f}입니다. "
            f"가장 큰 개별 기여 Feature는 "
            f"'{pretty_time_text(top['Feature'])}'이며 "
            f"SHAP={top['SHAP Value']:+.5f}입니다. "
            f"예측 증가 주요 Feature: {pos_text}. "
            f"예측 감소 주요 Feature: {neg_text}. "
            "이 결과는 해당 한 샘플에 대한 local explanation이며 "
            "전역 평균효과로 일반화해서는 안 됩니다."
        )

    # =========================================================
    # Global XAI: SHAP Summary / Model FI / Permutation Importance
    # =========================================================
    render_stylish_section(
        "① 🌐 전역 변수 중요도 · SHAP Summary + FI + Permutation",
        (
            f"예측 대상 '{report_target}'에 대해 SHAP Summary, "
            "Model Feature Importance, Permutation Importance를 동일한 모델·Feature 기준으로 비교합니다. "
            "전역 중요도는 어떤 환경변수가 전체 예측에서 반복적으로 중요한지 확인하는 단계입니다."
        ),
        kicker="GLOBAL XAI IMPORTANCE",
    )

    # SHAP Summary: 그래프 + 정량적 결과 2열, 자동해석 하단 배치
    shap_summary_container = st.container()

    with shap_summary_container:
        st.markdown(f"### 🔍 SHAP Summary")
        if model_choice == "GaussianNB":
            st.info("GaussianNB 모델은 SHAP 사용이 제한적입니다.")
        else:
            try:
                X_train_xai = align_xai_input(X_train, features, model)
                X_test_xai = align_xai_input(X_test, features, model)
                explainer = shap.Explainer(model, X_train_xai)
                shap_values = explainer(X_test_xai, check_additivity=False)

                shap_df = summarize_shap_results(shap_values, features)
                shap_plot_features = list(shap_df["Feature"])
                X_test_shap = align_xai_input(
                    X_test, shap_plot_features, model
                )
                shap_plot_values = np.asarray(
                    getattr(shap_values, "values", shap_values)
                )
                if shap_plot_values.ndim == 3:
                    shap_plot_values = shap_plot_values[:, :, 0]
                if shap_plot_values.ndim == 1:
                    shap_plot_values = shap_plot_values.reshape(-1, 1)
                shap_plot_values = shap_plot_values[
                    :, :len(shap_plot_features)
                ]

                shap_graph_col, shap_table_col = st.columns([1.05, 0.95], gap="large")
                with shap_graph_col:
                    fig_shap = plt.figure(figsize=(5.4, 4.2))
                    shap.summary_plot(
                        shap_plot_values, X_test_shap, show=False,
                        max_display=min(12, len(shap_plot_features)),
                    )
                    plt.title("SHAP Summary")
                    display_matplotlib(fig_shap)
                    plt.close(fig_shap)

                with shap_table_col:
                    if not shap_df.empty:
                        render_global_xai_metric_cards(
                            shap_df.iloc[0]["Feature"], float(shap_df["Mean(|SHAP|)"].mean()), "SHAP",
                        )
                    st.markdown("**정량적 결과**")
                    st.dataframe(
                        shap_df[["Feature", "Mean(|SHAP|)", "Mean(SHAP)"]].head(12).round(6),
                        use_container_width=True, hide_index=True, height=380,
                    )
                st.markdown("**자동 해석**")
                st.info(explain_shap_summary(shap_df))
                with st.expander("SHAP 상세 설명"):
                    st.markdown(
                        explain_shap_summary_detail(
                            shap_df, report_target, model_choice
                        ),
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"SHAP 계산/시각화 오류: {e}")


    # 2행: Model Feature Importance + Permutation Importance
    top_col3, top_col4 = st.columns(2, gap="large")

    with top_col3:
        st.markdown("### 📊 Model Feature Importance")
        try:
            fi_df = feature_importance_table(model, features)

            fig_fi, ax_fi = plt.subplots(figsize=(5.4, 4.2))
            fi_plot = fi_df.head(12).sort_values(
                "Importance", ascending=True
            )
            ax_fi.barh(
                fi_plot["Feature"],
                fi_plot["Importance"],
            )
            ax_fi.set_title("Model Feature Importance")
            ax_fi.set_xlabel("Importance")
            ax_fi.grid(
                True, axis="x", linestyle="--", alpha=0.30
            )
            fig_fi.tight_layout()
            display_matplotlib(fig_fi)
            plt.close(fig_fi)

            if not fi_df.empty:
                render_global_xai_metric_cards(
                    fi_df.iloc[0]["Feature"],
                    float(fi_df["Importance"].mean()),
                    "Model FI",
                )

            st.markdown("**정량적 결과**")
            fi_display = fi_df.head(12).copy()
            fi_total = float(fi_df["Importance"].sum())
            fi_display["Importance Ratio(%)"] = (
                fi_display["Importance"] / fi_total * 100.0
                if fi_total > 0
                else 0.0
            )
            st.dataframe(
                fi_display.round(6),
                use_container_width=True,
                hide_index=True,
                height=380,
            )
            st.markdown("**자동 해석**")
            st.info(explain_feature_importance(fi_df))
            with st.expander("Model Feature Importance 상세 설명"):
                st.markdown(
                    explain_feature_importance_detail(
                        fi_df, report_target, model_choice
                    ),
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Feature Importance 처리 오류: {e}")

    with top_col4:
        st.markdown("### 🔀 Permutation Importance")
        try:
            X_test_perm = align_xai_input(
                X_test, features, model
            )
            y_test_perm = pd.Series(
                np.asarray(y_test).reshape(-1),
                index=X_test_perm.index,
            )

            pi_df = permutation_importance_table(
                model=model,
                X_eval=X_test_perm,
                y_eval=y_test_perm,
                features=features,
                n_repeats=30,
                random_state=42,
            )

            if pi_df.empty:
                st.info(
                    "Permutation Importance를 계산할 유효 평가 데이터가 부족합니다."
                )
            else:
                pi_plot = (
                    pi_df.head(12)
                    .sort_values("Importance Mean", ascending=True)
                )
                fig_pi, ax_pi = plt.subplots(
                    figsize=(5.4, 4.2)
                )
                ax_pi.barh(
                    pi_plot["Feature"],
                    pi_plot["Importance Mean"],
                    xerr=pi_plot["Importance Std"],
                    capsize=2,
                )
                ax_pi.axvline(
                    0.0,
                    linewidth=1.2,
                    linestyle="--",
                )
                ax_pi.set_title("Permutation Importance")
                ax_pi.set_xlabel(
                    "Score decrease after permutation"
                )
                ax_pi.grid(
                    True, axis="x", linestyle="--", alpha=0.30
                )
                fig_pi.tight_layout()
                display_matplotlib(fig_pi)
                plt.close(fig_pi)

                render_global_xai_metric_cards(
                    pi_df.iloc[0]["Feature"],
                    float(pi_df["Importance Mean"].mean()),
                    "Permutation",
                )

                st.markdown("**정량적 결과**")
                st.dataframe(
                    pi_df[
                        [
                            "Feature",
                            "Importance Mean",
                            "Importance Std",
                        ]
                    ].head(12).round(6),
                    use_container_width=True,
                    hide_index=True,
                    height=380,
                )

                st.markdown("**자동 해석**")
                st.info(
                    explain_permutation_importance(
                        pi_df,
                        report_target,
                        model_choice,
                    )
                )
                with st.expander("Permutation Importance 해석 기준"):
                    st.markdown(
                        """
                        - **Importance Mean > 0**: 해당 Feature를 섞으면 모델 성능이 감소하므로 예측에 유용한 정보입니다.
                        - **Importance Mean ≈ 0**: Feature를 섞어도 성능 변화가 작아 현재 모델의 의존도가 낮을 수 있습니다.
                        - **Importance Mean < 0**: 섞었을 때 오히려 성능이 개선된 경우로, 표본 변동·과적합·다중공선성 또는 불안정성을 점검해야 합니다.
                        - **Importance Std**가 크면 반복 순열에 따른 변동성이 큰 Feature입니다.
                        """
                    )

        except Exception as e:
            st.error(f"Permutation Importance 처리 오류: {e}")

    # ---------------------------------------------------------
    # SHAP / FI / Permutation 순위 비교
    # ---------------------------------------------------------
    rank_compare_df = build_global_xai_rank_comparison(
        shap_df,
        fi_df,
        pi_df,
    )

    st.markdown("### 🏆 SHAP / FI / Permutation 중요도 순위 비교")
    if rank_compare_df.empty:
        st.info(
            "세 중요도 기법 중 유효 결과가 충분하지 않아 통합 순위표를 생성하지 못했습니다."
        )
    else:
        consensus_top = rank_compare_df.iloc[0]
        rank_c1, rank_c2, rank_c3, rank_c4 = st.columns(4)

        with rank_c1:
            st.metric(
                "Consensus Top Feature",
                str(consensus_top["Feature"]),
            )
        with rank_c2:
            shap_rank_value = consensus_top.get("SHAP Rank", np.nan)
            st.metric(
                "SHAP Rank",
                (
                    f"{int(shap_rank_value)}위"
                    if pd.notna(shap_rank_value)
                    else "N/A"
                ),
            )
        with rank_c3:
            fi_rank_value = consensus_top.get("FI Rank", np.nan)
            st.metric(
                "Model FI Rank",
                (
                    f"{int(fi_rank_value)}위"
                    if pd.notna(fi_rank_value)
                    else "N/A"
                ),
            )
        with rank_c4:
            pi_rank_value = consensus_top.get(
                "Permutation Rank", np.nan
            )
            st.metric(
                "Permutation Rank",
                (
                    f"{int(pi_rank_value)}위"
                    if pd.notna(pi_rank_value)
                    else "N/A"
                ),
            )

        comparison_cols = [
            c for c in [
                "Feature",
                "Mean(|SHAP|)",
                "SHAP Rank",
                "Importance",
                "FI Rank",
                "Importance Mean",
                "Importance Std",
                "Permutation Rank",
                "Mean Rank",
                "Rank Std",
                "Top3 Count",
            ]
            if c in rank_compare_df.columns
        ]

        st.dataframe(
            rank_compare_df[comparison_cols]
            .head(20)
            .round(6),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            explain_global_xai_consensus(rank_compare_df)
        )

        st.markdown(
            """
            <div class="xai-insight-card">
                <b>세 중요도 기법을 함께 보는 이유</b><br>
                <b>SHAP</b>은 각 Feature가 예측값을 얼마나 변화시키는지를 보여주고,
                <b>Model Feature Importance</b>는 모델 내부 학습구조에서 얼마나 많이 활용되었는지를 나타냅니다.
                <b>Permutation Importance</b>는 실제 평가 데이터에서 해당 Feature 정보를 파괴했을 때
                모델 성능이 얼마나 감소하는지를 측정합니다.<br><br>
                따라서 세 방법 모두에서 높은 순위를 보이는 Feature는 서로 다른 중요도 정의에서도
                반복적으로 중요하게 평가된 변수이며, 논문에서는 이를 <b>robust global importance candidate</b>로
                제시할 수 있습니다. 단, 중요도는 인과효과를 의미하지 않으며 변수 간 상관성과 데이터 규모를 함께 고려해야 합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown("---")
    render_stylish_section(
        "② 🔗 변수 상호작용 · SHAP Dependence + SHAP Interaction",
        (
            "SHAP Dependence는 한 Feature 값에 따른 예측 기여 변화와 다른 Feature의 색상 상호작용을 보여주고, "
            "SHAP Interaction은 두 환경변수가 동시에 작용할 때의 비선형 결합효과를 정량화합니다."
        ),
        kicker="FEATURE INTERACTION XAI",
    )

    interaction_col1, interaction_col2 = st.columns(
        2,
        gap="large",
    )

    with interaction_col1:
        st.markdown("### 🔗 SHAP Dependence Plot")

        if model_choice == "GaussianNB":
            st.info("GaussianNB 모델은 SHAP Dependence 분석이 제한적입니다.")
        elif shap_values is None or shap_df is None or shap_df.empty:
            st.info("SHAP Summary가 정상 계산되면 Dependence Plot이 활성화됩니다.")
        else:
            try:
                dep_features = [
                    f for f in list(shap_df["Feature"])
                    if f in X_test_shap.columns
                ]

                default_dep_index = 0
                dep_feature = st.selectbox(
                    "Dependence Feature",
                    dep_features,
                    index=default_dep_index,
                    key="shap_dependence_feature",
                )

                interaction_candidates = [
                    f for f in dep_features
                    if f != dep_feature
                ]
                if interaction_candidates:
                    default_interaction = (
                        interaction_candidates[0]
                    )
                    interaction_feature = st.selectbox(
                        "색상 Interaction Feature",
                        interaction_candidates,
                        index=0,
                        key="shap_dependence_interaction",
                    )
                else:
                    interaction_feature = None

                dep_idx = dep_features.index(dep_feature)
                shap_feature_order = list(shap_df["Feature"])
                shap_idx = shap_feature_order.index(dep_feature)

                dep_x = pd.to_numeric(
                    X_test_shap[dep_feature],
                    errors="coerce",
                ).to_numpy(dtype=float)
                dep_y = np.asarray(
                    shap_plot_values[:, shap_idx],
                    dtype=float,
                )

                dep_valid = np.isfinite(dep_x) & np.isfinite(dep_y)
                dep_x_plot = dep_x[dep_valid]
                dep_y_plot = dep_y[dep_valid]

                fig_dep = go.Figure()

                if interaction_feature is not None:
                    dep_color = pd.to_numeric(
                        X_test_shap.loc[dep_valid, interaction_feature],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    fig_dep.add_trace(
                        go.Scatter(
                            x=dep_x_plot,
                            y=dep_y_plot,
                            mode="markers",
                            marker=dict(
                                size=9,
                                opacity=0.78,
                                color=dep_color,
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(
                                    title=pretty_time_text(
                                        interaction_feature
                                    )
                                ),
                            ),
                            name="SHAP",
                            hovertemplate=(
                                f"{pretty_time_text(dep_feature)}=%{{x:.4f}}<br>"
                                "SHAP=%{y:.4f}<extra></extra>"
                            ),
                        )
                    )
                else:
                    fig_dep.add_trace(
                        go.Scatter(
                            x=dep_x_plot,
                            y=dep_y_plot,
                            mode="markers",
                            marker=dict(
                                size=9,
                                opacity=0.78,
                            ),
                            name="SHAP",
                        )
                    )

                if len(dep_x_plot) >= 3 and np.nanstd(dep_x_plot) > 0:
                    dep_lr = LinearRegression()
                    dep_lr.fit(
                        dep_x_plot.reshape(-1, 1),
                        dep_y_plot,
                    )
                    dep_order = np.argsort(dep_x_plot)
                    dep_x_sorted = dep_x_plot[dep_order]
                    dep_line = dep_lr.predict(
                        dep_x_sorted.reshape(-1, 1)
                    )
                    fig_dep.add_trace(
                        go.Scatter(
                            x=dep_x_sorted,
                            y=dep_line,
                            mode="lines",
                            line=dict(
                                width=3,
                                dash="dash",
                                color="#dc2626",
                            ),
                            name="Linear trend",
                        )
                    )

                fig_dep.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color="#64748b",
                )
                fig_dep.update_layout(
                    title=(
                        f"SHAP Dependence · "
                        f"{pretty_time_text(dep_feature)}"
                    ),
                    xaxis_title=pretty_time_text(dep_feature),
                    yaxis_title="SHAP value",
                    height=420,
                    template=plotly_template,
                    margin=dict(
                        l=45,
                        r=25,
                        t=65,
                        b=45,
                    ),
                    legend=dict(
                        orientation="h",
                        y=1.12,
                        x=1,
                        xanchor="right",
                    ),
                )
                st.plotly_chart(
                    fig_dep,
                    use_container_width=True,
                    key="shap_dependence_plot_v264",
                )

                dep_metrics, dep_table = (
                    build_shap_dependence_quantitative(
                        X_dependence=X_test_shap,
                        shap_array=shap_plot_values,
                        shap_features=shap_feature_order,
                        feature_name=dep_feature,
                    )
                )

                if dep_metrics is not None:
                    dep_m1, dep_m2 = st.columns(2)
                    with dep_m1:
                        st.metric(
                            "Pearson r",
                            (
                                f"{dep_metrics['pearson']:.3f}"
                                if np.isfinite(dep_metrics["pearson"])
                                else "N/A"
                            ),
                        )
                    with dep_m2:
                        st.metric(
                            "High-Low SHAP",
                            (
                                f"{dep_metrics['delta_high_low']:+.3f}"
                                if np.isfinite(
                                    dep_metrics["delta_high_low"]
                                )
                                else "N/A"
                            ),
                        )

                st.markdown("**정량적 결과**")
                st.dataframe(
                    dep_table.round(6),
                    use_container_width=True,
                    hide_index=True,
                    height=380,
                )

                st.markdown("**자동 해석**")
                st.info(
                    explain_shap_dependence_quantitative(
                        dep_metrics,
                        report_target,
                    )
                )

            except Exception as e:
                st.error(
                    f"SHAP Dependence 처리 오류: {e}"
                )


    with interaction_col2:
        st.markdown("### 🧬 SHAP Interaction")

        if model_choice == "GaussianNB":
            st.info(
                "GaussianNB 모델은 Tree SHAP Interaction 분석 대상이 아닙니다."
            )
        elif shap_values is None or shap_df is None or shap_df.empty:
            st.info(
                "SHAP Summary가 정상 계산되면 SHAP Interaction이 활성화됩니다."
            )
        else:
            try:
                interaction_matrix, interaction_df, interaction_names = (
                    compute_shap_interaction_summary(
                        model=model,
                        X_input=X_test_shap,
                        feature_names=list(X_test_shap.columns),
                    )
                )

                if (
                    interaction_matrix is None
                    or interaction_df.empty
                    or interaction_names is None
                ):
                    st.info(
                        "현재 모델/SHAP 버전에서 interaction value를 계산하지 못했습니다. "
                        "RandomForest, GradientBoosting, XGBoost, LGBM 등 Tree 기반 회귀모델을 사용하세요."
                    )
                else:
                    fig_interaction = go.Figure(
                        data=go.Heatmap(
                            z=interaction_matrix,
                            x=[
                                pretty_time_text(v)
                                for v in interaction_names
                            ],
                            y=[
                                pretty_time_text(v)
                                for v in interaction_names
                            ],
                            colorscale="Blues",
                            colorbar=dict(
                                title="Mean<br>|Interaction SHAP|"
                            ),
                            hovertemplate=(
                                "Feature 1=%{y}<br>"
                                "Feature 2=%{x}<br>"
                                "Mean |Interaction SHAP|=%{z:.5f}"
                                "<extra></extra>"
                            ),
                        )
                    )
                    fig_interaction.update_layout(
                        title="SHAP Interaction Heatmap",
                        xaxis_title="Feature",
                        yaxis_title="Feature",
                        height=470,
                        template=plotly_template,
                        margin=dict(
                            l=85,
                            r=25,
                            t=70,
                            b=90,
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(
                        fig_interaction,
                        use_container_width=True,
                        key="shap_interaction_heatmap_v270",
                    )

                    top_inter = interaction_df.iloc[0]
                    int_m1, int_m2 = st.columns(2)
                    with int_m1:
                        st.metric(
                            "Top Interaction",
                            (
                                f"{pretty_time_text(top_inter['Feature 1'])}"
                                " × "
                                f"{pretty_time_text(top_inter['Feature 2'])}"
                            ),
                        )
                    with int_m2:
                        st.metric(
                            "Mean |Interaction SHAP|",
                            f"{top_inter['Mean(|Interaction SHAP|)']:.4f}",
                        )

                    st.markdown("**정량적 결과**")
                    st.dataframe(
                        interaction_df[
                            [
                                "Rank",
                                "Feature 1",
                                "Feature 2",
                                "Mean(|Interaction SHAP|)",
                                "Mean Interaction SHAP",
                                "Interaction Share(%)",
                            ]
                        ]
                        .head(15)
                        .round(6),
                        use_container_width=True,
                        hide_index=True,
                        height=380,
                    )

                    st.markdown("**자동 해석**")
                    st.info(
                        explain_shap_interaction_result(
                            interaction_df,
                            report_target,
                        )
                    )

            except Exception as e:
                st.error(
                    f"SHAP Interaction 처리 오류: {e}"
                )

    # Temporal SHAP + Heatmap
    if shap_values is not None:
        try:
            # 1~7주 전체 선택 환경변수(수분부족분 포함)를 사용하여 주차 설명력 계산
            merged_df = add_harvest_enhancement_features(week_dfs[1].copy())
            # v26.9: 수확량 데이터 컬럼 선택에서 매핑된 모든 Target을
            # Temporal SHAP / Lag SHAP / Feature×Week Heatmap의 대상에 포함합니다.
            # 수확수·착과수·개화수·평균과중은 compute_rolling_summary에서
            # 표준 컬럼명으로 생성되므로, merged_df 구성 단계에서 누락되지 않게 유지합니다.
            temporal_target_candidates = [
                "수확수",
                "착과수",
                "개화수",
                "평균과중",
                "착과잔량(Fruit Load)",
                "누적수확수",
                "누적착과수",
            ]
            keep_cols = (
                ["조사일자"]
                + temporal_target_candidates
                + [f"{window}주평균수확수" for window in range(1, 5)]
                + [
                    col for col in growth_features
                    if col in merged_df.columns
                ]
            )
            keep_cols = [c for c in keep_cols if c in merged_df.columns]
            merged_df = merged_df[keep_cols].copy()
            for week in range(1, 8):
                wk_df = add_harvest_enhancement_features(week_dfs[week].copy())
                add_cols = get_environment_feature_columns(wk_df)
                merged_df = merged_df.merge(wk_df[["조사일자"] + add_cols], on="조사일자", how="left")

            temporal_features = get_environment_feature_columns(merged_df)
            mX = merged_df[temporal_features].copy().fillna(merged_df[temporal_features].mean(numeric_only=True))
            if target_col not in merged_df.columns:
                available_temporal_targets = [
                    c for c in (
                        temporal_target_candidates
                        + [f"{window}주평균수확수" for window in range(1, 5)]
                        + list(growth_features)
                    )
                    if c in merged_df.columns
                ]
                st.warning(
                    f"Temporal SHAP 대상 컬럼 '{target_col}'이 통합 데이터에 없습니다. "
                    f"현재 사용 가능한 대상: {', '.join(available_temporal_targets) if available_temporal_targets else '없음'}"
                )
                raise ValueError(f"Temporal target '{target_col}' not in merged_df")
            my = merged_df[target_col].copy()
            valid_mask2 = my.notna()
            mX = mX.loc[valid_mask2].copy()
            my = my.loc[valid_mask2].copy()

            mX_train, mX_test, my_train, my_test = train_test_split(mX, my, test_size=0.2, random_state=42)
            temporal_model = make_model(model_choice)
            temporal_model.fit(mX_train, my_train)
            temporal_explainer = shap.Explainer(temporal_model, mX_train)
            temporal_shap_values = temporal_explainer(mX_test, check_additivity=False)

            temporal_df, week_importance, heatmap_df = build_temporal_shap_tables(temporal_shap_values, temporal_features)

            if week_importance is not None and not week_importance.empty:
                render_stylish_section(
                    "⏱ Temporal SHAP",
                    "주차별 총 영향도와 방향성을 함께 비교하여 어느 시기의 환경관리가 예측대상에 가장 중요했는지 확인합니다.",
                    kicker="TIME-AWARE EXPLAINABILITY",
                )

                temporal_graph_col, temporal_table_col = st.columns(
                    [1.14, 0.86],
                    gap="large",
                )

                with temporal_graph_col:
                    render_panel_label("주차별 영향도 그래프")
                    ts_df = week_importance.sort_values("Week").copy()
                    best_ts_idx = ts_df["TotalMeanAbsSHAP"].idxmax()
                    best_ts_week = int(ts_df.loc[best_ts_idx, "Week"])

                    fig_ts = go.Figure(
                        go.Bar(
                            x=[f"{int(v)}주" for v in ts_df["Week"]],
                            y=ts_df["TotalMeanAbsSHAP"],
                            text=[
                                f"{v:.3f}"
                                for v in ts_df["TotalMeanAbsSHAP"]
                            ],
                            textposition="outside",
                            marker=dict(
                                color=ts_df["TotalMeanAbsSHAP"],
                                colorscale=[
                                    [0.0, "#bfdbfe"],
                                    [0.5, "#3b82f6"],
                                    [1.0, "#1e3a8a"],
                                ],
                                line=dict(color="rgba(255,255,255,0.85)", width=1.5),
                            ),
                            hovertemplate=(
                                "%{x}<br>총 Mean(|SHAP|)=%{y:.5f}"
                                "<extra></extra>"
                            ),
                            name="주차 영향도",
                        )
                    )
                    fig_ts.add_annotation(
                        x=f"{best_ts_week}주",
                        y=float(ts_df.loc[best_ts_idx, "TotalMeanAbsSHAP"]),
                        text=f"핵심 시점 · {best_ts_week}주 전",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-48,
                        bgcolor="rgba(15,118,110,0.92)",
                        bordercolor="#ffffff",
                        font=dict(color="#ffffff", size=12),
                    )
                    fig_ts.update_layout(
                        height=390,
                        title=dict(
                            text="주차별 환경 영향도",
                            x=0.02,
                            xanchor="left",
                        ),
                        xaxis=dict(title="시점", showgrid=False),
                        yaxis=dict(
                            title="Total Mean(|SHAP|)",
                            gridcolor="rgba(148,163,184,0.22)",
                        ),
                        margin=dict(l=55, r=20, t=64, b=48),
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.78)",
                    )
                    display_plotly(fig_ts)

                with temporal_table_col:
                    render_panel_label("주차별 영향도 결과표")
                    ts_table = week_importance.copy()
                    ts_table["Week"] = (
                        ts_table["Week"]
                        .astype(int)
                        .astype(str)
                        .add("주 전")
                    )
                    ts_table = ts_table.rename(
                        columns={
                            "Week": "시점",
                            "TotalMeanAbsSHAP": "총 |SHAP|",
                            "AvgSignedSHAP": "평균 방향성",
                            "FeatureCount": "변수 수",
                        }
                    )
                    st.dataframe(
                        ts_table.round(5),
                        use_container_width=True,
                        hide_index=True,
                        height=390,
                    )

                st.markdown("**Temporal SHAP 최종 결과**")
                st.markdown(
                    f'<div class="xai-insight-card">{explain_temporal_shap(week_importance, report_target)}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("**Temporal SHAP 지표 상세 설명**")

                st.markdown(
                    """
<div style="background:linear-gradient(135deg,#ffffff,#eef5ff); box-shadow:0 6px 20px rgba(0,0,0,0.05);
padding:14px;
border-radius:10px;
line-height:1.8;
font-size:16px">

<b>TotalMeanAbsSHAP 설명</b><br><br>

TotalMeanAbsSHAP는 해당 주차의 모든 변수들의 평균 절대 SHAP 값을 합산한 값입니다.<br>
즉, 특정 주차의 환경 정보가 전체 예측 결과에 얼마나 강하게 영향을 주었는지를 의미합니다.<br>
값이 클수록 해당 시기의 환경조건이 현재 생육 또는 수확 예측에 매우 중요하게 작용했다는 뜻입니다.<br><br>

<b>AvgSignedSHAP 설명</b><br><br>

AvgSignedSHAP는 해당 주차 변수들의 SHAP 방향성 평균입니다.<br>
양수이면 평균적으로 예측값을 증가시키는 방향으로 작용했고,<br>
음수이면 평균적으로 예측값을 감소시키는 방향으로 작용했음을 의미합니다.<br>
즉, 해당 시기의 환경이 생육/수확에 긍정적이었는지 부정적이었는지를 해석할 수 있습니다.

</div>
                    """,
                    unsafe_allow_html=True
                )


            else:
                st.warning("Temporal SHAP를 계산할 수 없습니다. 주차 feature가 충분하지 않습니다.")

            # =====================================================
            # v26.5 Lag SHAP
            # =====================================================
            if (
                temporal_df is not None
                and not temporal_df.empty
                and temporal_shap_values is not None
            ):
                render_stylish_section(
                    "🕒 Lag SHAP · 지연 시점 중요도 분석",
                    (
                        "주차별로 생성된 환경 Feature의 SHAP 값을 lag 단위로 재집계하여 "
                        "Peak Lag, 중요도 순위, 누적 기여도, 95% Bootstrap CI와 "
                        "Lag × Feature 조합을 정량적으로 분석합니다."
                    ),
                    kicker="LAG-WISE SHAP ATTRIBUTION",
                )

                lag_bootstrap_repeats = st.select_slider(
                    "Lag SHAP Bootstrap 반복횟수",
                    options=[100, 200, 500, 1000, 2000],
                    value=1000,
                    key="lag_shap_bootstrap_repeats_v265",
                    help=(
                        "평가 샘플별 Lag SHAP 행을 복원추출하여 "
                        "각 lag의 Mean(|SHAP|) 평균에 대한 95% percentile CI를 계산합니다. "
                        "모델을 매 반복 재학습하는 bootstrap은 아닙니다."
                    ),
                )

                lag_result = build_lag_shap_analysis(
                    shap_values=temporal_shap_values,
                    features=temporal_features,
                    n_bootstrap=int(
                        lag_bootstrap_repeats
                    ),
                    confidence=0.95,
                    random_state=42,
                )

                if (
                    lag_result is None
                    or lag_result["lag_table"].empty
                ):
                    st.warning(
                        "Lag SHAP을 계산할 수 없습니다. "
                        "1~7주 형식의 주차 Feature가 충분한지 확인하세요."
                    )
                else:
                    lag_df = lag_result[
                        "lag_table"
                    ].copy()

                    peak_lag = int(
                        lag_result["peak_lag"]
                    )
                    peak_share = float(
                        lag_result["peak_share"]
                    )
                    # v29.0: 작기 Knowledge Base에 저장할 수 있도록 최근 Lag SHAP 요약을 보존합니다.
                    st.session_state["lag_shap_peak_week"] = int(peak_lag)
                    st.session_state["lag_shap_peak_share"] = float(peak_share)
                    st.session_state["lag_shap_target"] = str(target_col) if target_col is not None else ""
                    # v29.3: Lag SHAP의 평가샘플을 조사일과 다시 연결해
                    # 작기 × 생육단계 × 계절별 Peak Lag을 요약할 수 있도록 보존합니다.
                    try:
                        st.session_state["v293_lag_sample_abs_matrix"] = np.asarray(
                            lag_result.get("sample_abs_matrix", np.empty((0, 0))), dtype=float
                        )
                        st.session_state["v293_lag_values"] = lag_df["Lag"].astype(int).tolist()
                        st.session_state["v293_lag_test_dates"] = pd.to_datetime(
                            merged_df.loc[mX_test.index, "조사일자"], errors="coerce"
                        ).astype(str).tolist()
                    except Exception:
                        st.session_state["v293_lag_sample_abs_matrix"] = np.empty((0, 0))
                        st.session_state["v293_lag_values"] = []
                        st.session_state["v293_lag_test_dates"] = []

                    top3_lags = (
                        lag_df.sort_values(
                            "MeanAbsLagSHAP",
                            ascending=False,
                        )
                        .head(3)
                    )

                    lag_metric1, lag_metric2, lag_metric3, lag_metric4 = st.columns(
                        4
                    )
                    with lag_metric1:
                        st.metric(
                            "Peak Lag",
                            f"{peak_lag}주 전",
                        )
                    with lag_metric2:
                        st.metric(
                            "Peak Lag 비중",
                            f"{peak_share:.1f}%",
                        )
                    with lag_metric3:
                        st.metric(
                            "Peak Lag 95% CI",
                            (
                                f"{lag_result['peak_ci_lower']:.3f}"
                                f" ~ "
                                f"{lag_result['peak_ci_upper']:.3f}"
                            ),
                        )
                    with lag_metric4:
                        st.metric(
                            "유효 Bootstrap",
                            (
                                f"{lag_result['valid_bootstrap']}"
                                f"/{lag_result['requested_bootstrap']}"
                            ),
                        )

                    lag_graph_col, lag_table_col = st.columns(
                        [1.16, 0.84],
                        gap="large",
                    )

                    with lag_graph_col:
                        render_panel_label(
                            "Lag Importance + 95% CI + Cumulative Lag SHAP"
                        )

                        lag_plot_df = lag_df.sort_values(
                            "Lag"
                        ).copy()

                        fig_lag = make_subplots(
                            specs=[
                                [
                                    {
                                        "secondary_y": True
                                    }
                                ]
                            ]
                        )

                        error_plus = (
                            lag_plot_df[
                                "CI95_Upper"
                            ]
                            - lag_plot_df[
                                "MeanAbsLagSHAP"
                            ]
                        ).clip(lower=0.0)

                        error_minus = (
                            lag_plot_df[
                                "MeanAbsLagSHAP"
                            ]
                            - lag_plot_df[
                                "CI95_Lower"
                            ]
                        ).clip(lower=0.0)

                        fig_lag.add_trace(
                            go.Bar(
                                x=[
                                    f"{int(v)}주 전"
                                    for v in lag_plot_df[
                                        "Lag"
                                    ]
                                ],
                                y=lag_plot_df[
                                    "MeanAbsLagSHAP"
                                ],
                                text=[
                                    f"{v:.3f}"
                                    for v in lag_plot_df[
                                        "MeanAbsLagSHAP"
                                    ]
                                ],
                                textposition="outside",
                                error_y=dict(
                                    type="data",
                                    symmetric=False,
                                    array=error_plus,
                                    arrayminus=error_minus,
                                    thickness=1.5,
                                    width=4,
                                ),
                                marker=dict(
                                    color=lag_plot_df[
                                        "MeanAbsLagSHAP"
                                    ],
                                    colorscale=[
                                        [
                                            0.0,
                                            "#dbeafe",
                                        ],
                                        [
                                            0.5,
                                            "#60a5fa",
                                        ],
                                        [
                                            1.0,
                                            "#1d4ed8",
                                        ],
                                    ],
                                    line=dict(
                                        color="rgba(255,255,255,0.85)",
                                        width=1.5,
                                    ),
                                ),
                                name="Lag Mean(|SHAP|)",
                                hovertemplate=(
                                    "%{x}<br>"
                                    "Lag Mean(|SHAP|)=%{y:.5f}"
                                    "<extra></extra>"
                                ),
                            ),
                            secondary_y=False,
                        )

                        fig_lag.add_trace(
                            go.Scatter(
                                x=[
                                    f"{int(v)}주 전"
                                    for v in lag_plot_df[
                                        "Lag"
                                    ]
                                ],
                                y=lag_plot_df[
                                    "CumulativePct"
                                ],
                                mode="lines+markers+text",
                                text=[
                                    f"{v:.1f}%"
                                    for v in lag_plot_df[
                                        "CumulativePct"
                                    ]
                                ],
                                textposition="top center",
                                line=dict(
                                    width=3,
                                    color="#f59e0b",
                                    shape="spline",
                                ),
                                marker=dict(
                                    size=9,
                                    color="#f59e0b",
                                ),
                                name="Cumulative %",
                                hovertemplate=(
                                    "%{x}<br>"
                                    "누적기여=%{y:.1f}%"
                                    "<extra></extra>"
                                ),
                            ),
                            secondary_y=True,
                        )

                        peak_row = lag_plot_df[
                            lag_plot_df["Lag"]
                            == peak_lag
                        ].iloc[0]

                        fig_lag.add_annotation(
                            x=f"{peak_lag}주 전",
                            y=float(
                                peak_row[
                                    "MeanAbsLagSHAP"
                                ]
                            ),
                            text=(
                                f"Peak Lag · "
                                f"{peak_lag}주 전"
                            ),
                            showarrow=True,
                            arrowhead=2,
                            ax=0,
                            ay=-55,
                            bgcolor=(
                                "rgba(15,118,110,0.92)"
                            ),
                            bordercolor="#ffffff",
                            font=dict(
                                color="#ffffff",
                                size=12,
                            ),
                        )

                        fig_lag.update_layout(
                            height=440,
                            title=dict(
                                text=(
                                    "Lag SHAP 중요도 · "
                                    "95% Bootstrap CI · 누적 기여도"
                                ),
                                x=0.02,
                                xanchor="left",
                            ),
                            margin=dict(
                                l=55,
                                r=55,
                                t=78,
                                b=55,
                            ),
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                y=1.16,
                                x=1,
                                xanchor="right",
                            ),
                            paper_bgcolor=(
                                "rgba(0,0,0,0)"
                            ),
                            plot_bgcolor=(
                                "rgba(255,255,255,0.80)"
                            ),
                        )
                        fig_lag.update_yaxes(
                            title_text=(
                                "Lag Mean(|SHAP|)"
                            ),
                            secondary_y=False,
                            gridcolor=(
                                "rgba(148,163,184,0.22)"
                            ),
                        )
                        fig_lag.update_yaxes(
                            title_text=(
                                "Cumulative importance (%)"
                            ),
                            range=[0, 105],
                            secondary_y=True,
                            showgrid=False,
                        )
                        fig_lag.update_xaxes(
                            title_text="Lag"
                        )

                        display_plotly(fig_lag)

                    with lag_table_col:
                        render_panel_label(
                            "Lag Importance Ranking"
                        )

                        lag_rank_table = (
                            lag_df.sort_values(
                                [
                                    "Rank",
                                    "Lag",
                                ]
                            )
                            .copy()
                        )

                        lag_rank_table[
                            "Lag"
                        ] = (
                            lag_rank_table["Lag"]
                            .astype(int)
                            .astype(str)
                            .add("주 전")
                        )

                        lag_rank_table = (
                            lag_rank_table.rename(
                                columns={
                                    "Lag": "Lag",
                                    "Rank": "순위",
                                    "MeanAbsLagSHAP": (
                                        "Mean |Lag SHAP|"
                                    ),
                                    "MeanSignedLagSHAP": (
                                        "Mean Signed SHAP"
                                    ),
                                    "ImportancePct": (
                                        "비중(%)"
                                    ),
                                    "CumulativePct": (
                                        "누적비중(%)"
                                    ),
                                    "CI95_Lower": (
                                        "95% CI Lower"
                                    ),
                                    "CI95_Upper": (
                                        "95% CI Upper"
                                    ),
                                    "FeatureCount": (
                                        "Feature 수"
                                    ),
                                }
                            )
                        )

                        rank_cols = [
                            "순위",
                            "Lag",
                            "Mean |Lag SHAP|",
                            "비중(%)",
                            "누적비중(%)",
                            "95% CI Lower",
                            "95% CI Upper",
                            "Mean Signed SHAP",
                            "Feature 수",
                        ]
                        st.dataframe(
                            lag_rank_table[
                                rank_cols
                            ].round(5),
                            use_container_width=True,
                            hide_index=True,
                            height=440,
                        )

                    # -----------------------------------------
                    # Top-3 Lag
                    # -----------------------------------------
                    st.markdown(
                        "**Peak Lag / Top-3 Lag 자동 탐지**"
                    )

                    top3_cols = st.columns(
                        min(
                            3,
                            len(top3_lags),
                        )
                    )
                    for top_idx, (
                        _,
                        top_row,
                    ) in enumerate(
                        top3_lags.iterrows()
                    ):
                        with top3_cols[
                            top_idx
                        ]:
                            st.metric(
                                (
                                    f"Top {top_idx + 1} Lag"
                                ),
                                (
                                    f"{int(top_row['Lag'])}"
                                    "주 전"
                                ),
                                delta=(
                                    f"{top_row['ImportancePct']:.1f}%"
                                ),
                            )

                    st.markdown(
                        "**Lag SHAP 자동 해석**"
                    )
                    st.markdown(
                        (
                            '<div class="xai-insight-card">'
                            + explain_lag_shap_result(
                                lag_result,
                                report_target,
                            )
                            + "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        "Lag SHAP 정의: 명시적 1~7주 Feature의 SHAP 값을 "
                        "lag 단위로 집계한 분석입니다. 별도의 독립 Shapley 알고리즘으로 "
                        "주장하기보다 'lag-wise aggregation of SHAP values'로 Methods에 "
                        "정의하는 것이 안전합니다."
                    )

                    # -----------------------------------------
                    # Lag SHAP × Feature Heatmap
                    # -----------------------------------------
                    render_panel_label(
                        "Lag SHAP × Feature Heatmap"
                    )

                    lag_heatmap_mode = st.radio(
                        "Heatmap 표시값",
                        options=[
                            "Mean(|SHAP|) · 영향 크기",
                            "Mean(SHAP) · 방향성",
                        ],
                        horizontal=True,
                        key=(
                            "lag_shap_feature_"
                            "heatmap_mode_v265"
                        ),
                    )

                    if (
                        lag_heatmap_mode
                        == "Mean(|SHAP|) · 영향 크기"
                    ):
                        lag_hm = lag_result[
                            "abs_heatmap"
                        ].copy()
                        lag_hm_zmid = None
                        lag_hm_scale = (
                            "Blues"
                        )
                        lag_color_title = (
                            "Mean<br>|SHAP|"
                        )
                        lag_hover_metric = (
                            "Mean(|SHAP|)"
                        )
                    else:
                        lag_hm = lag_result[
                            "signed_heatmap"
                        ].copy()
                        lag_hm_zmid = 0
                        lag_hm_scale = (
                            "RdBu"
                        )
                        lag_color_title = (
                            "Mean<br>SHAP"
                        )
                        lag_hover_metric = (
                            "Mean(SHAP)"
                        )

                    if (
                        lag_hm is not None
                        and not lag_hm.empty
                    ):
                        heatmap_kwargs = dict(
                            z=lag_hm.values,
                            x=[
                                f"{int(c)}주 전"
                                for c in lag_hm.columns
                            ],
                            y=[
                                pretty_time_text(v)
                                for v in lag_hm.index
                            ],
                            colorscale=lag_hm_scale,
                            colorbar=dict(
                                title=lag_color_title,
                                thickness=14,
                                len=0.84,
                            ),
                            hovertemplate=(
                                "Feature=%{y}<br>"
                                "Lag=%{x}<br>"
                                + lag_hover_metric
                                + "=%{z:.5f}"
                                "<extra></extra>"
                            ),
                        )

                        if lag_hm_zmid is not None:
                            heatmap_kwargs[
                                "zmid"
                            ] = lag_hm_zmid

                        fig_lag_hm = go.Figure(
                            data=go.Heatmap(
                                **heatmap_kwargs
                            )
                        )
                        fig_lag_hm.update_layout(
                            height=max(
                                430,
                                min(
                                    700,
                                    110
                                    + 36
                                    * len(
                                        lag_hm.index
                                    ),
                                ),
                            ),
                            title=dict(
                                text=(
                                    "Lag SHAP × "
                                    "Environment Feature"
                                ),
                                x=0.02,
                                xanchor="left",
                            ),
                            xaxis=dict(
                                title="Lag",
                                side="bottom",
                            ),
                            yaxis=dict(
                                title="환경 Feature",
                                automargin=True,
                            ),
                            margin=dict(
                                l=155,
                                r=25,
                                t=70,
                                b=55,
                            ),
                            paper_bgcolor=(
                                "rgba(0,0,0,0)"
                            ),
                            plot_bgcolor=(
                                "rgba(255,255,255,0.80)"
                            ),
                        )
                        display_plotly(
                            fig_lag_hm
                        )

                        st.markdown(
                            "**Lag SHAP × Feature Heatmap 자동 해석**"
                        )
                        st.markdown(
                            (
                                '<div class="xai-insight-card">'
                                + explain_lag_feature_heatmap(
                                    lag_result,
                                    report_target,
                                )
                                + "</div>"
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info(
                            "Lag SHAP × Feature Heatmap을 "
                            "생성할 데이터가 부족합니다."
                        )

            if heatmap_df is not None and not heatmap_df.empty:
                render_stylish_section(
                    "🔥 Feature × Week Heatmap",
                    "환경변수와 주차의 조합별 영향도를 색의 강도로 표현하여 핵심 변수와 핵심 관리시점을 동시에 식별합니다.",
                    kicker="FEATURE × TIME MATRIX",
                )

                heatmap_graph_col, heatmap_table_col = st.columns(
                    [1.14, 0.86],
                    gap="large",
                )

                with heatmap_graph_col:
                    render_panel_label("Feature × Week 영향도 Heatmap")
                    plotly_scale_map = {
                        "YlOrRd": "YlOrRd",
                        "RdYlGn": "RdYlGn",
                        "Blues": "Blues",
                        "viridis": "Viridis",
                        "turbo": "Turbo",
                        "coolwarm": "RdBu",
                        "Greens": "Greens",
                    }
                    colorscale_name = plotly_scale_map.get(
                        heatmap_cmap,
                        "RdBu",
                    )

                    fig_hm = go.Figure(
                        data=go.Heatmap(
                            z=heatmap_df.values,
                            x=[
                                f"{int(c)}주"
                                for c in heatmap_df.columns
                            ],
                            y=[
                                pretty_time_text(v)
                                for v in heatmap_df.index
                            ],
                            colorscale=colorscale_name,
                            colorbar=dict(
                                title="Mean<br>|SHAP|",
                                thickness=14,
                                len=0.82,
                            ),
                            hovertemplate=(
                                "변수=%{y}<br>시점=%{x}"
                                "<br>Mean(|SHAP|)=%{z:.5f}"
                                "<extra></extra>"
                            ),
                        )
                    )
                    fig_hm.update_layout(
                        height=max(
                            410,
                            min(620, 95 + 34 * len(heatmap_df.index)),
                        ),
                        title=dict(
                            text="환경변수 × 주차 영향도",
                            x=0.02,
                            xanchor="left",
                        ),
                        xaxis=dict(title="시점", side="bottom"),
                        yaxis=dict(title="환경 변수", automargin=True),
                        margin=dict(l=145, r=20, t=64, b=45),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.78)",
                    )
                    display_plotly(fig_hm)

                with heatmap_table_col:
                    render_panel_label("Feature × Week 영향도 결과표")
                    heatmap_table = (
                        temporal_df[
                            [
                                "BaseFeature",
                                "Week",
                                "Mean(|SHAP|)",
                                "Mean(SHAP)",
                            ]
                        ]
                        .sort_values(
                            "Mean(|SHAP|)",
                            ascending=False,
                        )
                        .copy()
                    )
                    heatmap_table["Week"] = (
                        heatmap_table["Week"]
                        .astype(int)
                        .astype(str)
                        .add("주 전")
                    )
                    heatmap_table["BaseFeature"] = (
                        heatmap_table["BaseFeature"]
                        .map(pretty_time_text)
                    )
                    heatmap_table = heatmap_table.rename(
                        columns={
                            "BaseFeature": "환경 변수",
                            "Week": "시점",
                            "Mean(|SHAP|)": "영향도",
                            "Mean(SHAP)": "방향성",
                        }
                    )
                    st.dataframe(
                        heatmap_table.round(5),
                        use_container_width=True,
                        hide_index=True,
                        height=max(
                            410,
                            min(620, 95 + 34 * len(heatmap_df.index)),
                        ),
                    )

                st.markdown("**Feature × Week Heatmap 최종 결과**")
                st.markdown(
                    f'<div class="xai-insight-card">{explain_heatmap(heatmap_df, report_target)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("**Feature × Week Heatmap 상세 설명**")
                st.markdown(
                    f'<div class="xai-insight-card">{explain_heatmap_detail(heatmap_df, temporal_df, report_target)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Heatmap을 계산할 수 없습니다.")

        except Exception as e:
            st.warning(f"Temporal SHAP / Lag SHAP / Heatmap 오류: {e}")

    # ICE / PDP / ALE
    st.markdown("---")
    render_stylish_section(
        "③ 📈 변수 효과 · 경향 · ICE + PDP",
        (
            "ICE는 개별 표본의 반응경로를, PDP는 전체 집단의 평균 반응경향을 보여줍니다. "
            "아래에서 ICE, PDP, ICE+PDP 통합 결과 설명과 최적 예측구간을 함께 확인합니다."
        ),
        kicker="FEATURE EFFECT & TREND",
    )
    features = get_model_feature_names(model, features)
    X_train = align_xai_input(X_train, features, model)
    X_test = align_xai_input(X_test, features, model)
    ice_feature = st.selectbox("분석할 Feature 선택 (ICE/PDP/ALE)", features, key="xai_feature")
    st.info(
        f"현재 선택한 분석 Feature는 **{ice_feature}**입니다. "
        "아래 ICE+PDP와 Centered ALE는 선택 Feature의 개별·집단 반응 및 비선형 효과를 서로 다른 관점에서 해석합니다."
    )
    n_samples = st.slider(
        "ICE 샘플 수",
        min_value=1,
        max_value=max(1, min(18, len(X_test))),
        value=max(1, min(18, len(X_test))),
        step=1,
        key="ice_samples",
        help="개체별 ICE 곡선을 최대 18개까지 표시합니다.",
    )
    ale_bins = st.slider("ALE bins 수", 4, 30, 10)

    def compute_centered_ale(model, X, feature, bins=10):
        x = X[feature].values
        mask = ~np.isnan(x)
        x = x[mask]

        X_valid = X.loc[mask].reset_index(drop=True)

        percentiles = np.linspace(0, 100, bins + 1)
        cutpoints = np.unique(np.percentile(x, percentiles))

        if len(cutpoints) < 2:
            return np.array([np.mean(x)]), np.array([0.0])

        local_effects = []
        bin_centers = []

        for i in range(len(cutpoints) - 1):

            lo = cutpoints[i]
            hi = cutpoints[i + 1]

            in_bin = (
                (X_valid[feature] >= lo) &
                (X_valid[feature] <= hi)
            )

            if in_bin.sum() == 0:
                local_effects.append(0.0)
                bin_centers.append((lo + hi) / 2.0)
                continue

            X_lo = X_valid.copy()
            X_hi = X_valid.copy()

            X_lo.loc[in_bin, feature] = lo
            X_hi.loc[in_bin, feature] = hi

            preds_hi = safe_predict(model, align_xai_input(X_hi, features, model), features)
            preds_lo = safe_predict(model, align_xai_input(X_lo, features, model), features)

            diff = preds_hi - preds_lo

            local_effect = (
                diff[in_bin.values].mean()
                if in_bin.sum() > 0 else 0.0
            )

            local_effects.append(local_effect)
            bin_centers.append((lo + hi) / 2.0)

        ale = np.cumsum(local_effects)

        # --------------------------------
        # Centered ALE
        # --------------------------------
        ale = ale - np.mean(ale)

        return np.array(bin_centers), ale

    def bootstrap_centered_ale_ci(
        model_choice,
        X_train_input,
        y_train_input,
        X_eval_input,
        feature,
        bins=10,
        n_bootstrap=200,
        confidence=0.95,
        random_state=42,
    ):
        """
        Centered ALE 1D Bootstrap Confidence Interval.

        방법
        ----
        1) 학습자료 (X_train, y_train)를 동일 크기로 복원추출합니다.
        2) 각 bootstrap 표본마다 사용자가 선택한 동일 모델을 새로 학습합니다.
        3) 고정 평가자료 X_eval에서 Centered ALE를 계산합니다.
        4) bootstrap마다 달라질 수 있는 ALE 중심점을 원본 ALE 중심점에 보간합니다.
        5) 각 중심점에서 percentile 95% CI(기본 2.5~97.5%)를 계산합니다.

        주의
        ----
        본 CI는 표본 재추출에 따른 모델/ALE 불확실성을 나타내며,
        인과적 효과의 신뢰구간을 의미하지 않습니다.
        """
        X_train_boot = align_xai_input(X_train_input, features, model)
        X_eval_boot = align_xai_input(X_eval_input, features, model)
        y_boot = pd.Series(y_train_input).reset_index(drop=True)
        X_train_boot = X_train_boot.reset_index(drop=True)

        if len(X_train_boot) != len(y_boot):
            n_common = min(len(X_train_boot), len(y_boot))
            X_train_boot = X_train_boot.iloc[:n_common].reset_index(drop=True)
            y_boot = y_boot.iloc[:n_common].reset_index(drop=True)

        base_centers, base_ale = compute_centered_ale(
            model,
            X_eval_boot,
            feature,
            bins=bins,
        )

        if len(base_centers) < 2 or len(X_train_boot) < 6:
            return {
                "centers": np.asarray(base_centers, dtype=float),
                "ale": np.asarray(base_ale, dtype=float),
                "lower": np.asarray(base_ale, dtype=float),
                "upper": np.asarray(base_ale, dtype=float),
                "bootstrap_matrix": np.empty((0, len(base_centers))),
                "valid_bootstrap": 0,
                "requested_bootstrap": int(n_bootstrap),
                "confidence": float(confidence),
            }

        rng = np.random.RandomState(int(random_state))
        curves = []

        for _ in range(int(n_bootstrap)):
            try:
                sample_idx = rng.randint(
                    0,
                    len(X_train_boot),
                    size=len(X_train_boot),
                )
                X_b = X_train_boot.iloc[sample_idx].reset_index(drop=True)
                y_b = y_boot.iloc[sample_idx].reset_index(drop=True)

                boot_model = make_model(model_choice)
                boot_model.fit(X_b, y_b)

                c_b, a_b = compute_centered_ale(
                    boot_model,
                    X_eval_boot,
                    feature,
                    bins=bins,
                )

                c_b = np.asarray(c_b, dtype=float)
                a_b = np.asarray(a_b, dtype=float)

                finite = np.isfinite(c_b) & np.isfinite(a_b)
                c_b = c_b[finite]
                a_b = a_b[finite]

                if len(c_b) < 2:
                    continue

                order = np.argsort(c_b)
                c_b = c_b[order]
                a_b = a_b[order]

                c_unique, unique_idx = np.unique(c_b, return_index=True)
                a_unique = a_b[unique_idx]

                if len(c_unique) < 2:
                    continue

                interp = np.interp(
                    base_centers,
                    c_unique,
                    a_unique,
                    left=a_unique[0],
                    right=a_unique[-1],
                )
                curves.append(interp)

            except Exception:
                continue

        if not curves:
            return {
                "centers": np.asarray(base_centers, dtype=float),
                "ale": np.asarray(base_ale, dtype=float),
                "lower": np.asarray(base_ale, dtype=float),
                "upper": np.asarray(base_ale, dtype=float),
                "bootstrap_matrix": np.empty((0, len(base_centers))),
                "valid_bootstrap": 0,
                "requested_bootstrap": int(n_bootstrap),
                "confidence": float(confidence),
            }

        matrix = np.asarray(curves, dtype=float)
        alpha = (1.0 - float(confidence)) / 2.0
        lower_q = 100.0 * alpha
        upper_q = 100.0 * (1.0 - alpha)

        lower = np.nanpercentile(matrix, lower_q, axis=0)
        upper = np.nanpercentile(matrix, upper_q, axis=0)

        return {
            "centers": np.asarray(base_centers, dtype=float),
            "ale": np.asarray(base_ale, dtype=float),
            "lower": np.asarray(lower, dtype=float),
            "upper": np.asarray(upper, dtype=float),
            "bootstrap_matrix": matrix,
            "valid_bootstrap": int(matrix.shape[0]),
            "requested_bootstrap": int(n_bootstrap),
            "confidence": float(confidence),
        }


    def detect_ale_threshold(
        centers,
        ale_values,
        lower_ci=None,
        upper_ci=None,
    ):
        """
        Centered ALE 감소 후보 임계점을 자동 탐지합니다.

        탐지 원칙
        --------
        - ALE 최고점 이후 구간을 우선 탐색
        - ALE 기울기가 음수로 전환되고 ALE가 0 이하가 되는 첫 지점을 우선 임계점으로 선택
        - 조건을 만족하지 않으면 최고점 이후 가장 큰 음의 기울기 지점을 후보로 선택
        - 95% CI 상한도 0보다 작으면 '높은 신뢰', CI가 0을 포함하면 '탐색적 후보'로 구분

        즉, 임계점은 생리학적 절대 기준이 아니라 현재 학습자료에서 관측된
        모델 반응의 자동 탐색 후보입니다.
        """
        x = np.asarray(centers, dtype=float)
        y = np.asarray(ale_values, dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if len(x) < 2:
            return None

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # 중복 x 제거
        x_unique, idx_unique = np.unique(x, return_index=True)
        y_unique = y[idx_unique]
        x = x_unique
        y = y_unique

        if len(x) < 2:
            return None

        try:
            slope = np.gradient(y, x)
        except Exception:
            slope = np.gradient(y)

        best_idx = int(np.nanargmax(y))
        worst_idx = int(np.nanargmin(y))

        candidate_idx = None
        reason = ""

        # 최고 ALE 이후, 음의 기울기 + 음의 ALE로 진입하는 첫 지점
        for idx in range(max(1, best_idx + 1), len(x)):
            if (
                np.isfinite(slope[idx])
                and slope[idx] < 0
                and y[idx] <= 0
            ):
                candidate_idx = idx
                reason = "ALE 최고점 이후 음의 기울기와 음의 ALE가 동시에 나타난 첫 지점"
                break

        # 위 조건이 없으면 최고점 이후 가장 급한 하락점
        if candidate_idx is None:
            start_idx = min(best_idx + 1, len(x) - 1)
            candidate_pool = np.arange(start_idx, len(x))
            if len(candidate_pool) > 0:
                candidate_idx = int(
                    candidate_pool[np.nanargmin(slope[candidate_pool])]
                )
                reason = "ALE 최고점 이후 가장 큰 음의 기울기가 나타난 지점"

        if candidate_idx is None:
            candidate_idx = worst_idx
            reason = "ALE 최소점 기반 탐색 후보"

        confidence_label = "탐색적 후보"
        ci_lower_value = np.nan
        ci_upper_value = np.nan

        if lower_ci is not None and upper_ci is not None:
            lower_arr = np.asarray(lower_ci, dtype=float)
            upper_arr = np.asarray(upper_ci, dtype=float)

            # 원 배열 순서와 ALE 중심점이 동일하다는 전제
            if len(lower_arr) == len(centers) and len(upper_arr) == len(centers):
                original_x = np.asarray(centers, dtype=float)
                nearest_original = int(
                    np.nanargmin(np.abs(original_x - x[candidate_idx]))
                )
                ci_lower_value = float(lower_arr[nearest_original])
                ci_upper_value = float(upper_arr[nearest_original])

                if np.isfinite(ci_upper_value) and ci_upper_value < 0:
                    confidence_label = "높은 신뢰(95% CI 상한 < 0)"
                elif (
                    np.isfinite(ci_lower_value)
                    and np.isfinite(ci_upper_value)
                    and ci_lower_value <= 0 <= ci_upper_value
                ):
                    confidence_label = "CI가 0 포함(탐색적 후보)"
                elif np.isfinite(ci_lower_value) and ci_lower_value > 0:
                    confidence_label = "감소 근거 약함(CI 하한 > 0)"

        return {
            "threshold": float(x[candidate_idx]),
            "threshold_ale": float(y[candidate_idx]),
            "threshold_slope": float(slope[candidate_idx]),
            "best": float(x[best_idx]),
            "best_ale": float(y[best_idx]),
            "worst": float(x[worst_idx]),
            "worst_ale": float(y[worst_idx]),
            "candidate_index": int(candidate_idx),
            "reason": reason,
            "confidence": confidence_label,
            "ci_lower": ci_lower_value,
            "ci_upper": ci_upper_value,
        }


    def choose_counterfactual_ale_target(
        centers,
        ale_values,
        threshold_info,
        current_value,
    ):
        """
        ALE 기반 1개 Feature 목표값 후보를 선택합니다.

        기본 목표:
        - ALE가 가장 높은 중심점(best)을 우선 목표로 사용
        - 현재값이 임계점 위험측에 있고 best가 지나치게 멀리 떨어진 경우에도
          학습 데이터 범위 안에서만 목표를 제안
        """
        x = np.asarray(centers, dtype=float)
        y = np.asarray(ale_values, dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if len(x) == 0:
            return None

        best_idx = int(np.nanargmax(y))
        target_value = float(x[best_idx])

        if threshold_info is not None:
            threshold_value = float(threshold_info["threshold"])

            # 현재가 임계점 이상이고 ALE 최고점도 임계점 이상에 있다면
            # 임계점 직전의 가장 우호적인 ALE 중심을 목표 후보로 선택합니다.
            if current_value >= threshold_value and target_value >= threshold_value:
                safe_mask = x < threshold_value
                if np.any(safe_mask):
                    safe_indices = np.where(safe_mask)[0]
                    safe_best = safe_indices[
                        np.nanargmax(y[safe_indices])
                    ]
                    target_value = float(x[safe_best])

        return float(np.clip(target_value, np.nanmin(x), np.nanmax(x)))


    def simulate_counterfactual_target_control(
        model,
        X_reference,
        feature,
        current_value,
        target_value,
    ):
        """
        선택 Feature 하나만 현재값→목표값으로 변경했을 때의 모델 예측 차이를 계산합니다.

        다른 Feature는 기준자료의 중앙값으로 고정합니다.
        따라서 실제 액추에이터 제어량 자체가 아니라,
        '어느 방향으로 환경목표를 이동시키는 것이 모델 예측에 유리한가'를
        확인하는 one-feature counterfactual 시뮬레이션입니다.
        """
        X_ref = align_xai_input(X_reference, features, model)
        if X_ref.empty or feature not in X_ref.columns:
            return None

        baseline = X_ref.median(numeric_only=True).reindex(features).fillna(0.0)
        current_row = pd.DataFrame([baseline.values], columns=features)
        target_row = current_row.copy()

        current_row.loc[0, feature] = float(current_value)
        target_row.loc[0, feature] = float(target_value)

        pred_current = float(
            safe_predict(model, current_row, features)[0]
        )
        pred_target = float(
            safe_predict(model, target_row, features)[0]
        )

        return {
            "feature": feature,
            "current_value": float(current_value),
            "target_value": float(target_value),
            "change": float(target_value - current_value),
            "pred_current": pred_current,
            "pred_target": pred_target,
            "pred_delta": float(pred_target - pred_current),
        }


    def recommend_control_action_from_feature(
        feature_name,
        current_value,
        target_value,
    ):
        """Feature명과 목표 방향으로 스마트온실 제어장치 의사결정 문구를 생성합니다."""
        name = str(feature_name).lower()
        direction = float(target_value) - float(current_value)

        if abs(direction) < 1e-12:
            return "현재값이 제안 목표값과 유사하므로 현 상태 유지 및 재평가를 권장합니다."

        if "온도" in name or "adt" in name or "gdd" in name or "dif" in name:
            if direction < 0:
                return (
                    "온도계열 값을 낮추는 방향: 천창 개방, 환기팬 가동, "
                    "차광스크린 전개, 필요 시 증발냉각을 우선 검토합니다."
                )
            return (
                "온도계열 값을 높이는 방향: 난방 가동, 천창 폐쇄, "
                "보온스크린 활용을 우선 검토합니다."
            )

        if "습도" in name:
            if direction < 0:
                return (
                    "습도를 낮추는 방향: 천창·측창 환기와 순환팬을 우선 검토하고, "
                    "미스트는 억제합니다."
                )
            return (
                "습도를 높이는 방향: 미스트/가습을 검토하되 결로와 병해 위험을 함께 점검합니다."
            )

        if "vpd" in name:
            if direction < 0:
                return (
                    "VPD를 낮추는 방향: 미스트/가습, 차광, 냉각을 검토하고 "
                    "작물 수분상태를 함께 확인합니다."
                )
            return (
                "VPD를 높이는 방향: 환기·제습 및 과도한 미스트 억제를 검토합니다."
            )

        if "co2" in name or "co₂" in name:
            if direction > 0:
                return (
                    "CO₂를 높이는 방향: 환기 상태와 일사조건을 확인한 뒤 CO₂ 시비를 검토합니다."
                )
            return (
                "CO₂를 낮추는 방향: CO₂ 시비량을 줄이거나 환기량을 조절합니다."
            )

        if "일사" in name or "solar" in name or "광" in name:
            if direction < 0:
                return (
                    "광부하를 낮추는 방향: 차광스크린 전개와 환기/냉각을 검토합니다."
                )
            return (
                "광환경을 높이는 방향: 차광 해제 또는 보광을 검토하되 "
                "일사량 GEI 및 작물 엽온을 함께 확인합니다."
            )

        return (
            "선택 Feature의 목표값 방향으로 제어 가능한 액추에이터를 지정하고, "
            "실제 제어 전 안전범위와 작물 생리 기준을 함께 확인합니다."
        )


    def summarize_ale_intervals(bin_centers, ale_vals):
        bc = np.array(bin_centers)
        av = np.array(ale_vals)
        deriv = np.gradient(av, bc)
        thr = 1.5 * (np.std(deriv) + 1e-9)
        steep_idx = np.where(np.abs(deriv) > thr)[0]

        def contiguous_ranges(mask):
            ranges = []
            i = 0
            while i < len(mask):
                if mask[i]:
                    j = i
                    while j < len(mask) and mask[j]:
                        j += 1
                    ranges.append((i, j - 1))
                    i = j
                else:
                    i += 1
            return ranges

        pos_ranges = contiguous_ranges(av > 0)
        neg_ranges = contiguous_ranges(av < 0)
        pos_intervals = [(float(bc[s]), float(bc[t]), float(np.mean(av[s:t + 1]))) for s, t in pos_ranges]
        neg_intervals = [(float(bc[s]), float(bc[t]), float(np.mean(av[s:t + 1]))) for s, t in neg_ranges]
        steep_points = [(int(i), float(bc[i]), float(deriv[i])) for i in steep_idx]
        return {"pos_intervals": pos_intervals, "neg_intervals": neg_intervals, "steep_points": steep_points}

    def find_top_contiguous_interval(x, y, top_frac=0.9, min_width=1):
        thresh = np.quantile(y, top_frac)
        mask = y >= thresh
        segments = []
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                if (j - i) >= min_width:
                    segments.append((i, j - 1))
                i = j
            else:
                i += 1
        if not segments:
            idx = int(np.argmax(y))
            left = max(0, idx - 1)
            right = min(len(x) - 1, idx + 1)
            return x[left], x[right], float(np.mean(y[left:right + 1])), float(np.max(y[left:right + 1]))
        best = None
        best_score = -1e9
        for s, t in segments:
            score = float(np.mean(y[s:t + 1]))
            if score > best_score:
                best_score = score
                best = (s, t)
        s, t = best
        return x[s], x[t], float(np.mean(y[s:t + 1])), float(np.max(y[s:t + 1]))

    def summarize_pdp(model, X, feature, grid_resolution=50):
        col = X[feature]
        x = np.linspace(col.min(), col.max(), grid_resolution)
        y = []
        Xbase = X.copy()
        for val in x:
            Xtmp = Xbase.copy()
            Xtmp[feature] = val
            preds = safe_predict(model, align_xai_input(Xtmp, features, model), features)
            y.append(np.mean(preds))
        x = np.array(x)
        y = np.array(y)
        start, end, mean_y, max_y = find_top_contiguous_interval(x, y, top_frac=0.9)
        return x, y, {"best_interval": (start, end), "mean_val": float(mean_y), "max_val": float(max_y)}

    def summarize_ice_linear_slope(model, X, feature, n_samples=50):
        Xs = X.sample(n=min(n_samples, len(X)), random_state=42)
        xs = np.linspace(X[feature].min(), X[feature].max(), 30)
        slopes = []
        for _, row in Xs.iterrows():
            Xtmp = pd.DataFrame(np.tile(row.values, (len(xs), 1)), columns=X.columns)
            Xtmp[feature] = xs
            preds = safe_predict(model, align_xai_input(Xtmp, features, model), features)
            lr = LinearRegression()
            lr.fit(xs.reshape(-1, 1), preds)
            slopes.append(lr.coef_[0])
        slopes = np.array(slopes)
        return float(np.mean(slopes)), float(np.std(slopes)), len(slopes)

    def compute_lime_local_explanation(
        model,
        X_reference,
        sample_index,
        n_samples=600,
        random_state=42,
    ):
        """
        LIME 로컬 설명.
        - lime 패키지가 있으면 LimeTabularExplainer 사용
        - 없으면 동일 개념의 거리 가중 로컬 선형 surrogate fallback 사용
        """
        X_ref = align_xai_input(
            X_reference,
            features,
            model,
        ).reset_index(drop=True)

        if X_ref.empty:
            return None

        sample_index = int(
            np.clip(
                int(sample_index),
                0,
                len(X_ref) - 1,
            )
        )
        sample = X_ref.iloc[sample_index].copy()
        feature_names = list(X_ref.columns)

        # --------------------------------------------------
        # 1) 공식 lime 패키지 사용
        # --------------------------------------------------
        if LimeTabularExplainer is not None:
            try:
                explainer_lime = LimeTabularExplainer(
                    training_data=X_ref.to_numpy(dtype=float),
                    feature_names=feature_names,
                    mode="regression",
                    discretize_continuous=True,
                    random_state=int(random_state),
                )

                def _lime_predict(arr):
                    arr_df = pd.DataFrame(
                        np.asarray(arr, dtype=float),
                        columns=feature_names,
                    )
                    return safe_predict(
                        model,
                        arr_df,
                        feature_names,
                    )

                exp = explainer_lime.explain_instance(
                    sample.to_numpy(dtype=float),
                    _lime_predict,
                    num_features=len(feature_names),
                    num_samples=int(n_samples),
                )

                map_items = exp.as_map().get(1, [])
                if not map_items and exp.as_map():
                    map_items = list(exp.as_map().values())[0]

                weight_map = {
                    feature_names[int(idx)]: float(weight)
                    for idx, weight in map_items
                    if int(idx) < len(feature_names)
                }

                result_rows = []
                for fname in feature_names:
                    result_rows.append({
                        "Feature": fname,
                        "Sample Value": float(sample[fname]),
                        "Local Weight": float(weight_map.get(fname, 0.0)),
                        "|Local Weight|": abs(float(weight_map.get(fname, 0.0))),
                    })

                result_df = (
                    pd.DataFrame(result_rows)
                    .sort_values("|Local Weight|", ascending=False)
                    .reset_index(drop=True)
                )

                pred_value = float(
                    safe_predict(
                        model,
                        pd.DataFrame([sample.values], columns=feature_names),
                        feature_names,
                    )[0]
                )

                return {
                    "method": "LIME",
                    "sample_index": sample_index,
                    "prediction": pred_value,
                    "fidelity_r2": float(getattr(exp, "score", np.nan)),
                    "local_prediction": float(
                        np.asarray(
                            getattr(exp, "local_pred", [np.nan])
                        ).reshape(-1)[0]
                    ),
                    "table": result_df,
                }
            except Exception:
                pass

        # --------------------------------------------------
        # 2) dependency-free LIME-compatible fallback
        # --------------------------------------------------
        rng = np.random.RandomState(int(random_state))
        X_np = X_ref.to_numpy(dtype=float)

        means = np.nanmean(X_np, axis=0)
        stds = np.nanstd(X_np, axis=0)
        stds = np.where(
            np.isfinite(stds) & (stds > 1e-12),
            stds,
            1.0,
        )

        mins = np.nanmin(X_np, axis=0)
        maxs = np.nanmax(X_np, axis=0)

        perturb = rng.normal(
            loc=sample.to_numpy(dtype=float),
            scale=stds * 0.35,
            size=(int(n_samples), len(feature_names)),
        )
        perturb = np.clip(
            perturb,
            mins,
            maxs,
        )
        perturb[0, :] = sample.to_numpy(dtype=float)

        perturb_df = pd.DataFrame(
            perturb,
            columns=feature_names,
        )
        y_local = safe_predict(
            model,
            perturb_df,
            feature_names,
        )

        z = (
            perturb
            - sample.to_numpy(dtype=float)
        ) / stds
        distances = np.sqrt(
            np.sum(z ** 2, axis=1)
        )
        kernel_width = max(
            0.75 * np.sqrt(len(feature_names)),
            1e-6,
        )
        weights = np.exp(
            -(distances ** 2)
            / (kernel_width ** 2)
        )

        local_model = LinearRegression()
        local_model.fit(
            perturb,
            y_local,
            sample_weight=weights,
        )

        local_pred = local_model.predict(
            sample.to_numpy(dtype=float).reshape(1, -1)
        )[0]

        try:
            fidelity = float(
                local_model.score(
                    perturb,
                    y_local,
                    sample_weight=weights,
                )
            )
        except Exception:
            fidelity = np.nan

        centered_sample = (
            sample.to_numpy(dtype=float) - means
        )
        contributions = (
            np.asarray(local_model.coef_, dtype=float)
            * centered_sample
        )

        result_df = pd.DataFrame({
            "Feature": feature_names,
            "Sample Value": sample.to_numpy(dtype=float),
            "Local Weight": np.asarray(
                local_model.coef_,
                dtype=float,
            ),
            "Local Contribution": contributions,
            "|Local Weight|": np.abs(
                np.asarray(
                    local_model.coef_,
                    dtype=float,
                )
            ),
        }).sort_values(
            "|Local Weight|",
            ascending=False,
        ).reset_index(drop=True)

        pred_value = float(
            safe_predict(
                model,
                pd.DataFrame(
                    [sample.values],
                    columns=feature_names,
                ),
                feature_names,
            )[0]
        )

        return {
            "method": "LIME-compatible local surrogate",
            "sample_index": sample_index,
            "prediction": pred_value,
            "fidelity_r2": fidelity,
            "local_prediction": float(local_pred),
            "table": result_df,
        }


    def explain_lime_result(lime_result, target_name):
        if not lime_result or lime_result["table"].empty:
            return "LIME 로컬 설명을 생성할 수 없습니다."

        table = lime_result["table"]
        top = table.iloc[0]
        positive = table[table["Local Weight"] > 0].head(3)
        negative = table[table["Local Weight"] < 0].head(3)

        pos_text = ", ".join(
            [
                f"{pretty_time_text(r['Feature'])}({r['Local Weight']:+.4f})"
                for _, r in positive.iterrows()
            ]
        ) or "뚜렷한 양의 로컬 가중치 없음"

        neg_text = ", ".join(
            [
                f"{pretty_time_text(r['Feature'])}({r['Local Weight']:+.4f})"
                for _, r in negative.iterrows()
            ]
        ) or "뚜렷한 음의 로컬 가중치 없음"

        fidelity = lime_result["fidelity_r2"]
        fidelity_text = (
            f"{fidelity:.3f}"
            if np.isfinite(fidelity)
            else "N/A"
        )

        return (
            f"선택 샘플 #{lime_result['sample_index']}에서 {target_name} 예측값은 "
            f"{lime_result['prediction']:.4f}입니다. "
            f"로컬 surrogate의 설명 적합도 R²는 {fidelity_text}입니다. "
            f"가장 큰 로컬 영향 Feature는 '{pretty_time_text(top['Feature'])}'이고 "
            f"Local Weight={top['Local Weight']:+.5f}입니다. "
            f"양의 방향 주요 Feature: {pos_text}. "
            f"음의 방향 주요 Feature: {neg_text}. "
            "LIME은 특정 한 샘플 주변의 국소 설명이므로 전역 중요도로 일반화하지 않고 "
            "SHAP, ICE/PDP, Centered ALE 결과와 함께 비교해야 합니다."
        )

    # ICE + PDP는 전체 폭으로 단독 배치
    ice_pdp_container = st.container()

    with ice_pdp_container:

        st.markdown("**ICE + PDP 통합 그래프 & 최적 구간**")

        try:

            fig_mix, ax_mix = plt.subplots(figsize=(6, 4))

            # -------------------------------
            # ICE
            # -------------------------------
            X_test_ice = align_xai_input(X_test, features, model)
            Xs = X_test_ice.sample(
                n=min(n_samples, len(X_test_ice)),
                random_state=42
            )

            xs = np.linspace(
                X_test_ice[ice_feature].min(),
                X_test_ice[ice_feature].max(),
                50
            )

            for _, row in Xs.iterrows():

                Xtmp = pd.DataFrame(
                    np.tile(row.values, (len(xs), 1)),
                    columns=X_test_ice.columns
                )

                Xtmp[ice_feature] = xs

                preds = safe_predict(model, align_xai_input(Xtmp, features, model), features)

                ax_mix.plot(
                    xs,
                    preds,
                    alpha=0.15
                )

            # -------------------------------
            # PDP
            # -------------------------------
            xvals, yvals, pdp_summary = summarize_pdp(
                model,
                X_test_ice,
                ice_feature,
                grid_resolution=50
            )

            ax_mix.plot(
                xvals,
                yvals,
                color="red",
                linewidth=3,
                label="PDP"
            )

            ax_mix.set_title(f"ICE + PDP: {pretty_time_text(ice_feature)}")
            ax_mix.set_xlabel(pretty_time_text(ice_feature))
            ax_mix.set_ylabel("Predicted")
            ax_mix.legend()

            ice_graph_col, ice_explain_col = st.columns([1.05, 0.95], gap="large")
            with ice_graph_col:
                display_matplotlib(fig_mix)
            plt.close(fig_mix)

            # 최적 구간
            start, end = pdp_summary["best_interval"]

            st.write(
                f"예측대상({report_target})의 최적(예측이 큰) 구간: "
                f"{start:.3f} ~ {end:.3f}"
            )

            st.write(
                f"구간 평균 예측값: "
                f"{pdp_summary['mean_val']:.3f}"
            )

            st.write(
                f"구간 최대값: "
                f"{pdp_summary['max_val']:.3f}"
            )

            ice_mean_slope, ice_std_slope, cnt = summarize_ice_linear_slope(
                model,
                X_test_ice,
                ice_feature,
                n_samples=min(n_samples, len(X_test))
            )

            st.write(
                f"ICE 평균 기울기: "
                f"{ice_mean_slope:.4f} ± {ice_std_slope:.4f}"
            )

            ice_desc, pdp_desc, combined_desc = explain_ice_pdp_result(
                ice_feature,
                ice_mean_slope,
                ice_std_slope,
                pdp_summary,
                report_target
            )

            with ice_explain_col:
                st.markdown("**ICE 그래프 해석**")
                st.markdown(ice_desc, unsafe_allow_html=True)
                st.markdown("**PDP 그래프 해석**")
                st.markdown(pdp_desc, unsafe_allow_html=True)

            st.markdown("**ICE + PDP 전체 그래프 결과 설명**")
            st.markdown(combined_desc, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"ICE + PDP 처리 오류: {e}")


    st.markdown("---")
    render_stylish_section(
        "④ 🎯 개별 예측 설명 · LIME + SHAP Waterfall + SHAP Force Plot",
        (
            "동일한 평가 샘플을 세 가지 local XAI 관점으로 비교합니다. "
            "LIME은 샘플 주변의 로컬 surrogate를, Waterfall은 Base value에서 개별 Feature 기여의 누적과정을, "
            "Force Plot은 양·음의 SHAP 기여가 예측값을 어느 방향으로 밀었는지 보여줍니다."
        ),
        kicker="LOCAL XAI EXPLANATION",
    )

    X_test_local_common = align_xai_input(
        X_test,
        features,
        model,
    ).reset_index(drop=True)

    local_sample_default = 0
    local_sample_index_common = st.slider(
        "LIME / Waterfall / Force Plot 공통 샘플 선택",
        min_value=0,
        max_value=max(
            0,
            len(X_test_local_common) - 1,
        ),
        value=local_sample_default,
        step=1,
        key="local_xai_common_sample_v270",
    ) if not X_test_local_common.empty else 0

    local_col1, local_col2, local_col3 = st.columns(
        3,
        gap="large",
    )

    with local_col1:
        st.markdown("**LIME (그래프 + 정량적 결과)**")

        try:
            X_test_lime = align_xai_input(
                X_test,
                features,
                model,
            ).reset_index(drop=True)

            if X_test_lime.empty:
                st.info("LIME 분석에 사용할 평가 데이터가 없습니다.")
            else:
                lime_sample_index = int(
                    np.clip(
                        local_sample_index_common,
                        0,
                        max(0, len(X_test_lime) - 1),
                    )
                )

                lime_num_samples = st.select_slider(
                    "LIME perturbation 수",
                    options=[200, 400, 600, 800, 1000],
                    value=600,
                    key="lime_num_samples_v264",
                )

                lime_result = compute_lime_local_explanation(
                    model=model,
                    X_reference=X_test_lime,
                    sample_index=lime_sample_index,
                    n_samples=int(lime_num_samples),
                    random_state=42,
                )

                if lime_result is None:
                    st.warning("LIME 로컬 설명을 계산하지 못했습니다.")
                else:
                    lime_table = lime_result["table"].copy()
                    lime_top = lime_table.head(
                        min(12, len(lime_table))
                    ).sort_values(
                        "Local Weight",
                        ascending=True,
                    )

                    fig_lime = go.Figure(
                        go.Bar(
                            x=lime_top["Local Weight"],
                            y=[
                                pretty_time_text(v)
                                for v in lime_top["Feature"]
                            ],
                            orientation="h",
                            text=[
                                f"{v:+.4f}"
                                for v in lime_top["Local Weight"]
                            ],
                            textposition="outside",
                            hovertemplate=(
                                "%{y}<br>"
                                "Local Weight=%{x:.5f}"
                                "<extra></extra>"
                            ),
                        )
                    )
                    fig_lime.add_vline(
                        x=0,
                        line_dash="dash",
                        line_color="#64748b",
                    )
                    fig_lime.update_layout(
                        title=(
                            f"LIME Local Explanation · "
                            f"Sample #{lime_result['sample_index']}"
                        ),
                        xaxis_title="Local Weight",
                        yaxis_title="Feature",
                        height=430,
                        template=plotly_template,
                        margin=dict(
                            l=55,
                            r=35,
                            t=70,
                            b=45,
                        ),
                    )
                    st.plotly_chart(
                        fig_lime,
                        use_container_width=True,
                        key="lime_local_bar_v264",
                    )

                    lime_m1, lime_m2, lime_m3 = st.columns(3)
                    with lime_m1:
                        st.metric(
                            "모델 예측",
                            f"{lime_result['prediction']:.3f}",
                        )
                    with lime_m2:
                        st.metric(
                            "Local surrogate 예측",
                            (
                                f"{lime_result['local_prediction']:.3f}"
                                if np.isfinite(
                                    lime_result["local_prediction"]
                                )
                                else "N/A"
                            ),
                        )
                    with lime_m3:
                        st.metric(
                            "Local Fidelity R²",
                            (
                                f"{lime_result['fidelity_r2']:.3f}"
                                if np.isfinite(
                                    lime_result["fidelity_r2"]
                                )
                                else "N/A"
                            ),
                        )

                    st.caption(
                        "사용 방식: "
                        + lime_result["method"]
                        + (
                            " · `pip install lime` 설치 시 공식 LIME 패키지를 우선 사용합니다."
                            if LimeTabularExplainer is None
                            else ""
                        )
                    )

                    with lime_result_col:
                        st.markdown("**정량적 결과**")
                        lime_display_cols = [
                            c for c in [
                                "Feature", "Sample Value", "Local Weight",
                                "Local Contribution", "|Local Weight|",
                            ]
                            if c in lime_table.columns
                        ]
                        st.dataframe(
                            lime_table[lime_display_cols].head(15).round(6),
                            use_container_width=True, hide_index=True, height=390,
                        )

                    st.markdown("**자동 해석**")
                    st.info(
                        explain_lime_result(
                            lime_result,
                            report_target,
                        )
                    )

        except Exception as e:
            st.error(f"LIME 처리 오류: {e}")




    with local_col2:
        st.markdown("**SHAP Waterfall (그래프 + 정량적 결과)**")

        if (
            model_choice == "GaussianNB"
            or shap_values is None
            or shap_df is None
            or shap_df.empty
        ):
            st.info(
                "SHAP Summary가 정상 계산되면 Waterfall Plot이 활성화됩니다."
            )
        else:
            try:
                waterfall_result = get_local_shap_sample(
                    shap_values_input=shap_values,
                    X_input=X_test_shap,
                    feature_names=list(shap_df["Feature"]),
                    sample_index=local_sample_index_common,
                )

                if waterfall_result is None:
                    st.info(
                        "Waterfall Plot을 생성할 샘플이 없습니다."
                    )
                else:
                    waterfall_exp = shap.Explanation(
                        values=waterfall_result[
                            "shap_values"
                        ],
                        base_values=waterfall_result[
                            "base_value"
                        ],
                        data=waterfall_result[
                            "sample_row"
                        ].values,
                        feature_names=waterfall_result[
                            "feature_names"
                        ],
                    )

                    plt.figure(
                        figsize=(6.2, 4.7)
                    )
                    shap.plots.waterfall(
                        waterfall_exp,
                        max_display=min(
                            12,
                            len(
                                waterfall_result[
                                    "feature_names"
                                ]
                            ),
                        ),
                        show=False,
                    )
                    fig_waterfall = plt.gcf()
                    display_matplotlib(
                        fig_waterfall
                    )
                    plt.close(
                        fig_waterfall
                    )

                    wf_m1, wf_m2 = st.columns(2)
                    with wf_m1:
                        st.metric(
                            "Base value",
                            f"{waterfall_result['base_value']:.3f}",
                        )
                    with wf_m2:
                        st.metric(
                            f"복원 예측 {report_target}",
                            (
                                f"{waterfall_result['reconstructed_prediction']:.3f}"
                            ),
                        )

                    st.markdown(
                        "**정량적 결과**"
                    )
                    st.dataframe(
                        waterfall_result[
                            "table"
                        ][
                            [
                                "Feature",
                                "Feature Value",
                                "SHAP Value",
                                "|SHAP|",
                                "Direction",
                            ]
                        ]
                        .head(12)
                        .round(6),
                        use_container_width=True,
                        hide_index=True,
                        height=390,
                    )

                    st.markdown(
                        "**자동 해석**"
                    )
                    st.info(
                        explain_local_shap_result(
                            waterfall_result,
                            report_target,
                            "SHAP Waterfall",
                        )
                    )

            except Exception as e:
                st.error(
                    f"SHAP Waterfall 처리 오류: {e}"
                )

    with local_col3:
        st.markdown("**SHAP Force Plot (그래프 + 정량적 결과)**")

        if (
            model_choice == "GaussianNB"
            or shap_values is None
            or shap_df is None
            or shap_df.empty
        ):
            st.info(
                "SHAP Summary가 정상 계산되면 Force Plot이 활성화됩니다."
            )
        else:
            try:
                force_result = get_local_shap_sample(
                    shap_values_input=shap_values,
                    X_input=X_test_shap,
                    feature_names=list(shap_df["Feature"]),
                    sample_index=local_sample_index_common,
                )

                if force_result is None:
                    st.info(
                        "Force Plot을 생성할 샘플이 없습니다."
                    )
                else:
                    plt.figure(
                        figsize=(7.0, 2.7)
                    )
                    shap.force_plot(
                        force_result[
                            "base_value"
                        ],
                        force_result[
                            "shap_values"
                        ],
                        force_result[
                            "sample_row"
                        ].values,
                        feature_names=force_result[
                            "feature_names"
                        ],
                        matplotlib=True,
                        show=False,
                    )
                    fig_force = plt.gcf()
                    display_matplotlib(
                        fig_force
                    )
                    plt.close(
                        fig_force
                    )

                    positive_sum = float(
                        force_result[
                            "table"
                        ].loc[
                            force_result[
                                "table"
                            ][
                                "SHAP Value"
                            ] > 0,
                            "SHAP Value",
                        ].sum()
                    )
                    negative_sum = float(
                        force_result[
                            "table"
                        ].loc[
                            force_result[
                                "table"
                            ][
                                "SHAP Value"
                            ] < 0,
                            "SHAP Value",
                        ].sum()
                    )

                    force_m1, force_m2 = st.columns(2)
                    with force_m1:
                        st.metric(
                            "양의 SHAP 합",
                            f"{positive_sum:+.3f}",
                        )
                    with force_m2:
                        st.metric(
                            "음의 SHAP 합",
                            f"{negative_sum:+.3f}",
                        )

                    st.markdown(
                        "**정량적 결과**"
                    )
                    force_display = (
                        force_result[
                            "table"
                        ].copy()
                    )
                    st.dataframe(
                        force_display[
                            [
                                "Feature",
                                "Feature Value",
                                "SHAP Value",
                                "Direction",
                            ]
                        ]
                        .head(12)
                        .round(6),
                        use_container_width=True,
                        hide_index=True,
                        height=390,
                    )

                    st.markdown(
                        "**자동 해석**"
                    )
                    st.info(
                        explain_local_shap_result(
                            force_result,
                            report_target,
                            "SHAP Force Plot",
                        )
                    )

            except Exception as e:
                st.error(
                    f"SHAP Force Plot 처리 오류: {e}"
                )

    # ALE 확장 결과를 종합리포트 전까지 안전하게 공유하기 위한 기본값
    ale_bootstrap_result = None
    ale_threshold_info = None
    ale_cf_result = None
    ale_control_recommendation = None

    render_stylish_section(
        "⑤ 📉 ALE 분석 · Centered ALE",
        "선택 Feature의 국소 누적효과를 계산하고, ALE bins 수에 따른 기본 영향곡선과 최적 구간을 확인합니다.",
        kicker="ACCUMULATED LOCAL EFFECTS",
    )
    st.caption(f"현재 ALE bins 수: {ale_bins} · 위 슬라이더에서 4~30 범위로 조정할 수 있습니다.")

    # 기본 Centered ALE는 전체 폭으로 단독 배치
    ale_basic_container = st.container()

    with ale_basic_container:
        st.markdown("**Centered ALE · 기본 영향곡선 + 최적 구간 리포트**")

        try:
            X_test_ale_basic = align_xai_input(
                X_test,
                features,
                model,
            ).reset_index(drop=True)

            bin_centers, ale_vals = compute_centered_ale(
                model,
                X_test_ale_basic,
                ice_feature,
                bins=ale_bins,
            )

            ale_summary = summarize_ale_intervals(
                bin_centers,
                ale_vals,
            )

            fig_ale_basic = go.Figure()
            fig_ale_basic.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=ale_vals,
                    mode="lines+markers",
                    line=dict(
                        width=4,
                        color="#2563eb",
                    ),
                    marker=dict(size=8),
                    name="Centered ALE",
                )
            )
            fig_ale_basic.add_hline(
                y=0,
                line_dash="dash",
                line_color="#64748b",
            )

            if len(bin_centers) > 1:
                best_basic_idx = int(
                    np.nanargmax(
                        np.asarray(
                            ale_vals,
                            dtype=float,
                        )
                    )
                )
                best_basic_value = float(
                    np.asarray(
                        bin_centers,
                        dtype=float,
                    )[best_basic_idx]
                )
                fig_ale_basic.add_vline(
                    x=best_basic_value,
                    line_dash="dot",
                    line_color="#059669",
                    annotation_text=(
                        f"우호 중심 {best_basic_value:.3f}"
                    ),
                    annotation_position="top left",
                )

            fig_ale_basic.update_layout(
                title=(
                    f"Centered ALE · "
                    f"{pretty_time_text(ice_feature)}"
                ),
                xaxis_title=pretty_time_text(
                    ice_feature
                ),
                yaxis_title="Centered ALE",
                height=430,
                template=plotly_template,
                margin=dict(
                    l=50,
                    r=25,
                    t=70,
                    b=48,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.82)",
            )
            st.plotly_chart(
                fig_ale_basic,
                use_container_width=True,
                key="centered_ale_basic_v264",
            )

            if ale_summary["pos_intervals"]:
                pos_best = max(
                    ale_summary["pos_intervals"],
                    key=lambda item: item[2],
                )
                st.success(
                    f"우호 구간 후보: {pos_best[0]:.3f} ~ {pos_best[1]:.3f} "
                    f"· 평균 ALE {pos_best[2]:.4f}"
                )
            else:
                st.info(
                    "양의 ALE가 연속적으로 나타나는 우호 구간을 찾지 못했습니다."
                )

            if ale_summary["neg_intervals"]:
                neg_worst = min(
                    ale_summary["neg_intervals"],
                    key=lambda item: item[2],
                )
                st.warning(
                    f"불리 구간 후보: {neg_worst[0]:.3f} ~ {neg_worst[1]:.3f} "
                    f"· 평균 ALE {neg_worst[2]:.4f}"
                )

            st.markdown("**정량적 결과**")
            ale_basic_table = pd.DataFrame({
                "Feature 중심": np.asarray(
                    bin_centers,
                    dtype=float,
                ),
                "Centered ALE": np.asarray(
                    ale_vals,
                    dtype=float,
                ),
            })
            st.dataframe(
                ale_basic_table.round(6),
                use_container_width=True,
                hide_index=True,
                height=330,
            )

            ale_pos = ale_basic_table[
                ale_basic_table["Centered ALE"] > 0
            ].copy()
            ale_neg = ale_basic_table[
                ale_basic_table["Centered ALE"] < 0
            ].copy()

            fav_col, unfav_col = st.columns(2, gap="large")
            with fav_col:
                st.markdown("**우호 구간 후보**")
                if not ale_pos.empty:
                    fav = ale_pos.loc[ale_pos["Centered ALE"].idxmax()]
                    st.success(
                        f"Feature 중심 약 {fav['Feature 중심']:.4f} · "
                        f"Centered ALE {fav['Centered ALE']:+.4f}"
                    )
                else:
                    st.info("양의 ALE 우호 후보가 탐지되지 않았습니다.")
            with unfav_col:
                st.markdown("**불리 구간 후보**")
                if not ale_neg.empty:
                    unfav = ale_neg.loc[ale_neg["Centered ALE"].idxmin()]
                    st.warning(
                        f"Feature 중심 약 {unfav['Feature 중심']:.4f} · "
                        f"Centered ALE {unfav['Centered ALE']:+.4f}"
                    )
                else:
                    st.info("음의 ALE 불리 후보가 탐지되지 않았습니다.")

            st.markdown("**자동 해석**")
            st.markdown(
                explain_centered_ale_result(
                    ice_feature,
                    bin_centers,
                    ale_vals,
                    ale_summary,
                    report_target,
                ),
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(
                f"기본 Centered ALE 처리 오류: {e}"
            )

    # Bootstrap CI / Threshold Detection ALE는 기본 ALE 아래 전체 폭으로 단독 배치
    st.markdown(
        "<hr style='border:none;border-top:2px solid #94a3b8;margin:26px 0 22px 0;'>",
        unsafe_allow_html=True,
    )
    ale_extended_container = st.container()

    with ale_extended_container:
        st.markdown("### Centered ALE → 1D Bootstrap 95% CI → Threshold Detection")

        try:
            X_test_ale = align_xai_input(
                X_test,
                features,
                model,
            ).reset_index(drop=True)

            bin_centers, ale_vals = compute_centered_ale(
                model,
                X_test_ale,
                ice_feature,
                bins=ale_bins,
            )

            # Bootstrap 반복횟수는 연산량과 안정성의 균형을 위해 사용자 선택
            bootstrap_repeats = st.select_slider(
                "1D Bootstrap 반복횟수",
                options=[100, 200, 300, 500, 1000],
                value=200,
                key="ale_bootstrap_repeats",
                help=(
                    "학습자료를 복원추출하여 모델을 다시 학습하고 ALE를 재계산합니다. "
                    "반복횟수가 많을수록 CI가 안정적이지만 계산시간이 증가합니다."
                ),
            )

            with st.spinner(
                f"Centered ALE Bootstrap {bootstrap_repeats}회와 95% CI를 계산하고 있습니다..."
            ):
                ale_bootstrap_result = bootstrap_centered_ale_ci(
                    model_choice=model_choice,
                    X_train_input=X_train,
                    y_train_input=y_train,
                    X_eval_input=X_test_ale,
                    feature=ice_feature,
                    bins=ale_bins,
                    n_bootstrap=int(bootstrap_repeats),
                    confidence=0.95,
                    random_state=42,
                )

            ale_centers = np.asarray(
                ale_bootstrap_result["centers"],
                dtype=float,
            )
            ale_curve = np.asarray(
                ale_bootstrap_result["ale"],
                dtype=float,
            )
            ale_lower = np.asarray(
                ale_bootstrap_result["lower"],
                dtype=float,
            )
            ale_upper = np.asarray(
                ale_bootstrap_result["upper"],
                dtype=float,
            )

            ale_threshold_info = detect_ale_threshold(
                centers=ale_centers,
                ale_values=ale_curve,
                lower_ci=ale_lower,
                upper_ci=ale_upper,
            )

            # ---------------------------------------------
            # Plotly: ALE + 95% CI + 자동 임계점
            # ---------------------------------------------
            fig_ale_ci = go.Figure()

            if len(ale_centers) > 1:
                # 상한 → 하한 순서로 그려 fill='tonexty'
                fig_ale_ci.add_trace(
                    go.Scatter(
                        x=ale_centers,
                        y=ale_upper,
                        mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                        name="95% CI upper",
                    )
                )
                fig_ale_ci.add_trace(
                    go.Scatter(
                        x=ale_centers,
                        y=ale_lower,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(37,99,235,0.16)",
                        hovertemplate=(
                            "Feature=%{x:.4f}<br>"
                            "95% CI lower=%{y:.4f}<extra></extra>"
                        ),
                        name="95% Bootstrap CI",
                    )
                )
                fig_ale_ci.add_trace(
                    go.Scatter(
                        x=ale_centers,
                        y=ale_curve,
                        mode="lines+markers",
                        line=dict(width=4, color="#1d4ed8"),
                        marker=dict(size=8),
                        name="Centered ALE",
                        hovertemplate=(
                            "Feature=%{x:.4f}<br>"
                            "ALE=%{y:.4f}<extra></extra>"
                        ),
                    )
                )
            elif len(ale_centers) == 1:
                fig_ale_ci.add_trace(
                    go.Scatter(
                        x=ale_centers,
                        y=ale_curve,
                        mode="markers",
                        marker=dict(size=10),
                        name="Centered ALE",
                    )
                )

            fig_ale_ci.add_hline(
                y=0,
                line_dash="dash",
                line_color="#64748b",
                annotation_text="ALE = 0",
                annotation_position="bottom right",
            )

            if ale_threshold_info is not None:
                fig_ale_ci.add_vline(
                    x=ale_threshold_info["threshold"],
                    line_dash="dot",
                    line_width=3,
                    line_color="#dc2626",
                    annotation_text=(
                        f"자동 임계점 {ale_threshold_info['threshold']:.3f}"
                    ),
                    annotation_position="top right",
                )

                fig_ale_ci.add_vline(
                    x=ale_threshold_info["best"],
                    line_dash="dash",
                    line_width=2,
                    line_color="#059669",
                    annotation_text=(
                        f"우호 중심 {ale_threshold_info['best']:.3f}"
                    ),
                    annotation_position="top left",
                )

            fig_ale_ci.update_layout(
                title=(
                    f"Centered ALE + 95% Bootstrap CI · "
                    f"{pretty_time_text(ice_feature)}"
                ),
                xaxis_title=pretty_time_text(ice_feature),
                yaxis_title="Centered ALE",
                height=430,
                template=plotly_template,
                margin=dict(l=50, r=25, t=70, b=48),
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.82)",
                legend=dict(
                    orientation="h",
                    y=1.14,
                    x=1,
                    xanchor="right",
                ),
            )
            st.plotly_chart(
                fig_ale_ci,
                use_container_width=True,
                key="centered_ale_bootstrap_ci_plot",
            )

            # 기존 ALE 구간 분석 유지
            ale_summary = summarize_ale_intervals(
                bin_centers,
                ale_vals,
            )

            # ---------------------------------------------
            # Bootstrap / Threshold 정량 결과
            # ---------------------------------------------
            valid_boot = int(
                ale_bootstrap_result["valid_bootstrap"]
            )
            requested_boot = int(
                ale_bootstrap_result["requested_bootstrap"]
            )

            metric_ci1, metric_ci2, metric_ci3 = st.columns(3)

            with metric_ci1:
                st.metric(
                    "유효 Bootstrap",
                    f"{valid_boot}/{requested_boot}",
                )

            with metric_ci2:
                if ale_threshold_info is not None:
                    st.metric(
                        "자동 감소 임계점",
                        f"{ale_threshold_info['threshold']:.3f}",
                    )
                else:
                    st.metric(
                        "자동 감소 임계점",
                        "탐지 불가",
                    )

            with metric_ci3:
                if ale_threshold_info is not None:
                    st.metric(
                        "우호 ALE 중심",
                        f"{ale_threshold_info['best']:.3f}",
                    )
                else:
                    st.metric(
                        "우호 ALE 중심",
                        "N/A",
                    )

            ci_table = pd.DataFrame({
                "Feature 값": ale_centers,
                "Centered ALE": ale_curve,
                "95% CI Lower": ale_lower,
                "95% CI Upper": ale_upper,
                "CI 폭": ale_upper - ale_lower,
            })

            with st.expander(
                "1D Bootstrap 95% CI 정량표 보기",
                expanded=False,
            ):
                st.dataframe(
                    ci_table.round(5),
                    use_container_width=True,
                    hide_index=True,
                )

            if ale_threshold_info is not None:
                st.markdown(
                    f"""
                    <div class="xai-insight-card">
                        <b>Threshold Detection 결과</b><br>
                        • 감소 후보 임계점: <b>{ale_threshold_info['threshold']:.4f}</b><br>
                        • 임계점 ALE: <b>{ale_threshold_info['threshold_ale']:.4f}</b><br>
                        • 임계점 기울기: <b>{ale_threshold_info['threshold_slope']:.4f}</b><br>
                        • 우호적인 ALE 중심: <b>{ale_threshold_info['best']:.4f}</b><br>
                        • 가장 불리한 ALE 중심: <b>{ale_threshold_info['worst']:.4f}</b><br>
                        • Bootstrap 판정: <b>{ale_threshold_info['confidence']}</b><br>
                        • 탐지근거: {ale_threshold_info['reason']}<br><br>
                        이 값은 작물의 절대 생리 임계값이 아니라
                        <b>현재 데이터와 학습모델에서 자동 탐색된 감소 후보 임계점</b>입니다.
                        특히 95% CI가 0을 포함하면 확정적 임계점이 아니라 탐색적 후보로 해석해야 합니다.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if ale_summary["pos_intervals"]:
                st.write(
                    "모델이 우호적으로 보는 구간(양의 ALE):"
                )
                for a, b, mv in ale_summary["pos_intervals"]:
                    st.write(
                        f"• {a:.2f} ~ {b:.2f} "
                        f"(평균 ALE: {mv:.3f})"
                    )

            if ale_summary["neg_intervals"]:
                st.write(
                    "모델이 불리하게 보는 구간(음의 ALE):"
                )
                for a, b, mv in ale_summary["neg_intervals"]:
                    st.write(
                        f"• {a:.2f} ~ {b:.2f} "
                        f"(평균 ALE: {mv:.3f})"
                    )

            st.markdown(
                "**Centered ALE + Bootstrap CI 결과 설명**"
            )
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#ffffff,#eef5ff);
                box-shadow:0 6px 20px rgba(0,0,0,0.05);
                padding:14px;border-radius:10px;line-height:1.8;font-size:16px">
                {explain_centered_ale_result(
                    ice_feature,
                    bin_centers,
                    ale_vals,
                    ale_summary,
                    report_target
                )}
                <br><br>
                <b>95% Bootstrap CI 해석:</b>
                파란 음영은 학습표본을 복원추출하고 모델을 다시 학습했을 때
                ALE 곡선이 변동할 수 있는 범위를 나타냅니다.
                음영이 좁으면 해당 구간의 ALE 반응이 상대적으로 안정적이고,
                넓으면 표본 수 부족·환경변수 상관·모델 불확실성의 영향을 더 크게 받을 수 있습니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(
                f"ALE Bootstrap/Threshold 처리 오류: {e}"
            )

    # ---------------------------------------------------------
    # Counterfactual Target Control
    # ---------------------------------------------------------
    st.markdown(
        "### 🎯 Centered ALE 기반 Counterfactual Target Control"
    )

    if (
        ale_bootstrap_result is not None
        and ale_threshold_info is not None
        and len(ale_bootstrap_result.get("centers", [])) > 0
    ):
        try:
            ale_centers_cf = np.asarray(
                ale_bootstrap_result["centers"],
                dtype=float,
            )
            ale_curve_cf = np.asarray(
                ale_bootstrap_result["ale"],
                dtype=float,
            )

            default_current = float(
                pd.to_numeric(
                    X_test_ale[ice_feature],
                    errors="coerce",
                ).dropna().median()
            )

            feature_min = float(
                np.nanmin(ale_centers_cf)
            )
            feature_max = float(
                np.nanmax(ale_centers_cf)
            )

            if not np.isfinite(default_current):
                default_current = float(
                    np.nanmedian(ale_centers_cf)
                )

            default_current = float(
                np.clip(
                    default_current,
                    feature_min,
                    feature_max,
                )
            )

            current_feature_value = st.number_input(
                f"현재 {pretty_time_text(ice_feature)} 값",
                min_value=float(feature_min),
                max_value=float(feature_max),
                value=float(default_current),
                key="ale_cf_current_value",
                help=(
                    "초기값은 평가자료 중앙값입니다. "
                    "실시간 제어에 사용할 때는 현재 센서/누적환경 값을 입력하세요."
                ),
            )

            auto_target = choose_counterfactual_ale_target(
                centers=ale_centers_cf,
                ale_values=ale_curve_cf,
                threshold_info=ale_threshold_info,
                current_value=float(current_feature_value),
            )

            if auto_target is None:
                st.warning(
                    "ALE 기반 목표값을 계산하지 못했습니다."
                )
            else:
                target_feature_value = st.number_input(
                    f"Counterfactual 목표 {pretty_time_text(ice_feature)} 값",
                    min_value=float(feature_min),
                    max_value=float(feature_max),
                    value=float(auto_target),
                    key="ale_cf_target_value",
                    help=(
                        "기본값은 Centered ALE에서 예측에 가장 우호적인 중심 또는 "
                        "자동 탐지 임계점 이전의 우호 구간입니다. 사용자가 직접 변경할 수 있습니다."
                    ),
                )

                ale_cf_result = simulate_counterfactual_target_control(
                    model=model,
                    X_reference=X_test_ale,
                    feature=ice_feature,
                    current_value=float(current_feature_value),
                    target_value=float(target_feature_value),
                )

                if ale_cf_result is not None:
                    ale_control_recommendation = (
                        recommend_control_action_from_feature(
                            feature_name=ice_feature,
                            current_value=ale_cf_result[
                                "current_value"
                            ],
                            target_value=ale_cf_result[
                                "target_value"
                            ],
                        )
                    )

                    cf_c1, cf_c2, cf_c3, cf_c4 = st.columns(4)

                    with cf_c1:
                        st.metric(
                            "현재 Feature",
                            f"{ale_cf_result['current_value']:.3f}",
                        )

                    with cf_c2:
                        st.metric(
                            "목표 Feature",
                            f"{ale_cf_result['target_value']:.3f}",
                            delta=(
                                f"{ale_cf_result['change']:+.3f}"
                            ),
                        )

                    with cf_c3:
                        st.metric(
                            f"현재 예측 {report_target}",
                            f"{ale_cf_result['pred_current']:.3f}",
                        )

                    with cf_c4:
                        st.metric(
                            f"목표 예측 {report_target}",
                            f"{ale_cf_result['pred_target']:.3f}",
                            delta=(
                                f"{ale_cf_result['pred_delta']:+.3f}"
                            ),
                        )

                    current_side = (
                        "임계점 이상"
                        if float(current_feature_value)
                        >= float(
                            ale_threshold_info["threshold"]
                        )
                        else "임계점 미만"
                    )

                    improvement_text = (
                        "예측값 개선 방향"
                        if ale_cf_result["pred_delta"] > 0
                        else (
                            "예측값 감소 방향"
                            if ale_cf_result["pred_delta"] < 0
                            else "예측 변화 거의 없음"
                        )
                    )

                    st.markdown(
                        f"""
                        <div class="xai-insight-card">
                            <b>Counterfactual Target Control 결과</b><br>
                            현재값은 자동 탐지 임계점
                            <b>{ale_threshold_info['threshold']:.4f}</b> 기준
                            <b>{current_side}</b>에 위치합니다.<br>
                            선택 Feature를
                            <b>{ale_cf_result['current_value']:.4f}</b> →
                            <b>{ale_cf_result['target_value']:.4f}</b>로 변경하고
                            다른 Feature를 기준값에 고정한 one-feature counterfactual에서
                            예측 {report_target}은
                            <b>{ale_cf_result['pred_current']:.4f}</b> →
                            <b>{ale_cf_result['pred_target']:.4f}</b>
                            ({ale_cf_result['pred_delta']:+.4f})로 변화합니다.<br><br>
                            <b>판정:</b> {improvement_text}<br>
                            <b>환경제어 의사결정 제안:</b>
                            {ale_control_recommendation}<br><br>
                            이 결과는 실제 액추에이터의 즉시 제어량을 직접 산출하는 것이 아니라,
                            <b>현재 모델에서 예측값을 개선할 수 있는 환경 목표 방향 후보</b>입니다.
                            실제 자동제어에서는 장치 안전범위, 작물 생리범위,
                            환기·냉난방·CO₂·미스트 간 상호작용을 추가 제약조건으로 적용해야 합니다.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 현재/목표 상태를 ALE 그래프 위에서 추가 시각화
                    fig_cf = go.Figure()
                    fig_cf.add_trace(
                        go.Scatter(
                            x=ale_centers_cf,
                            y=ale_curve_cf,
                            mode="lines+markers",
                            line=dict(
                                width=4,
                                color="#2563eb",
                            ),
                            marker=dict(size=7),
                            name="Centered ALE",
                        )
                    )
                    fig_cf.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="#64748b",
                    )
                    fig_cf.add_vline(
                        x=float(
                            ale_threshold_info["threshold"]
                        ),
                        line_dash="dot",
                        line_color="#dc2626",
                        annotation_text="Threshold",
                    )
                    fig_cf.add_vline(
                        x=float(
                            ale_cf_result["current_value"]
                        ),
                        line_dash="dash",
                        line_color="#f59e0b",
                        annotation_text="Current",
                    )
                    fig_cf.add_vline(
                        x=float(
                            ale_cf_result["target_value"]
                        ),
                        line_dash="dash",
                        line_color="#059669",
                        annotation_text="Target",
                    )
                    fig_cf.update_layout(
                        title=(
                            "ALE 기반 Current → Counterfactual Target"
                        ),
                        xaxis_title=pretty_time_text(
                            ice_feature
                        ),
                        yaxis_title="Centered ALE",
                        height=390,
                        template=plotly_template,
                        margin=dict(
                            l=50,
                            r=25,
                            t=65,
                            b=45,
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.82)",
                    )
                    st.plotly_chart(
                        fig_cf,
                        use_container_width=True,
                        key="ale_counterfactual_target_plot",
                    )

        except Exception as e:
            st.error(
                f"Counterfactual Target Control 처리 오류: {e}"
            )
    else:
        st.info(
            "Centered ALE와 Threshold Detection이 정상 계산되면 "
            "Counterfactual Target Control이 활성화됩니다."
        )


    st.markdown(
        """
        <div class="xai-insight-card">
            <b>확장 XAI 의사결정 흐름</b><br>
            Centered ALE → 1D Bootstrap 95% CI → Threshold Detection →
            Counterfactual Target Control 순으로 연결됩니다.<br>
            ALE는 비선형 영향방향을, Bootstrap CI는 표본 불확실성을,
            Threshold Detection은 감소 후보 전환점을,
            Counterfactual은 예측 개선을 위한 목표 Feature 방향을 제시합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 9) 종합 리포트")

    try:
        comprehensive_report = generate_comprehensive_report(
            model_choice=model_choice,
            target_col=target_col,
            metrics=metrics,
            weekly_metrics_df=weekly_metrics_df,
            shap_df=shap_df,
            fi_df=fi_df,
            week_importance=week_importance,
            heatmap_df=heatmap_df,
            cf_result=cf_result,
            ice_feature=ice_feature if "ice_feature" in locals() else None,
            ice_mean_slope=ice_mean_slope,
            ice_std_slope=ice_std_slope,
            pdp_summary=pdp_summary,
            ale_summary=ale_summary,
            bin_centers=bin_centers,
            ale_vals=ale_vals,
        )

        st.markdown(
            f"""
<div style="background:linear-gradient(135deg,#ffffff,#eef5ff); box-shadow:0 6px 20px rgba(0,0,0,0.05);
padding:18px;
border-radius:12px;
line-height:2.0;
font-size:16px;
word-break:normal;
overflow-wrap:break-word">

{comprehensive_report}

</div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.warning(f"종합 리포트 생성 오류: {e}")

    st.success("통합 XAI 분석이 완료되었습니다.")
else:
    st.info("환경센서 CSV와 수확/생육 CSV를 업로드하면 분석을 시작합니다.")


# ===== Harvest smoothing features =====
def add_harvest_features(df, target_cols=None):
    """수확수 원자료에서 1~4주 고정 이동평균 컬럼을 생성합니다."""
    df = df.copy()
    harvest_candidates = ["수확수", "harvest_count", "HarvestCount"]
    found = next(
        (col for col in harvest_candidates if col in df.columns),
        None,
    )

    if found is not None:
        harvest_series = pd.to_numeric(
            df[found],
            errors="coerce",
        )
        for window in range(1, 5):
            df[f"{window}주평균수확수"] = (
                harvest_series
                .rolling(window=window, min_periods=window)
                .mean()
            )
    return df


# 1~4주 수확수 이동평균 성능 비교 요약
if (
    "df" in locals()
    and isinstance(df, pd.DataFrame)
    and "수확수" in df.columns
    and "target_col" in locals()
    and is_harvest_window_optimizer_target(target_col)
):
    with st.expander(
        "🍅 1~4주 수확수 이동평균 R² 비교 요약",
        expanded=False,
    ):
        try:
            if (
                "harvest_window_comparison_df" in locals()
                and isinstance(harvest_window_comparison_df, pd.DataFrame)
                and not harvest_window_comparison_df.empty
            ):
                summary_cols = [
                    "평균기간(주)",
                    "예측대상",
                    "MSE",
                    "MAE",
                    "R2",
                    "공통 데이터수",
                ]
                st.dataframe(
                    style_dataframe(
                        harvest_window_comparison_df[
                            summary_cols
                        ].round(4)
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                valid_summary = harvest_window_comparison_df.dropna(
                    subset=["R2"]
                )
                if not valid_summary.empty:
                    best_row = valid_summary.loc[
                        valid_summary["R2"].idxmax()
                    ]
                    st.success(
                        f"최고 R² 기간은 {int(best_row['평균기간(주)'])}주이며, "
                        f"R²={best_row['R2']:.4f}입니다."
                    )
            else:
                st.info(
                    "환경·수확 데이터를 업로드하면 1~4주 평균기간의 "
                    "R² 비교 결과가 표시됩니다."
                )
        except Exception as e:
            st.warning(f"수확수 평균기간 비교 요약 오류: {e}")


# -------------------------------------------------------------
# v12.1 안정화 점검 기준
# -------------------------------------------------------------
# 1. 앱 실행 시 target_col NameError가 발생하지 않아야 합니다.
# 2. 예측대상 목록에 수확수와 1~4주평균수확수가 포함되는지 확인합니다.
# 3. 1~4주 평균 R² 비교표와 최고기간 추천 메시지가 표시됩니다.
# 4. 수확수 안정화 변수(수확수_Lag1~Lag7, 착과잔량(Fruit Load))가 df에 생성됩니다.
# 5. SHAP / ALE / 은 현재 선택한 target_col 기준으로 작동합니다.
# 6. 1~4주 수확수 평균기간 비교 표가 expander 안에 표시됩니다.



# ================================
# Graph options extension
# ================================
try:
    extra_graph_cols = (
        [f"{window}주평균수확수" for window in range(1, 5)]
        + [
            "누적수확수",
            "누적착과수",
            "착과잔량(Fruit Load)",
        ]
    )

    if 'graph_options' in locals():
        for c in extra_graph_cols:
            if c in df.columns and c not in graph_options:
                graph_options.append(c)
except Exception:
    pass


# =============================================================
# v29.0 누적 학습형 스마트팜 작기 Knowledge Base / 우수작기 Benchmark
# =============================================================
CYCLE_KB_SCHEMA_VERSION = "29.3"
CYCLE_KB_DEFAULT_DB = "crop_cycle_knowledge_base.sqlite3"
CYCLE_KB_GEI_FEATURES = ["온도 GEI", "습도 GEI", "CO₂ GEI", "일사량 GEI", "통합 GEI"]


def _kb_to_float(value, default=np.nan):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def _kb_json_safe(value):
    """numpy/pandas 값을 JSON 직렬화 가능한 파이썬 기본형으로 변환합니다."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _kb_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_kb_json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def init_crop_cycle_kb(db_path):
    """작기별 요약 레코드를 저장할 경량 SQLite DB를 초기화합니다."""
    path = Path(str(db_path)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crop_cycle_records (
                cycle_id TEXT PRIMARY KEY,
                farm_id TEXT,
                crop TEXT,
                cultivar TEXT,
                season TEXT,
                start_date TEXT,
                end_date TEXT,
                created_at TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cycle_crop ON crop_cycle_records(crop)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cycle_farm ON crop_cycle_records(farm_id)")
        conn.commit()
    return str(path)


def save_crop_cycle_record(db_path, record):
    db_path = init_crop_cycle_kb(db_path)
    summary_json = json.dumps(_kb_json_safe(record), ensure_ascii=False)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO crop_cycle_records (
                cycle_id, farm_id, crop, cultivar, season, start_date, end_date,
                created_at, schema_version, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cycle_id) DO UPDATE SET
                farm_id=excluded.farm_id,
                crop=excluded.crop,
                cultivar=excluded.cultivar,
                season=excluded.season,
                start_date=excluded.start_date,
                end_date=excluded.end_date,
                created_at=excluded.created_at,
                schema_version=excluded.schema_version,
                summary_json=excluded.summary_json
            """,
            (
                str(record.get("작기 ID", "")),
                str(record.get("농가 ID", "")),
                str(record.get("작물", "")),
                str(record.get("품종", "")),
                str(record.get("계절", "")),
                str(record.get("재배 시작일", "")),
                str(record.get("재배 종료일", "")),
                pd.Timestamp.now().isoformat(),
                CYCLE_KB_SCHEMA_VERSION,
                summary_json,
            ),
        )
        conn.commit()


def delete_crop_cycle_record(db_path, cycle_id):
    db_path = init_crop_cycle_kb(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM crop_cycle_records WHERE cycle_id = ?", (str(cycle_id),))
        conn.commit()


def load_crop_cycle_records(db_path):
    db_path = init_crop_cycle_kb(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT cycle_id, summary_json FROM crop_cycle_records ORDER BY created_at"
        ).fetchall()
    records = []
    for cycle_id, payload in rows:
        try:
            rec = json.loads(payload)
            rec.setdefault("작기 ID", cycle_id)
            records.append(rec)
        except Exception:
            continue
    return records


def _kb_records_to_dataframe(records):
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep=".")


def _kb_numeric_candidates(df_input):
    if not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return []
    cols = []
    for col in df_input.columns:
        s = pd.to_numeric(df_input[col], errors="coerce")
        if s.notna().sum() >= 2:
            cols.append(col)
    return cols


def _kb_aggregate(series, mode):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan
    if mode == "합계":
        return float(s.sum())
    if mode == "마지막값":
        return float(s.iloc[-1])
    if mode == "최대값":
        return float(s.max())
    return float(s.mean())


def _kb_compute_ngr(gei_df, target_col):
    if gei_df is None or gei_df.empty or not target_col or target_col not in gei_df.columns:
        return np.nan
    try:
        gei_feature = next((c for c in CYCLE_KB_GEI_FEATURES if c in gei_df.columns), None)
        if gei_feature is None:
            return np.nan
        response_df, _ = build_gei_growth_response_curve(
            gei_df=gei_df,
            gei_feature=gei_feature,
            target_col=target_col,
            baseline_mode="생육단계 기대 변화량 대비(최종 권장)",
            stable_band_pct=2.0,
            danger_pct=-10.0,
        )
        if response_df is None or response_df.empty or "반응률(%)" not in response_df.columns:
            return np.nan
        return float(pd.to_numeric(response_df["반응률(%)"], errors="coerce").mean())
    except Exception:
        return np.nan


def _kb_threshold_for_feature(feature):
    thresholds = st.session_state.get("gei_thresholds", {})
    if isinstance(thresholds, dict):
        value = _kb_to_float(thresholds.get(feature, np.nan))
        if np.isfinite(value):
            return value
    return np.nan


def build_current_cycle_kb_snapshot(
    gei_df,
    metadata,
    performance,
    ngr_target=None,
    integrated_threshold=None,
):
    """현재 분석결과를 작기 단위 표준 레코드로 요약합니다."""
    record = dict(metadata)
    gei_work = gei_df.copy() if isinstance(gei_df, pd.DataFrame) else pd.DataFrame()
    if not gei_work.empty and "조사일자" in gei_work.columns:
        gei_work["조사일자"] = pd.to_datetime(gei_work["조사일자"], errors="coerce")
        gei_work = gei_work.sort_values("조사일자")
        if not record.get("재배 시작일") and gei_work["조사일자"].notna().any():
            record["재배 시작일"] = gei_work["조사일자"].min().date().isoformat()
        if not record.get("재배 종료일") and gei_work["조사일자"].notna().any():
            record["재배 종료일"] = gei_work["조사일자"].max().date().isoformat()

    # GEI 평균/변동/임계초과율
    for feature in CYCLE_KB_GEI_FEATURES:
        if feature not in gei_work.columns:
            continue
        s = pd.to_numeric(gei_work[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        record[f"{feature} 평균"] = float(s.mean())
        record[f"{feature} 표준편차"] = float(s.std(ddof=0))
        record[f"{feature} 최종"] = float(s.iloc[-1])
        threshold = _kb_threshold_for_feature(feature)
        if feature == "통합 GEI" and integrated_threshold is not None and np.isfinite(_kb_to_float(integrated_threshold)):
            threshold = float(integrated_threshold)
        if np.isfinite(threshold):
            record[f"{feature} 임계값"] = float(threshold)
            record[f"{feature} 위험초과율(%)"] = float((s >= threshold).mean() * 100.0)

    # 데이터 기반 통합 GEI 가중치
    selected_weights = st.session_state.get("gei_selected_weights", {})
    if isinstance(selected_weights, dict):
        for feature, weight in selected_weights.items():
            record[f"가중치.{feature}"] = _kb_to_float(weight)
    record["GEI 가중방식"] = GEI_WEIGHT_METHOD_LABELS.get(
        st.session_state.get("gei_weight_method", ""), st.session_state.get("gei_weight_method", "")
    )

    # NGR / XAI
    record["평균 NGR(%)"] = _kb_compute_ngr(gei_work, ngr_target)
    record["NGR Target"] = str(ngr_target or "")
    record["Lag SHAP Peak(주)"] = _kb_to_float(st.session_state.get("lag_shap_peak_week", np.nan))
    record["Lag SHAP Peak 비중(%)"] = _kb_to_float(st.session_state.get("lag_shap_peak_share", np.nan))
    record["Lag SHAP Target"] = str(st.session_state.get("lag_shap_target", ""))

    # 성과/운영 지표
    for key, value in performance.items():
        record[key] = _kb_to_float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value

    area = _kb_to_float(record.get("재배면적(m²)", np.nan))
    total_weight = _kb_to_float(record.get("총수확중량(kg)", np.nan))
    if np.isfinite(area) and area > 0 and np.isfinite(total_weight):
        record["단위면적 수확량(kg/m²)"] = total_weight / area

    # 위험 안정성은 임계초과가 적을수록 높도록 정의합니다.
    risk_pct = _kb_to_float(record.get("통합 GEI 위험초과율(%)", np.nan))
    record["GEI 안정성(0-100)"] = float(np.clip(100.0 - risk_pct, 0.0, 100.0)) if np.isfinite(risk_pct) else np.nan
    return record


def compute_cycle_ranking(records_df, score_weights):
    """생산량·상품과율·NGR·GEI 안정성·효율의 다목적 0~100 점수를 계산합니다."""
    if records_df is None or records_df.empty:
        return pd.DataFrame()
    out = records_df.copy()
    metrics = {
        "단위면적 수확량(kg/m²)": float(score_weights.get("yield", 0.35)),
        "상품과율(%)": float(score_weights.get("quality", 0.20)),
        "평균 NGR(%)": float(score_weights.get("growth", 0.20)),
        "GEI 안정성(0-100)": float(score_weights.get("gei", 0.15)),
        "에너지 효율지수(0-100)": float(score_weights.get("efficiency", 0.10)),
    }
    normalized = {}
    for col, weight in metrics.items():
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        finite = s[np.isfinite(s)]
        if finite.empty:
            continue
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-12:
            norm = pd.Series(50.0, index=out.index)
            norm[~np.isfinite(s)] = np.nan
        else:
            norm = (s - lo) / (hi - lo) * 100.0
        normalized[col] = (norm, weight)

    scores, coverages = [], []
    for idx in out.index:
        numerator = 0.0
        denominator = 0.0
        total_requested = sum(metrics.values()) or 1.0
        for col, (norm, weight) in normalized.items():
            value = norm.loc[idx]
            if np.isfinite(value) and weight > 0:
                numerator += float(value) * weight
                denominator += weight
        scores.append(numerator / denominator if denominator > 0 else np.nan)
        coverages.append(denominator / total_requested * 100.0)
    out["종합 생산성/안정성 점수"] = scores
    out["점수 데이터충족률(%)"] = coverages
    out = out.sort_values(["종합 생산성/안정성 점수", "단위면적 수확량(kg/m²)" if "단위면적 수확량(kg/m²)" in out.columns else "작기 ID"], ascending=[False, False]).reset_index(drop=True)
    out["순위"] = np.arange(1, len(out) + 1)
    return out


def build_top_cycle_reference(ranked_df, top_percent=20, min_top_n=1):
    if ranked_df is None or ranked_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    n = max(int(min_top_n), int(np.ceil(len(ranked_df) * float(top_percent) / 100.0)))
    n = min(n, len(ranked_df))
    top = ranked_df.head(n).copy()
    profile_features = [
        c for c in [
            "온도 GEI 평균", "습도 GEI 평균", "CO₂ GEI 평균", "일사량 GEI 평균", "통합 GEI 평균",
            "통합 GEI 위험초과율(%)", "평균 NGR(%)", "단위면적 수확량(kg/m²)", "상품과율(%)"
        ] if c in top.columns
    ]
    rows = []
    for feature in profile_features:
        s = pd.to_numeric(top[feature], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "지표": feature,
            "Q25": float(s.quantile(0.25)),
            "Median": float(s.median()),
            "Q75": float(s.quantile(0.75)),
            "Mean": float(s.mean()),
            "Std": float(s.std(ddof=0)),
            "상위작기수": int(s.size),
        })
    return top, pd.DataFrame(rows)


def compute_current_to_reference_similarity(current_record, reference_profile):
    if not isinstance(current_record, dict) or reference_profile is None or reference_profile.empty:
        return np.nan, pd.DataFrame()
    rows = []
    for _, ref in reference_profile.iterrows():
        feature = ref["지표"]
        current = _kb_to_float(current_record.get(feature, np.nan))
        if not np.isfinite(current):
            continue
        median = _kb_to_float(ref.get("Median", np.nan))
        q25, q75 = _kb_to_float(ref.get("Q25", np.nan)), _kb_to_float(ref.get("Q75", np.nan))
        if not np.isfinite(median):
            continue
        iqr = abs(q75 - q25) if np.isfinite(q25) and np.isfinite(q75) else np.nan
        scale = max(iqr, abs(median) * 0.10, 1e-6) if np.isfinite(iqr) else max(abs(median) * 0.10, 1e-6)
        distance = abs(current - median) / scale
        similarity = float(100.0 * np.exp(-distance))
        state = "Reference 범위"
        if np.isfinite(q25) and current < q25:
            state = "Reference 하단 이탈"
        if np.isfinite(q75) and current > q75:
            state = "Reference 상단 이탈"
        rows.append({
            "지표": feature,
            "현재": current,
            "Reference Q25": q25,
            "Reference Median": median,
            "Reference Q75": q75,
            "편차(현재-Median)": current - median,
            "유사도(%)": similarity,
            "상태": state,
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return np.nan, detail
    # GEI/생육 중심으로 동일가중 평균. 생산성과 상품과율은 결과지표이므로 실시간 추종 유사도에서 제외합니다.
    tracking_mask = ~detail["지표"].isin(["단위면적 수확량(kg/m²)", "상품과율(%)"])
    base = detail.loc[tracking_mask, "유사도(%)"] if tracking_mask.any() else detail["유사도(%)"]
    return float(base.mean()), detail


def compute_historical_gei_weights(ranked_df, top_only=False, top_percent=20):
    """저장된 작기별 선택 GEI 가중치를 생산성 점수로 가중평균하여 다음 작기 prior로 사용합니다."""
    if ranked_df is None or ranked_df.empty:
        return {}
    work = ranked_df.copy()
    if top_only:
        n = max(1, int(np.ceil(len(work) * float(top_percent) / 100.0)))
        work = work.head(n)
    features = [f"{env} GEI" for env in GEI_ENV_ORDER]
    accum = {f: 0.0 for f in features}
    denom = 0.0
    for _, row in work.iterrows():
        score = _kb_to_float(row.get("종합 생산성/안정성 점수", np.nan), default=50.0)
        cycle_weight = max(0.05, score / 100.0 if np.isfinite(score) else 0.5)
        vals = []
        for f in features:
            vals.append(_kb_to_float(row.get(f"가중치.{f}", np.nan)))
        valid = np.asarray([v if np.isfinite(v) else 0.0 for v in vals], dtype=float)
        if valid.sum() <= 0:
            continue
        valid = valid / valid.sum()
        for f, v in zip(features, valid):
            accum[f] += float(v) * cycle_weight
        denom += cycle_weight
    if denom <= 0:
        return {}
    return _normalize_nonnegative_weights([accum[f] / denom for f in features], features)


def _kb_import_csv_to_db(db_path, uploaded_file):
    if uploaded_file is None:
        return 0, []
    imported, errors = 0, []
    try:
        frame = pd.read_csv(uploaded_file)
    except Exception as exc:
        return 0, [str(exc)]
    for idx, row in frame.iterrows():
        try:
            record = {k: _kb_json_safe(v) for k, v in row.to_dict().items() if not (isinstance(v, float) and np.isnan(v))}
            cycle_id = str(record.get("작기 ID", "")).strip()
            if not cycle_id:
                raise ValueError("작기 ID가 없습니다.")
            save_crop_cycle_record(db_path, record)
            imported += 1
        except Exception as exc:
            errors.append(f"행 {idx + 1}: {exc}")
    return imported, errors


def render_continual_crop_cycle_knowledge_base():
    """현재 분석결과를 누적 저장하고 작기 간 ranking/reference/추종진단을 제공합니다."""
    render_stylish_section(
        "🧠 누적 학습형 작기 Knowledge Base · 우수작기 추종 의사결정",
        "매 작기의 환경·생육·수확·GEI·ALE 임계값·Lag SHAP 결과를 표준 레코드로 저장하고, 누적된 작기에서 상위 생산 작기의 공통 환경 profile을 학습하여 현재 작기의 위치와 위험 편차를 비교합니다.",
        kicker="CONTINUAL CROP-CYCLE LEARNING",
    )

    gei_df_current = st.session_state.get("gei_growth_dataset", pd.DataFrame())
    if not isinstance(gei_df_current, pd.DataFrame) or gei_df_current.empty:
        st.info("GEI 분석을 먼저 실행하면 현재 작기 Snapshot 저장과 우수작기 비교가 활성화됩니다. 기존 Knowledge Base 조회/가져오기는 사용할 수 있습니다.")

    # DB 위치: Raspberry Pi/로컬 환경에서는 이 파일이 작기 간 지속 저장소 역할을 합니다.
    db_default = os.environ.get("CROP_CYCLE_DB_PATH", CYCLE_KB_DEFAULT_DB)
    db_path = st.text_input(
        "작기 Knowledge Base SQLite 경로",
        value=str(st.session_state.get("crop_cycle_db_path", db_default)),
        key="crop_cycle_db_path_input_v290",
        help="Raspberry Pi에서는 영구 저장 디렉터리의 절대경로 사용을 권장합니다. 예: /home/pi/smartfarm/crop_cycle_knowledge.sqlite3",
    )
    st.session_state["crop_cycle_db_path"] = db_path
    try:
        resolved_db = init_crop_cycle_kb(db_path)
        st.caption(f"현재 저장소: {resolved_db}")
    except Exception as exc:
        st.error(f"Knowledge Base 초기화 오류: {exc}")
        return

    records = load_crop_cycle_records(resolved_db)
    records_df = _kb_records_to_dataframe(records)

    tabs = st.tabs(["① 현재 작기 저장", "② 작기 Ranking", "③ 우수작기 Reference", "④ 현재작기 추종진단", "⑤ DB 관리"])

    # ---------------------------------------------------------
    # 1. 현재 작기 Snapshot 저장
    # ---------------------------------------------------------
    with tabs[0]:
        if isinstance(gei_df_current, pd.DataFrame) and not gei_df_current.empty:
            gei_dates = pd.to_datetime(gei_df_current.get("조사일자"), errors="coerce") if "조사일자" in gei_df_current.columns else pd.Series(dtype="datetime64[ns]")
            default_start = gei_dates.min().date() if not gei_dates.empty and gei_dates.notna().any() else pd.Timestamp.today().date()
            default_end = gei_dates.max().date() if not gei_dates.empty and gei_dates.notna().any() else pd.Timestamp.today().date()
        else:
            default_start = default_end = pd.Timestamp.today().date()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            farm_id = st.text_input("농가 ID", value="Farm-01", key="kb_farm_id_v290")
            crop = st.text_input("작물", value="완숙토마토", key="kb_crop_v290")
        with m2:
            cycle_id = st.text_input("작기 ID", value=f"{farm_id}-Cycle-{default_end.year}", key="kb_cycle_id_v290")
            cultivar = st.text_input("품종", value="", key="kb_cultivar_v290")
        with m3:
            season = st.selectbox("계절/작형", ["미지정", "겨울작기", "봄작기", "여름작기", "가을작기", "연중"], key="kb_season_v290")
            area_m2 = st.number_input("재배면적(m²)", min_value=0.0, value=0.0, step=10.0, key="kb_area_v290")
        with m4:
            start_date = st.date_input("재배 시작일", value=default_start, key="kb_start_v290")
            end_date = st.date_input("재배 종료일", value=default_end, key="kb_end_v290")

        numeric_candidates = _kb_numeric_candidates(gei_df_current)
        target_candidates = [c for c in numeric_candidates if "GEI" not in str(c) and "누적시간" not in str(c)]
        none_option = "(직접입력/없음)"
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            total_weight_col = st.selectbox("총수확중량 자동산출 컬럼", [none_option] + target_candidates, key="kb_total_weight_col_v290")
            total_weight_agg = st.selectbox("총수확중량 집계", ["합계", "마지막값", "평균"], key="kb_total_weight_agg_v290")
            auto_total_weight = _kb_aggregate(gei_df_current[total_weight_col], total_weight_agg) if total_weight_col != none_option else np.nan
            total_weight = st.number_input("총수확중량(kg)", min_value=0.0, value=float(auto_total_weight) if np.isfinite(auto_total_weight) and auto_total_weight >= 0 else 0.0, step=1.0, key="kb_total_weight_v290")
        with p2:
            harvest_count_col_kb = st.selectbox("수확수 자동산출 컬럼", [none_option] + target_candidates, key="kb_harvest_count_col_v290")
            auto_harvest_count = _kb_aggregate(gei_df_current[harvest_count_col_kb], "합계") if harvest_count_col_kb != none_option else np.nan
            total_harvest_count = st.number_input("총수확수", min_value=0.0, value=float(auto_harvest_count) if np.isfinite(auto_harvest_count) and auto_harvest_count >= 0 else 0.0, step=1.0, key="kb_harvest_count_v290")
        with p3:
            avg_weight_col = st.selectbox("평균과중 자동산출 컬럼", [none_option] + target_candidates, key="kb_avg_weight_col_v290")
            auto_avg_weight = _kb_aggregate(gei_df_current[avg_weight_col], "평균") if avg_weight_col != none_option else np.nan
            avg_weight = st.number_input("평균과중(g 또는 데이터 단위)", min_value=0.0, value=float(auto_avg_weight) if np.isfinite(auto_avg_weight) and auto_avg_weight >= 0 else 0.0, step=1.0, key="kb_avg_weight_v290")
        with p4:
            marketable_rate = st.number_input("상품과율(%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="kb_marketable_v290")
            efficiency_index = st.number_input("에너지 효율지수(0~100, 선택)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="kb_efficiency_v290")

        x1, x2, x3 = st.columns(3)
        with x1:
            ngr_target = st.selectbox("작기 평균 NGR Target", [none_option] + target_candidates, key="kb_ngr_target_v290")
        with x2:
            stored_integrated_threshold = _kb_threshold_for_feature("통합 GEI")
            default_threshold = float(stored_integrated_threshold) if np.isfinite(stored_integrated_threshold) else 60.0
            integrated_threshold = st.number_input("통합 GEI 위험 임계값", min_value=0.0, max_value=100.0, value=default_threshold, step=0.1, key="kb_integrated_threshold_v290", help="Centered ALE에서 통합 GEI 임계값이 계산되었다면 자동값을 우선 사용합니다. 미계산 시 연구자가 provisional threshold를 지정할 수 있습니다.")
        with x3:
            st.metric("현재 선택 GEI 가중방식", GEI_WEIGHT_METHOD_LABELS.get(st.session_state.get("gei_weight_method", ""), st.session_state.get("gei_weight_method", "미계산")))

        metadata = {
            "작기 ID": str(cycle_id).strip(), "농가 ID": str(farm_id).strip(), "작물": str(crop).strip(),
            "품종": str(cultivar).strip(), "계절": str(season), "재배 시작일": start_date.isoformat(),
            "재배 종료일": end_date.isoformat(), "재배면적(m²)": float(area_m2),
        }
        performance = {
            "총수확중량(kg)": float(total_weight), "총수확수": float(total_harvest_count),
            "평균과중": float(avg_weight), "상품과율(%)": float(marketable_rate),
            "에너지 효율지수(0-100)": float(efficiency_index),
        }
        current_snapshot = build_current_cycle_kb_snapshot(
            gei_df=gei_df_current,
            metadata=metadata,
            performance=performance,
            ngr_target=None if ngr_target == none_option else ngr_target,
            integrated_threshold=integrated_threshold,
        )
        st.session_state["current_cycle_kb_snapshot"] = current_snapshot

        preview_keys = [
            "작기 ID", "농가 ID", "작물", "품종", "계절", "재배 시작일", "재배 종료일",
            "온도 GEI 평균", "습도 GEI 평균", "CO₂ GEI 평균", "일사량 GEI 평균", "통합 GEI 평균",
            "통합 GEI 임계값", "통합 GEI 위험초과율(%)", "GEI 안정성(0-100)", "평균 NGR(%)",
            "총수확중량(kg)", "단위면적 수확량(kg/m²)", "상품과율(%)", "Lag SHAP Peak(주)"
        ]
        preview = pd.DataFrame([{k: current_snapshot.get(k, np.nan) for k in preview_keys}])
        st.dataframe(preview.round(4), use_container_width=True, hide_index=True)

        if st.button("💾 현재 작기 결과를 Knowledge Base에 저장/업데이트", type="primary", key="kb_save_v290"):
            if not str(cycle_id).strip():
                st.error("작기 ID를 입력하세요.")
            else:
                try:
                    save_crop_cycle_record(resolved_db, current_snapshot)
                    st.success(f"'{cycle_id}' 작기 결과를 누적 저장했습니다. 동일 작기 ID가 있으면 최신 결과로 갱신됩니다.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"작기 저장 오류: {exc}")

    # ---------------------------------------------------------
    # 공통 scoring 설정 / ranking
    # ---------------------------------------------------------
    with tabs[1]:
        if records_df.empty:
            st.info("저장된 작기가 없습니다. 먼저 ① 현재 작기 저장에서 레코드를 추가하세요.")
        else:
            st.markdown("**다목적 작기 성과점수 가중치**")
            wc1, wc2, wc3, wc4, wc5 = st.columns(5)
            with wc1: wy = st.number_input("수확량", 0.0, 1.0, 0.35, 0.05, key="kb_score_yield_v290")
            with wc2: wq = st.number_input("상품과율", 0.0, 1.0, 0.20, 0.05, key="kb_score_quality_v290")
            with wc3: wg = st.number_input("NGR", 0.0, 1.0, 0.20, 0.05, key="kb_score_growth_v290")
            with wc4: we = st.number_input("GEI 안정성", 0.0, 1.0, 0.15, 0.05, key="kb_score_gei_v290")
            with wc5: wf = st.number_input("효율", 0.0, 1.0, 0.10, 0.05, key="kb_score_efficiency_v290")
            score_weights = {"yield": wy, "quality": wq, "growth": wg, "gei": we, "efficiency": wf}
            ranked = compute_cycle_ranking(records_df, score_weights)
            st.session_state["crop_cycle_ranked_df"] = ranked
            display_cols = [c for c in ["순위", "작기 ID", "농가 ID", "작물", "품종", "계절", "종합 생산성/안정성 점수", "점수 데이터충족률(%)", "단위면적 수확량(kg/m²)", "상품과율(%)", "평균 NGR(%)", "통합 GEI 평균", "통합 GEI 위험초과율(%)", "GEI 안정성(0-100)"] if c in ranked.columns]
            st.dataframe(ranked[display_cols].round(4), use_container_width=True, hide_index=True, height=min(540, 38 * (len(ranked) + 1)))
            st.caption("점수는 저장된 작기 집단 안에서 각 지표를 0~100 min-max 정규화한 뒤 가중 평균합니다. 결측 지표는 해당 작기에서 제외하고 사용 가능한 가중치로 재정규화하며, '점수 데이터충족률'이 낮은 작기는 순위를 보수적으로 해석해야 합니다.")

    # ranking을 다른 탭에서도 사용할 수 있게 재계산(기본가중치)
    if records_df.empty:
        ranked_all = pd.DataFrame()
    else:
        ranked_all = compute_cycle_ranking(records_df, {"yield": 0.35, "quality": 0.20, "growth": 0.20, "gei": 0.15, "efficiency": 0.10})

    # ---------------------------------------------------------
    # 3. 우수작기 Reference
    # ---------------------------------------------------------
    with tabs[2]:
        if ranked_all.empty:
            st.info("Reference profile을 만들 저장 작기가 없습니다.")
        else:
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                crops = ["전체"] + sorted(ranked_all.get("작물", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
                filter_crop = st.selectbox("작물 필터", crops, key="kb_ref_crop_v290")
            with f2:
                cultivars = ["전체"] + sorted(ranked_all.get("품종", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
                filter_cultivar = st.selectbox("품종 필터", cultivars, key="kb_ref_cultivar_v290")
            with f3:
                seasons = ["전체"] + sorted(ranked_all.get("계절", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
                filter_season = st.selectbox("계절/작형 필터", seasons, key="kb_ref_season_v290")
            with f4:
                top_percent = st.slider("우수작기 상위 비율(%)", 10, 50, 20, 5, key="kb_ref_top_pct_v290")

            cohort = ranked_all.copy()
            if filter_crop != "전체" and "작물" in cohort.columns: cohort = cohort[cohort["작물"].astype(str) == filter_crop]
            if filter_cultivar != "전체" and "품종" in cohort.columns: cohort = cohort[cohort["품종"].astype(str) == filter_cultivar]
            if filter_season != "전체" and "계절" in cohort.columns: cohort = cohort[cohort["계절"].astype(str) == filter_season]
            cohort = cohort.sort_values("종합 생산성/안정성 점수", ascending=False).reset_index(drop=True)
            top_cycles, ref_profile = build_top_cycle_reference(cohort, top_percent=top_percent, min_top_n=1)
            st.session_state["top_cycle_reference_profile"] = ref_profile
            st.session_state["top_cycle_reference_records"] = top_cycles

            rc1, rc2, rc3 = st.columns(3)
            with rc1: st.metric("비교 작기 수", len(cohort))
            with rc2: st.metric("Reference 상위 작기 수", len(top_cycles))
            with rc3:
                best_cycle = str(top_cycles.iloc[0].get("작기 ID", "-")) if not top_cycles.empty else "-"
                st.metric("현재 최고 작기", best_cycle)
            if not top_cycles.empty:
                top_cols = [c for c in ["순위", "작기 ID", "농가 ID", "종합 생산성/안정성 점수", "단위면적 수확량(kg/m²)", "평균 NGR(%)", "통합 GEI 평균", "통합 GEI 위험초과율(%)"] if c in top_cycles.columns]
                st.markdown("**상위 Reference 작기**")
                st.dataframe(top_cycles[top_cols].round(4), use_container_width=True, hide_index=True)
            if not ref_profile.empty:
                st.markdown("**상위작기 공통 환경·생육 Profile (Q25–Median–Q75)**")
                st.dataframe(ref_profile.round(4), use_container_width=True, hide_index=True)

            historical_weights = compute_historical_gei_weights(cohort, top_only=True, top_percent=top_percent)
            if historical_weights:
                st.session_state["historical_gei_weights"] = historical_weights
                st.markdown("**상위작기 기반 누적 GEI 가중치 prior**")
                hcols = st.columns(len(historical_weights))
                for idx, (feature, value) in enumerate(historical_weights.items()):
                    with hcols[idx]:
                        st.metric(feature.replace(" GEI", ""), f"{value * 100:.1f}%")
                st.caption("이 값은 각 작기에서 이미 학습된 GEI 가중치를 해당 작기의 생산성/안정성 점수로 가중평균한 메타 수준 prior입니다. 다음 앱 rerun부터 GEI 가중방식에서 '누적작기 학습가중'으로 선택할 수 있습니다. 원자료 전체를 다시 학습하는 pooled model과는 구분됩니다.")

    # ---------------------------------------------------------
    # 4. 현재 작기 추종 진단
    # ---------------------------------------------------------
    with tabs[3]:
        current_record = st.session_state.get("current_cycle_kb_snapshot", None)
        ref_profile = st.session_state.get("top_cycle_reference_profile", pd.DataFrame())
        if not isinstance(current_record, dict):
            st.info("① 현재 작기 저장 탭에서 현재 Snapshot을 먼저 생성하세요.")
        elif not isinstance(ref_profile, pd.DataFrame) or ref_profile.empty:
            st.info("③ 우수작기 Reference 탭에서 비교집단과 상위 비율을 선택해 Reference profile을 생성하세요.")
        else:
            similarity, detail = compute_current_to_reference_similarity(current_record, ref_profile)
            s1, s2, s3 = st.columns(3)
            with s1: st.metric("Top-cycle 환경·생육 유사도", f"{similarity:.1f}%" if np.isfinite(similarity) else "N/A")
            with s2:
                current_risk = _kb_to_float(current_record.get("통합 GEI 위험초과율(%)", np.nan))
                ref_risk_row = ref_profile[ref_profile["지표"] == "통합 GEI 위험초과율(%)"]
                ref_risk = float(ref_risk_row.iloc[0]["Median"]) if not ref_risk_row.empty else np.nan
                delta_risk = current_risk - ref_risk if np.isfinite(current_risk) and np.isfinite(ref_risk) else np.nan
                st.metric("위험초과율 편차", f"{delta_risk:+.1f}%p" if np.isfinite(delta_risk) else "N/A")
            with s3:
                current_yield = _kb_to_float(current_record.get("단위면적 수확량(kg/m²)", np.nan))
                ref_yield_row = ref_profile[ref_profile["지표"] == "단위면적 수확량(kg/m²)"]
                ref_yield = float(ref_yield_row.iloc[0]["Median"]) if not ref_yield_row.empty else np.nan
                yield_gap = current_yield - ref_yield if np.isfinite(current_yield) and np.isfinite(ref_yield) else np.nan
                st.metric("Reference 수확량 대비", f"{yield_gap:+.2f} kg/m²" if np.isfinite(yield_gap) else "N/A")

            if not detail.empty:
                st.dataframe(detail.round(4), use_container_width=True, hide_index=True)
                deviations = detail[detail["상태"] != "Reference 범위"].copy()
                if deviations.empty:
                    st.success("현재 작기의 주요 GEI/NGR 지표가 상위작기 Reference Q25–Q75 범위 안에 있습니다.")
                else:
                    st.markdown("**의사결정 우선순위 후보**")
                    deviations["절대 표준화편차"] = 100.0 - deviations["유사도(%)"]
                    deviations = deviations.sort_values("절대 표준화편차", ascending=False)
                    for _, row in deviations.head(5).iterrows():
                        feature = str(row["지표"])
                        direction = "높음" if float(row["편차(현재-Median)"]) > 0 else "낮음"
                        if "GEI" in feature and "평균" in feature and direction == "높음":
                            message = "상위작기보다 위험노출 지수가 높습니다. 해당 환경의 불리 구간 누적시간을 우선 확인하고 기존 ALE/Counterfactual 분석의 목표제어값과 연결하세요."
                        elif feature == "평균 NGR(%)" and direction == "낮음":
                            message = "기대 생육속도 대비 성장반응이 상위작기보다 낮습니다. Lag SHAP의 peak week와 최근 GEI 임계초과 이벤트를 교차확인하세요."
                        else:
                            message = "Reference 중앙값과 차이가 큽니다. 품종·계절·생육단계가 동일한 비교집단인지 확인한 뒤 원환경 시계열과 함께 진단하세요."
                        st.warning(f"{feature}: 현재가 Reference 중앙값보다 {direction} ({row['편차(현재-Median)']:+.3f}). {message}")
            st.caption("이 추종진단은 과거 상위작기의 Q25–Q75 profile과 현재 작기의 거리 기반 benchmarking입니다. 인과제어 명령이 아니며, 실제 자동제어는 기존 Counterfactual Target Control + 작물 안전제약 + 장치제약을 통과한 뒤 수행해야 합니다.")

    # ---------------------------------------------------------
    # 5. DB 관리 / export / import
    # ---------------------------------------------------------
    with tabs[4]:
        st.markdown("**누적 Knowledge Base 내보내기**")
        if records_df.empty:
            st.info("저장된 레코드가 없습니다.")
        else:
            csv_bytes = records_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ 작기 Knowledge Base CSV 다운로드", data=csv_bytes, file_name="crop_cycle_knowledge_base_export.csv", mime="text/csv", key="kb_download_v290")
            st.dataframe(records_df.round(4), use_container_width=True, hide_index=True, height=min(420, 38 * (len(records_df) + 1)))

        st.markdown("**CSV에서 Knowledge Base 가져오기**")
        uploaded_kb = st.file_uploader("이 앱에서 내보낸 CSV 또는 동일 컬럼구조 CSV", type=["csv"], key="kb_import_file_v290")
        if st.button("CSV 가져오기", key="kb_import_btn_v290", disabled=uploaded_kb is None):
            count, errors = _kb_import_csv_to_db(resolved_db, uploaded_kb)
            if count:
                st.success(f"{count}개 작기 레코드를 가져왔습니다.")
            if errors:
                st.warning("일부 행 오류: " + " | ".join(errors[:5]))
            if count:
                st.rerun()

        st.markdown("**작기 레코드 삭제**")
        cycle_ids = [str(r.get("작기 ID", "")) for r in records if str(r.get("작기 ID", ""))]
        if cycle_ids:
            delete_id = st.selectbox("삭제할 작기 ID", cycle_ids, key="kb_delete_cycle_v290")
            confirm_delete = st.checkbox("선택한 작기를 삭제하는 데 동의합니다.", key="kb_delete_confirm_v290")
            if st.button("🗑️ 선택 작기 삭제", key="kb_delete_btn_v290", disabled=not confirm_delete):
                delete_crop_cycle_record(resolved_db, delete_id)
                st.success(f"{delete_id}를 삭제했습니다.")
                st.rerun()


# 앱 하단에 누적학습형 작기 Knowledge Base를 렌더링합니다.
try:
    render_continual_crop_cycle_knowledge_base()
except Exception as _kb_runtime_error:
    st.warning(f"누적 작기 Knowledge Base 모듈 처리 오류: {_kb_runtime_error}")


# =============================================================
# v29.1 통합 GEI 상승 ↔ 생육·수확 증감 세부 원인분해 모듈
# - 통합 GEI의 환경별 구성기여(온도/습도/CO₂/일사량)를 행별 가중치로 정확 분해
# - 조사일별 Δ통합GEI를 Δ환경기여량으로 분해하여 "무엇이 GEI를 올렸는가" 표시
# - 개별 환경 GEI → 선택 생육/수확 반응(NGR/변화량)에 대한 Local SHAP 방향·크기 계산
# - 통합 GEI 상승요인과 생육감소요인을 같은 표에서 교차해 위험강화/완충/불일치 판정
# - 초장/엽장/엽폭/엽수/줄기굵기/생장길이/화방높이/수확수/평균중량 등 동적 Target 지원
# - "원인"은 모델 attribution 의미이며 인과효과로 단정하지 않도록 UI에 명시
# =============================================================


def _v291_gei_feature_aliases():
    return {
        "온도 GEI": ["온도 GEI"],
        "습도 GEI": ["습도 GEI"],
        "CO₂ GEI": ["CO₂ GEI", "CO2 GEI"],
        "일사량 GEI": ["일사량 GEI"],
    }


def _v291_available_component_geis(df):
    """표준 표시명 -> 실제 데이터 컬럼명을 반환합니다."""
    result = {}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return result
    for canonical, aliases in _v291_gei_feature_aliases().items():
        for col in aliases:
            if col in df.columns:
                result[canonical] = col
                break
    return result


def _v291_match_target_candidates(df):
    """완숙토마토 생육·수확 주요 지표를 우선순위대로 찾아 반환합니다."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    preferred_groups = [
        ("초장", ["초장", "plant height", "height"]),
        ("엽장", ["엽장", "leaf length"]),
        ("엽폭", ["엽폭", "leaf width"]),
        ("엽수", ["엽수", "leaf number", "leaf count"]),
        ("줄기굵기", ["줄기굵기", "줄기 굵기", "경경", "stem diameter", "stem thickness"]),
        ("생장길이", ["생장길이", "생장 길이", "growth length"]),
        ("화방높이", ["화방높이", "화방 높이", "truss height"]),
        ("수확수", ["수확수", "harvest count"]),
        ("평균중량", ["평균중량", "평균 중량", "평균과중", "fruit weight", "average weight"]),
    ]

    numeric_cols = []
    for col in df.columns:
        if col == "조사일자" or "GEI" in str(col):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() >= 4:
            numeric_cols.append(col)

    chosen = []
    lower_map = {str(c).lower().replace(" ", ""): c for c in numeric_cols}
    for _label, aliases in preferred_groups:
        found = None
        for alias in aliases:
            key = str(alias).lower().replace(" ", "")
            # exact normalized match first
            if key in lower_map:
                found = lower_map[key]
                break
            # then contains match
            for norm, original in lower_map.items():
                if key and key in norm:
                    found = original
                    break
            if found is not None:
                break
        if found is not None and found not in chosen:
            chosen.append(found)

    # 사용자가 매핑한 다른 생육·수확 지표도 선택 가능하도록 뒤에 추가합니다.
    for col in numeric_cols:
        if col not in chosen:
            chosen.append(col)
    return chosen


def _v291_effective_gei_contributions(df, component_map, selected_weights):
    """행별 유효 가중치를 재정규화하여 통합 GEI 구성기여량을 정확히 분해합니다."""
    if not isinstance(df, pd.DataFrame) or df.empty or not component_map:
        return pd.DataFrame(index=getattr(df, "index", None))

    out = pd.DataFrame(index=df.index)
    base_weights = {}
    for canonical, actual_col in component_map.items():
        # 세션 가중치는 표준/실제 컬럼 어느 쪽 키도 허용
        weight = selected_weights.get(canonical, selected_weights.get(actual_col, np.nan)) if isinstance(selected_weights, dict) else np.nan
        if not np.isfinite(weight) or float(weight) < 0:
            weight = 1.0
        base_weights[canonical] = float(weight)

    for idx in df.index:
        valid = []
        for canonical, actual_col in component_map.items():
            value = pd.to_numeric(pd.Series([df.at[idx, actual_col]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                valid.append((canonical, actual_col, float(value), base_weights[canonical]))
        total_w = sum(v[3] for v in valid)
        if total_w <= 0 and valid:
            total_w = float(len(valid))
            valid = [(a, b, c, 1.0) for a, b, c, _ in valid]

        for canonical, _actual_col, value, weight in valid:
            eff_w = float(weight) / float(total_w) if total_w > 0 else np.nan
            out.at[idx, f"{canonical} 유효가중치"] = eff_w
            out.at[idx, f"{canonical} 통합기여량"] = eff_w * value

    contribution_cols = [f"{c} 통합기여량" for c in component_map]
    if contribution_cols:
        out["통합 GEI 재구성"] = out[contribution_cols].sum(axis=1, min_count=1)
        for c in component_map:
            out[f"{c} 통합기여량 변화"] = out[f"{c} 통합기여량"].diff()
        out["통합 GEI 변화(분해합)"] = out[[f"{c} 통합기여량 변화" for c in component_map]].sum(axis=1, min_count=1)
    return out


def _v291_make_attribution_model(model_name, n_rows):
    min_leaf = 1 if n_rows < 18 else 2
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=350,
            random_state=42,
            min_samples_leaf=min_leaf,
            max_features="sqrt" if n_rows < 20 else 1.0,
        )
    if model_name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=42, n_estimators=180, learning_rate=0.04, max_depth=2)
    if model_name == "XGBoost":
        return XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_estimators=260,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
        )
    if model_name == "LGBM":
        return LGBMRegressor(
            random_state=42,
            n_estimators=220,
            learning_rate=0.04,
            max_depth=3,
            verbosity=-1,
        )
    return RandomForestRegressor(n_estimators=350, random_state=42, min_samples_leaf=min_leaf)


def _v291_tree_shap_values(model, X):
    """TreeExplainer 우선, 실패 시 shap.Explainer로 안전하게 계산합니다."""
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        expected = np.asarray(explainer.expected_value).reshape(-1)
        expected_value = float(expected[0]) if len(expected) else np.nan
        return arr, expected_value
    except Exception:
        explainer = shap.Explainer(model, X)
        exp = explainer(X)
        arr = np.asarray(exp.values, dtype=float)
        if arr.ndim == 3:
            arr = arr[..., 0]
        base = np.asarray(exp.base_values, dtype=float).reshape(-1)
        expected_value = float(np.nanmean(base)) if len(base) else np.nan
        return arr, expected_value


def build_integrated_gei_growth_factor_attribution(
    gei_df,
    target_col,
    response_basis="생육단계 NGR",
    model_name="RandomForest",
    selected_weights=None,
):
    """
    통합 GEI 상승요인과 생육/수확 반응요인을 하나의 조사일-환경 long table로 반환합니다.

    해석 축
    1) 통합 GEI 구성/상승: 환경별 유효가중치 × 개별 GEI 및 그 변화량
    2) 생육/수확 증감: 개별 GEI를 입력으로 학습한 모델의 Local SHAP
    3) 두 축의 일치 여부: 통합 GEI를 올리면서 동시에 반응을 낮추는 환경을 위험 강화요인으로 표시
    """
    if not isinstance(gei_df, pd.DataFrame) or gei_df.empty or target_col not in gei_df.columns:
        return pd.DataFrame(), pd.DataFrame(), {}, "사용 불가", None

    component_map = _v291_available_component_geis(gei_df)
    if len(component_map) < 2:
        return pd.DataFrame(), pd.DataFrame(), component_map, "개별 GEI 부족", None

    actual_features = list(component_map.values())
    calibration_df, response_label = _prepare_gei_weight_target(
        gei_df=gei_df,
        target_col=target_col,
        response_basis=response_basis,
        gei_features=actual_features,
    )
    if calibration_df.empty or len(calibration_df) < 6:
        return pd.DataFrame(), pd.DataFrame(), component_map, response_label, None

    # 표준 표시명으로 입력 Feature명을 통일합니다.
    rename_to_canonical = {actual: canonical for canonical, actual in component_map.items()}
    calibration_df = calibration_df.rename(columns=rename_to_canonical)
    canonical_features = list(component_map.keys())
    X = calibration_df[canonical_features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(calibration_df["가중치학습반응"], errors="coerce")
    valid = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
    calibration_df = calibration_df.loc[valid].reset_index(drop=True)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    if len(calibration_df) < 6:
        return pd.DataFrame(), pd.DataFrame(), component_map, response_label, None

    model = _v291_make_attribution_model(model_name, len(calibration_df))
    model.fit(X, y)
    pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    shap_values, expected_value = _v291_tree_shap_values(model, X)
    if shap_values.shape[0] != len(X) or shap_values.shape[1] != len(canonical_features):
        return pd.DataFrame(), pd.DataFrame(), component_map, response_label, model

    # 원본 GEI를 조사일 기준으로 표준명화하여 통합 GEI 수학적 분해를 계산합니다.
    original_cols = ["조사일자"] + list(component_map.values())
    if "통합 GEI" in gei_df.columns:
        original_cols.append("통합 GEI")
    original = gei_df[original_cols].copy().rename(columns=rename_to_canonical)
    original["조사일자"] = pd.to_datetime(original["조사일자"], errors="coerce")
    original = original.dropna(subset=["조사일자"]).sort_values("조사일자").drop_duplicates("조사일자", keep="last").reset_index(drop=True)

    # 사용 중 가중치가 없으면 동일가중치로 안전 대체합니다.
    if not isinstance(selected_weights, dict) or not selected_weights:
        selected_weights = {c: 1.0 / len(canonical_features) for c in canonical_features}
    else:
        normalized_source = []
        for canonical in canonical_features:
            actual = component_map[canonical]
            normalized_source.append(selected_weights.get(canonical, selected_weights.get(actual, 0.0)))
        selected_weights = _normalize_nonnegative_weights(normalized_source, canonical_features)

    contribution = _v291_effective_gei_contributions(original, {c: c for c in canonical_features}, selected_weights)
    original = pd.concat([original, contribution], axis=1)
    original["조사일자"] = pd.to_datetime(original["조사일자"], errors="coerce")

    wide = calibration_df[["조사일자", "가중치학습반응"]].copy()
    wide["조사일자"] = pd.to_datetime(wide["조사일자"], errors="coerce")
    wide["모델예측반응"] = pred
    wide["SHAP 기준값"] = expected_value
    for j, feature in enumerate(canonical_features):
        wide[f"SHAP {feature}"] = shap_values[:, j]
        wide[feature] = X[feature].to_numpy(dtype=float)

    wide = wide.merge(original, on="조사일자", how="left", suffixes=("", "_원본"))
    if "통합 GEI" not in wide.columns:
        wide["통합 GEI"] = wide.get("통합 GEI 재구성", np.nan)
    wide["통합 GEI 변화"] = pd.to_numeric(wide["통합 GEI"], errors="coerce").diff()

    long_rows = []
    for _, row in wide.iterrows():
        neg_total = sum(max(-float(row.get(f"SHAP {f}", 0.0) or 0.0), 0.0) for f in canonical_features)
        pos_gei_total = sum(max(float(row.get(f"{f} 통합기여량 변화", 0.0) or 0.0), 0.0) for f in canonical_features)
        abs_shap_total = sum(abs(float(row.get(f"SHAP {f}", 0.0) or 0.0)) for f in canonical_features)

        for feature in canonical_features:
            shap_v = float(row.get(f"SHAP {feature}", np.nan))
            delta_contrib = float(row.get(f"{feature} 통합기여량 변화", np.nan))
            current_contrib = float(row.get(f"{feature} 통합기여량", np.nan))
            current_gei = float(row.get(feature, np.nan))
            eff_weight = float(row.get(f"{feature} 유효가중치", np.nan))
            neg_share = (max(-shap_v, 0.0) / neg_total * 100.0) if neg_total > 0 and np.isfinite(shap_v) else 0.0
            gei_rise_share = (max(delta_contrib, 0.0) / pos_gei_total * 100.0) if pos_gei_total > 0 and np.isfinite(delta_contrib) else 0.0
            total_shap_share = (abs(shap_v) / abs_shap_total * 100.0) if abs_shap_total > 0 and np.isfinite(shap_v) else 0.0

            if np.isfinite(delta_contrib) and delta_contrib > 0 and np.isfinite(shap_v) and shap_v < 0:
                verdict = "위험 강화: 통합 GEI↑ + 감소기여"
                risk_score = 0.5 * gei_rise_share + 0.5 * neg_share
            elif np.isfinite(delta_contrib) and delta_contrib > 0 and np.isfinite(shap_v) and shap_v >= 0:
                verdict = "GEI↑이나 생육·수확 완충/증가기여"
                risk_score = 0.5 * gei_rise_share
            elif np.isfinite(delta_contrib) and delta_contrib <= 0 and np.isfinite(shap_v) and shap_v < 0:
                verdict = "GEI 완화 중이나 감소기여 지속"
                risk_score = 0.5 * neg_share
            else:
                verdict = "우호/중립"
                risk_score = 0.0

            long_rows.append({
                "조사일자": row["조사일자"],
                "Target": target_col,
                "반응기준": response_label,
                "실제 반응": float(row.get("가중치학습반응", np.nan)),
                "모델예측 반응": float(row.get("모델예측반응", np.nan)),
                "통합 GEI": float(row.get("통합 GEI", np.nan)),
                "통합 GEI 변화": float(row.get("통합 GEI 변화", np.nan)),
                "환경": feature.replace(" GEI", ""),
                "개별 GEI": current_gei,
                "유효가중치": eff_weight,
                "통합 GEI 구성기여량": current_contrib,
                "통합 GEI 상승기여량": delta_contrib,
                "GEI 상승기여율(%)": gei_rise_share,
                "Local SHAP": shap_v,
                "전체 영향도(|SHAP|,%)": total_shap_share,
                "감소기여율(%)": neg_share,
                "복합 위험점수(0-100)": risk_score,
                "판정": verdict,
            })

    long_df = pd.DataFrame(long_rows)

    summary_rows = []
    if not long_df.empty:
        for env, grp in long_df.groupby("환경", dropna=False):
            n = max(len(grp), 1)
            summary_rows.append({
                "환경": env,
                "Mean |SHAP|": float(pd.to_numeric(grp["Local SHAP"], errors="coerce").abs().mean()),
                "평균 SHAP": float(pd.to_numeric(grp["Local SHAP"], errors="coerce").mean()),
                "감소기여 빈도(%)": float((pd.to_numeric(grp["Local SHAP"], errors="coerce") < 0).mean() * 100.0),
                "GEI 상승 빈도(%)": float((pd.to_numeric(grp["통합 GEI 상승기여량"], errors="coerce") > 0).mean() * 100.0),
                "위험강화 동시발생(%)": float((grp["판정"] == "위험 강화: 통합 GEI↑ + 감소기여").mean() * 100.0),
                "평균 복합위험점수": float(pd.to_numeric(grp["복합 위험점수(0-100)"], errors="coerce").mean()),
                "관측수": int(n),
            })
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["위험강화 동시발생(%)", "Mean |SHAP|"], ascending=[False, False]).reset_index(drop=True)

    metadata = {
        "response_label": response_label,
        "expected_value": expected_value,
        "model_name": model_name,
        "weights": selected_weights,
        "n": len(calibration_df),
    }
    return wide, long_df, component_map, response_label, model, summary_df, metadata


def render_integrated_gei_growth_factor_attribution():
    """Streamlit UI: 통합 GEI 상승원인과 생육/수확 증가·감소 요인을 같은 조사일에서 분해합니다."""
    gei_df = st.session_state.get("gei_growth_dataset", pd.DataFrame())
    if not isinstance(gei_df, pd.DataFrame) or gei_df.empty:
        return

    component_map = _v291_available_component_geis(gei_df)
    if len(component_map) < 2:
        return

    targets = _v291_match_target_candidates(gei_df)
    if not targets:
        return

    render_stylish_section(
        "🧬 통합 GEI 상승 ↔ 생육·수확 증가·감소 세부 요인분해",
        "통합 GEI가 높아지거나 낮아진 이유를 온도·습도·CO₂·일사량 GEI로 수학적으로 분해하고, 같은 조사일의 생육·수확 반응에는 각 환경 GEI가 어느 방향으로 기여했는지 Local SHAP으로 교차해석합니다.",
        kicker="GEI FACTOR ATTRIBUTION",
    )

    st.markdown(
        """
        <div class="xai-insight-card">
        <b>두 종류의 '요인'을 분리해서 봅니다.</b><br>
        ① <b>통합 GEI 상승요인</b> = 현재 적용 가중치 × 개별 환경 GEI의 변화로 계산되는 <u>지수의 수학적 구성요인</u><br>
        ② <b>생육·수확 증가/감소요인</b> = 개별 환경 GEI를 입력으로 학습한 모델의 <u>Local SHAP 기여방향</u><br>
        따라서 같은 온도 GEI가 통합 GEI를 올렸더라도 특정 시점에서는 생육을 완충하는 방향으로 나타날 수 있고, 반대로 통합 GEI 상승의 주원인과 실제 생육감소의 주기여 환경이 서로 다를 수도 있습니다.<br>
        <b>주의:</b> SHAP은 모델 attribution이며 인과효과를 직접 증명하지 않습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.35, 1.35, 1.0])
    with c1:
        default_idx = 0
        target_col = st.selectbox("분석 생육·수확 지표", targets, index=default_idx, key="gei_factor_target_v291")
    with c2:
        response_basis = st.selectbox(
            "증감 반응 기준",
            ["생육단계 NGR", "월평균 NGR", "조사간 변화량", "원자료"],
            index=0,
            key="gei_factor_response_basis_v291",
            help="초장·엽장·엽폭·생장길이처럼 누적되는 생육지표는 생육단계 NGR을 권장합니다. 수확수/중량은 조사간 변화량 또는 원자료도 비교할 수 있습니다.",
        )
    with c3:
        model_name = st.selectbox(
            "세부 기여도 모델",
            ["RandomForest", "XGBoost", "LGBM", "GradientBoosting"],
            index=0,
            key="gei_factor_model_v291",
        )

    selected_weights = st.session_state.get("gei_selected_weights", {})
    try:
        result = build_integrated_gei_growth_factor_attribution(
            gei_df=gei_df,
            target_col=target_col,
            response_basis=response_basis,
            model_name=model_name,
            selected_weights=selected_weights,
        )
        if len(result) == 7:
            wide_df, long_df, _component_map, response_label, _model, summary_df, meta = result
        else:
            st.info("세부 요인분해에 필요한 유효 데이터가 부족합니다.")
            return
    except Exception as exc:
        st.warning(f"GEI 세부 요인분해 계산 오류: {exc}")
        return

    if long_df.empty or wide_df.empty:
        st.info("선택 Target에서 최소 6개 이상의 완전한 조사일과 2개 이상의 개별 GEI가 필요합니다.")
        return

    # 가장 감소가 큰 조사일을 기본 선택으로 사용합니다.
    date_response = (
        long_df[["조사일자", "실제 반응"]]
        .drop_duplicates("조사일자")
        .dropna(subset=["조사일자", "실제 반응"])
        .sort_values("조사일자")
    )
    if date_response.empty:
        return
    worst_idx = pd.to_numeric(date_response["실제 반응"], errors="coerce").idxmin()
    worst_date = pd.to_datetime(date_response.loc[worst_idx, "조사일자"])
    date_options = date_response["조사일자"].tolist()
    default_date_idx = date_options.index(worst_date) if worst_date in date_options else len(date_options) - 1

    selector_col, metric_col1, metric_col2, metric_col3 = st.columns([1.45, 1.0, 1.0, 1.0])
    with selector_col:
        selected_date = st.selectbox(
            "상세 분석 조사일",
            date_options,
            index=default_date_idx,
            format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"),
            key="gei_factor_date_v291",
            help="기본값은 선택 Target의 반응이 가장 낮았던 조사일입니다.",
        )

    day = long_df[pd.to_datetime(long_df["조사일자"]) == pd.to_datetime(selected_date)].copy()
    day = day.sort_values("복합 위험점수(0-100)", ascending=False)
    day_first = day.iloc[0]
    with metric_col1:
        st.metric("실제 반응", f"{float(day_first['실제 반응']):+.2f}")
    with metric_col2:
        st.metric("통합 GEI", f"{float(day_first['통합 GEI']):.2f}" if np.isfinite(day_first["통합 GEI"]) else "N/A")
    with metric_col3:
        delta_int = float(day_first.get("통합 GEI 변화", np.nan))
        st.metric("Δ 통합 GEI", f"{delta_int:+.2f}" if np.isfinite(delta_int) else "N/A")

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        render_panel_label("통합 GEI 상승/하락의 환경별 구성기여")
        fig_gei = go.Figure()
        fig_gei.add_trace(go.Bar(
            x=day["환경"],
            y=day["통합 GEI 상승기여량"],
            text=[f"{v:+.2f}" if np.isfinite(v) else "" for v in pd.to_numeric(day["통합 GEI 상승기여량"], errors="coerce")],
            textposition="outside",
            name="Δ통합GEI 기여량",
        ))
        fig_gei.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig_gei.update_layout(
            height=380,
            title=f"{pd.to_datetime(selected_date).strftime('%Y-%m-%d')} · Δ통합 GEI 환경별 분해",
            yaxis_title="통합 GEI 변화 기여량",
            margin=dict(l=50, r=20, t=65, b=45),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.82)",
        )
        display_plotly(fig_gei)

    with chart_right:
        render_panel_label(f"{target_col} 반응에 대한 환경별 Local SHAP")
        shap_day = day.sort_values("Local SHAP")
        fig_shap = go.Figure(go.Bar(
            x=shap_day["환경"],
            y=shap_day["Local SHAP"],
            text=[f"{v:+.3f}" if np.isfinite(v) else "" for v in pd.to_numeric(shap_day["Local SHAP"], errors="coerce")],
            textposition="outside",
            name="Local SHAP",
        ))
        fig_shap.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig_shap.update_layout(
            height=380,
            title=f"{response_label} · 음수=감소방향 / 양수=증가방향",
            yaxis_title="Local SHAP",
            margin=dict(l=50, r=20, t=65, b=45),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.82)",
        )
        display_plotly(fig_shap)

    # 상세 표
    show_cols = [
        "환경", "개별 GEI", "유효가중치", "통합 GEI 구성기여량", "통합 GEI 상승기여량",
        "GEI 상승기여율(%)", "Local SHAP", "전체 영향도(|SHAP|,%)", "감소기여율(%)",
        "복합 위험점수(0-100)", "판정",
    ]
    show_day = day[show_cols].copy()
    show_day["유효가중치"] = show_day["유효가중치"] * 100.0
    show_day = show_day.rename(columns={"유효가중치": "유효가중치(%)"})
    st.dataframe(show_day.round(4), use_container_width=True, hide_index=True)

    # 자동 해석
    rise_candidates = day[pd.to_numeric(day["통합 GEI 상승기여량"], errors="coerce") > 0].copy()
    decline_candidates = day[pd.to_numeric(day["Local SHAP"], errors="coerce") < 0].copy()
    top_rise_env = None
    top_decline_env = None
    if not rise_candidates.empty:
        top_rise_env = str(rise_candidates.sort_values("통합 GEI 상승기여량", ascending=False).iloc[0]["환경"])
    if not decline_candidates.empty:
        top_decline_env = str(decline_candidates.sort_values("Local SHAP", ascending=True).iloc[0]["환경"])

    response_value = float(day_first["실제 반응"])
    response_state = "기준 대비 감소/억제" if response_value < 0 else ("기준 대비 증가" if response_value > 0 else "기준 수준")
    if top_rise_env and top_decline_env and top_rise_env == top_decline_env:
        consistency_text = f"<b>{top_rise_env}</b>가 통합 GEI 상승의 가장 큰 요인이면서 동시에 {target_col} 감소방향의 가장 큰 Local SHAP 요인으로 나타났습니다. 두 분석축의 방향이 일치합니다."
    elif top_rise_env and top_decline_env:
        consistency_text = f"통합 GEI 상승의 최대 요인은 <b>{top_rise_env}</b>이지만, {target_col} 감소방향의 최대 모델 기여요인은 <b>{top_decline_env}</b>입니다. 즉 <u>지수 상승 원인과 실제 반응의 주기여 환경이 다릅니다.</u>"
    elif top_decline_env:
        consistency_text = f"해당 조사일에는 통합 GEI의 뚜렷한 상승이 없지만 <b>{top_decline_env}</b>가 {target_col} 감소방향으로 가장 크게 기여했습니다."
    else:
        consistency_text = "해당 조사일에는 네 환경 GEI 중 뚜렷한 감소방향 Local SHAP이 확인되지 않았습니다."

    st.markdown(
        f"""
        <div class="xai-insight-card">
        <b>자동 해석 · {pd.to_datetime(selected_date).strftime('%Y-%m-%d')}</b><br>
        • 실제 반응: <b>{response_value:+.2f}</b> → {response_state}<br>
        • Δ통합 GEI: <b>{delta_int:+.2f}</b> → {'위험지수 상승' if np.isfinite(delta_int) and delta_int > 0 else '위험지수 하락/유지'}<br>
        • {consistency_text}<br>
        • 세부 판정에서 <b>“위험 강화: 통합 GEI↑ + 감소기여”</b>는 해당 환경이 통합지수를 올리는 동시에 모델에서 생육·수확 반응을 낮추는 방향으로 나타난 경우입니다.<br>
        • 반대로 <b>“GEI↑이나 생육·수확 완충/증가기여”</b>가 나타나면 통합 GEI가 높다는 사실만으로 모든 환경이 동시에 악영향을 준다고 해석하면 안 됩니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 전체 기간 요약
    st.markdown("**전체 조사기간 환경별 감소기여·위험강화 요약**")
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        st.dataframe(summary_df.round(4), use_container_width=True, hide_index=True)

    with st.expander("📅 전체 조사일별 세부 요인 데이터 보기", expanded=False):
        long_show = long_df.copy().sort_values(["조사일자", "복합 위험점수(0-100)"], ascending=[True, False])
        st.dataframe(long_show.round(4), use_container_width=True, hide_index=True, height=520)

    # 다른 모듈/향후 DB 확장을 위해 결과를 세션에 보존합니다.
    st.session_state["gei_factor_attribution_target"] = target_col
    st.session_state["gei_factor_attribution_response_basis"] = response_basis
    st.session_state["gei_factor_attribution_long"] = long_df
    st.session_state["gei_factor_attribution_summary"] = summary_df
    st.session_state["gei_factor_attribution_metadata"] = meta


# 기존 v29.0 누적학습 Knowledge Base 아래에 v29.1 원인분해 기능을 추가 렌더링합니다.
try:
    render_integrated_gei_growth_factor_attribution()
except Exception as _gei_factor_runtime_error:
    st.warning(f"통합 GEI 세부 요인분해 모듈 처리 오류: {_gei_factor_runtime_error}")


# =============================================================
# v29.2 완숙토마토 생장균형 판정 + 방향별 환경 GEI 기여도 확장본
# - 줄기굵기: <=10 mm 생식생장, >10~<13 mm 평균생장, >=13 mm 영양생장 (기준 12 mm)
# - 화방높이: <15 cm 생식생장, 15~<20 cm 평균생장, >=20 cm 영양생장 (기준 18 cm)
# - 줄기굵기/화방높이는 절대값 기반 생장균형 상태를 판정하고 Local SHAP 방향으로 영양/생식 기여 환경 GEI 분석
# - 기타 생육지표는 NGR/변화량 기반 증가·감소 방향으로 환경 GEI 기여도를 분석
# - 수확수/수확중량/평균중량의 증가·감소 방향별 환경 GEI 기여도 분석
# - 원환경(온도/습도/CO2/일사량) 시계열이 존재하면 계절·월 변화와 GEI 기여를 결합한 자동 설명 생성
# - 모델 attribution과 관측 시계열의 결합 해석이며 인과효과로 단정하지 않음
# =============================================================


def _v292_target_kind(target_col):
    name = str(target_col).lower().replace(" ", "")
    if any(k in name for k in ["줄기굵기", "경경", "stemdiameter", "stemthickness"]):
        return "stem"
    if any(k in name for k in ["화방높이", "trussheight"]):
        return "truss"
    if any(k in name for k in ["수확중량", "총수확중량", "yieldweight", "harvestweight"]):
        return "harvest_weight"
    if any(k in name for k in ["평균중량", "평균과중", "averageweight", "fruitweight"]):
        return "avg_weight"
    if any(k in name for k in ["수확수", "harvestcount"]):
        return "harvest_count"
    return "growth"


def _v292_classify_balance(target_col, value):
    """사용자 제공 완숙토마토 기준으로 줄기굵기/화방높이 생장균형 상태를 판정합니다."""
    try:
        v = float(value)
    except Exception:
        return "판정불가", np.nan, ""
    if not np.isfinite(v):
        return "판정불가", np.nan, ""
    kind = _v292_target_kind(target_col)
    if kind == "stem":
        # 사용자 문장에 '12cm 기준'이 있었지만 경계값이 mm로 제시되어 12 mm 기준으로 해석합니다.
        if v <= 10.0:
            return "생식생장", 12.0, "≤10 mm"
        if v < 13.0:
            return "평균생장", 12.0, ">10~<13 mm"
        return "영양생장", 12.0, "≥13 mm"
    if kind == "truss":
        if v < 15.0:
            return "생식생장", 18.0, "<15 cm"
        if v < 20.0:
            return "평균생장", 18.0, "15~<20 cm"
        return "영양생장", 18.0, "≥20 cm"
    return "증가" if v > 0 else ("감소" if v < 0 else "유지"), np.nan, ""


def _v292_expand_target_candidates(df):
    targets = _v291_match_target_candidates(df)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return targets
    preferred_extra = [
        "수확중량", "총수확중량", "수확 중량", "yield weight", "harvest weight",
        "평균중량", "평균과중", "수확수", "초장", "엽장", "엽폭", "엽수",
        "줄기굵기", "경경", "생장길이", "화방높이",
    ]
    numeric_cols = []
    for c in df.columns:
        if c == "조사일자" or "GEI" in str(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 4:
            numeric_cols.append(c)
    norm = {str(c).lower().replace(" ", ""): c for c in numeric_cols}
    ordered = []
    for p in preferred_extra:
        key = str(p).lower().replace(" ", "")
        match = norm.get(key)
        if match is None:
            for nk, original in norm.items():
                if key in nk or nk in key:
                    match = original
                    break
        if match is not None and match not in ordered:
            ordered.append(match)
    for c in targets:
        if c not in ordered:
            ordered.append(c)
    return ordered


def _v292_raw_env_aliases():
    return {
        "온도 GEI": ["일평균온도", "평균온도", "ADT", "주간평균온도", "야간평균온도", "온도평균", "temperature", "temp"],
        "습도 GEI": ["일평균습도", "평균습도", "상대습도", "습도평균", "humidity", "rh"],
        "CO₂ GEI": ["평균co2", "co2평균", "co₂평균", "co2", "co₂"],
        "일사량 GEI": ["평균일사량", "일평균일사량", "적산일사", "dli", "radiation", "solar"],
    }


def _v292_find_raw_env_column(df, gei_name):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    aliases = _v292_raw_env_aliases().get(gei_name, [])
    candidates = []
    for c in df.columns:
        cs = str(c)
        if "GEI" in cs or cs == "조사일자":
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 4:
            continue
        n = cs.lower().replace(" ", "").replace("₂", "2")
        score = 0
        for a in aliases:
            ak = str(a).lower().replace(" ", "").replace("₂", "2")
            if n == ak:
                score = max(score, 100)
            elif ak in n:
                score = max(score, 60 + min(len(ak), 20))
        # 환경별 기본 토큰 보조 탐색
        if gei_name.startswith("온도") and any(k in n for k in ["온도", "temp", "adt"]): score = max(score, 40)
        if gei_name.startswith("습도") and any(k in n for k in ["습도", "humidity", "rh"]): score = max(score, 40)
        if gei_name.startswith("CO") and any(k in n for k in ["co2"]): score = max(score, 40)
        if gei_name.startswith("일사") and any(k in n for k in ["일사", "radiation", "solar", "dli"]): score = max(score, 40)
        if score:
            candidates.append((score, c))
    return sorted(candidates, key=lambda x: (-x[0], str(x[1])))[0][1] if candidates else None


def _v292_env_trend_explanation(df, gei_name, selected_date=None):
    """원환경 추세가 존재할 때 계절/월 변화와 결합한 안전한 자동설명 문장을 생성합니다."""
    col = _v292_find_raw_env_column(df, gei_name)
    if col is None or "조사일자" not in df.columns:
        return None
    temp = df[["조사일자", col]].copy()
    temp["조사일자"] = pd.to_datetime(temp["조사일자"], errors="coerce")
    temp[col] = pd.to_numeric(temp[col], errors="coerce")
    temp = temp.dropna().sort_values("조사일자")
    if selected_date is not None:
        temp = temp[temp["조사일자"] <= pd.to_datetime(selected_date)]
    if len(temp) < 6:
        return None
    # 최근 구간과 그 직전 구간의 중앙값 비교. 조사주기가 주간이어도 안정적으로 동작하도록 행수 기반 사용.
    n = max(3, min(6, len(temp)//3))
    recent = temp.tail(n)
    previous = temp.iloc[max(0, len(temp)-2*n):len(temp)-n]
    if len(previous) < 2:
        previous = temp.head(n)
    a = float(previous[col].median())
    b = float(recent[col].median())
    delta = b - a
    start_month = int(previous["조사일자"].iloc[0].month)
    end_month = int(recent["조사일자"].iloc[-1].month)
    unit = ""
    if gei_name.startswith("온도"): unit = "℃"
    elif gei_name.startswith("습도"): unit = "%"
    elif gei_name.startswith("CO"): unit = " ppm"
    elif gei_name.startswith("일사"): unit = ""
    direction = "상승" if delta > 0 else ("하락" if delta < 0 else "유지")
    return {
        "column": col, "previous": a, "recent": b, "delta": delta,
        "start_month": start_month, "end_month": end_month,
        "direction": direction, "unit": unit,
    }


def _v292_fit_raw_balance_attribution(gei_df, target_col, model_name="RandomForest"):
    """줄기굵기/화방높이는 절대값을 직접 모델링해 높은 값=영양, 낮은 값=생식 방향 SHAP을 계산합니다."""
    component_map = _v291_available_component_geis(gei_df)
    if len(component_map) < 2 or target_col not in gei_df.columns:
        return pd.DataFrame(), None, component_map
    cols = ["조사일자", target_col] + list(component_map.values())
    d = gei_df[cols].copy()
    d["조사일자"] = pd.to_datetime(d["조사일자"], errors="coerce")
    rename = {actual: canonical for canonical, actual in component_map.items()}
    d = d.rename(columns=rename)
    feats = list(component_map.keys())
    for c in feats + [target_col]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["조사일자", target_col] + feats).sort_values("조사일자").reset_index(drop=True)
    if len(d) < 6:
        return pd.DataFrame(), None, component_map
    X = d[feats]
    y = d[target_col]
    model = _v291_make_attribution_model(model_name, len(d))
    model.fit(X, y)
    pred = np.asarray(model.predict(X), dtype=float)
    sv, base = _v291_tree_shap_values(model, X)
    if sv.shape != X.shape:
        return pd.DataFrame(), model, component_map
    rows = []
    for i in range(len(d)):
        state, ref, rule = _v292_classify_balance(target_col, d.loc[i, target_col])
        for j, env in enumerate(feats):
            shapv = float(sv[i, j])
            if shapv > 0:
                directional = "영양생장 방향 기여"
            elif shapv < 0:
                directional = "생식생장 방향 기여"
            else:
                directional = "중립"
            rows.append({
                "조사일자": d.loc[i, "조사일자"],
                "지표": target_col,
                "실측값": float(d.loc[i, target_col]),
                "생장상태": state,
                "기준값": ref,
                "판정구간": rule,
                "환경": env,
                "개별 GEI": float(d.loc[i, env]),
                "Local SHAP": shapv,
                "방향기여": directional,
                "예측값": float(pred[i]),
                "기준예측값": float(base) if np.isfinite(base) else np.nan,
            })
    return pd.DataFrame(rows), model, component_map


def _v292_build_direction_summary(long_df, target_col, target_kind):
    if not isinstance(long_df, pd.DataFrame) or long_df.empty:
        return pd.DataFrame()
    rows = []
    if target_kind in ["stem", "truss"]:
        for state, sign, label in [("영양생장", 1, "영양생장"), ("생식생장", -1, "생식생장")]:
            sub = long_df[long_df["생장상태"] == state].copy()
            if sub.empty:
                continue
            for env, g in sub.groupby("환경"):
                vals = pd.to_numeric(g["Local SHAP"], errors="coerce")
                if sign > 0:
                    directional = vals.clip(lower=0)
                    hit = (vals > 0).mean() * 100
                else:
                    directional = (-vals.clip(upper=0))
                    hit = (vals < 0).mean() * 100
                rows.append({
                    "반응/상태": label,
                    "환경": env,
                    "평균 방향기여": float(directional.mean()),
                    "방향일치 빈도(%)": float(hit),
                    "평균 |SHAP|": float(vals.abs().mean()),
                    "관측수": int(len(g)),
                })
    else:
        # v29.1 long_df 구조: 실제 반응과 Local SHAP 사용
        resp = pd.to_numeric(long_df.get("실제 반응", np.nan), errors="coerce")
        for label, mask, sign in [
            ("증가", resp > 0, 1),
            ("감소", resp < 0, -1),
        ]:
            sub = long_df[mask].copy()
            if sub.empty:
                continue
            for env, g in sub.groupby("환경"):
                vals = pd.to_numeric(g["Local SHAP"], errors="coerce")
                directional = vals.clip(lower=0) if sign > 0 else (-vals.clip(upper=0))
                hit = ((vals > 0) if sign > 0 else (vals < 0)).mean() * 100
                rows.append({
                    "반응/상태": label,
                    "환경": env,
                    "평균 방향기여": float(directional.mean()),
                    "방향일치 빈도(%)": float(hit),
                    "평균 |SHAP|": float(vals.abs().mean()),
                    "관측수": int(len(g)),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["순위"] = out.groupby("반응/상태")["평균 방향기여"].rank(method="dense", ascending=False).astype(int)
        out = out.sort_values(["반응/상태", "순위", "평균 |SHAP|"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def _v292_reason_sentence(gei_df, top_env, target_col, response_desc, selected_date, shap_value):
    trend = _v292_env_trend_explanation(gei_df, top_env, selected_date)
    env_label = top_env.replace(" GEI", "")
    if trend is None:
        return (
            f"{target_col}의 {response_desc} 방향에서 <b>{top_env}</b>가 가장 큰 모델 기여를 보였습니다 "
            f"(Local SHAP {shap_value:+.3f}). 다만 원환경 시계열 컬럼이 충분하지 않아 실제 {env_label} 값의 상승/하락 원인은 자동 확정하지 않았습니다."
        )
    d = trend["delta"]
    unit = trend["unit"]
    month_txt = f"{trend['start_month']}월→{trend['end_month']}월"
    if top_env.startswith("온도"):
        if d < 0:
            cause = f"{month_txt} 진행 중 원온도 지표({trend['column']}) 중앙값이 {trend['previous']:.2f}{unit}에서 {trend['recent']:.2f}{unit}로 낮아졌습니다"
            interpretation = "계절적 저온화와 함께 생장속도가 둔화된 패턴과 일치합니다"
        else:
            cause = f"{month_txt} 진행 중 원온도 지표({trend['column']}) 중앙값이 {trend['previous']:.2f}{unit}에서 {trend['recent']:.2f}{unit}로 높아졌습니다"
            interpretation = "고온 노출 증가와 연결된 반응 가능성을 우선 확인할 수 있습니다"
    else:
        cause = f"{month_txt} 동안 {trend['column']} 중앙값이 {trend['previous']:.2f}{unit}에서 {trend['recent']:.2f}{unit}로 {trend['direction']}했습니다"
        interpretation = f"이 원환경 추세와 {top_env}의 {response_desc} 방향 모델기여가 함께 관찰되었습니다"
    return (
        f"{target_col}의 {response_desc} 방향에서 <b>{top_env}</b>가 가장 큰 모델 기여를 보였습니다 "
        f"(Local SHAP {shap_value:+.3f}). {cause}. 따라서 {interpretation}. "
        f"이는 관측 시계열과 모델 attribution을 결합한 해석이며 인과효과를 직접 증명하지는 않습니다."
    )


def render_tomato_growth_balance_and_direction_attribution_v292():
    gei_df = st.session_state.get("gei_growth_dataset", pd.DataFrame())
    if not isinstance(gei_df, pd.DataFrame) or gei_df.empty:
        return
    if len(_v291_available_component_geis(gei_df)) < 2:
        return
    targets = _v292_expand_target_candidates(gei_df)
    if not targets:
        return

    render_stylish_section(
        "🍅 완숙토마토 생장균형 판정 + 환경 GEI 방향기여 분석",
        "줄기굵기·화방높이는 절대값 기준으로 생식/평균/영양 생장을 판정하고, 다른 생육·수확지표는 NGR·변화량의 증가/감소 방향을 기준으로 온도·습도·CO₂·일사량 GEI가 어느 방향에 기여했는지 분석합니다.",
        kicker="TOMATO GROWTH BALANCE",
    )

    st.markdown(
        """
        <div class="xai-insight-card">
        <b>완숙토마토 생장균형 기준</b><br>
        • <b>줄기굵기</b>: ≤10 mm = 생식생장 · &gt;10~&lt;13 mm = 평균생장 · ≥13 mm = 영양생장 · 중심 기준 12 mm<br>
        • <b>화방높이</b>: &lt;15 cm = 생식생장 · 15~&lt;20 cm = 평균생장 · ≥20 cm = 영양생장 · 중심 기준 18 cm<br>
        • 그 외 지표(초장·엽장·엽폭·엽수·생장길이 등)는 기존 NGR/변화량 방식으로 <b>증가/감소</b>를 판단합니다.<br>
        • 수확수·수확중량·평균중량도 증가/감소 방향별 환경 GEI 기여도를 계산합니다.<br>
        <b>주의:</b> 사용자 제공 기준을 그대로 구현했으며, 품종·생육단계·조사 위치가 다르면 기준값은 별도 검증이 필요합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.35, 1.2, 1.0])
    with c1:
        target_col = st.selectbox("생장/수확 방향 분석 지표", targets, key="tomato_balance_target_v292")
    kind = _v292_target_kind(target_col)
    with c2:
        if kind in ["stem", "truss"]:
            response_basis = "원자료(생장균형 절대값)"
            st.selectbox("반응 판정 기준", [response_basis], index=0, disabled=True, key="tomato_balance_basis_v292_fixed")
        else:
            default_options = ["생육단계 NGR", "월평균 NGR", "조사간 변화량", "원자료"]
            if kind in ["harvest_count", "harvest_weight", "avg_weight"]:
                default_idx = 2
            else:
                default_idx = 0
            response_basis = st.selectbox("반응 판정 기준", default_options, index=default_idx, key="tomato_balance_basis_v292")
    with c3:
        model_name = st.selectbox("방향 기여도 모델", ["RandomForest", "XGBoost", "LGBM", "GradientBoosting"], index=0, key="tomato_balance_model_v292")

    if kind in ["stem", "truss"]:
        long_df, model, component_map = _v292_fit_raw_balance_attribution(gei_df, target_col, model_name=model_name)
        if long_df.empty:
            st.info("줄기굵기/화방높이 생장균형 분석에 필요한 유효 조사일이 부족합니다.")
            return
        # 날짜별 상태표
        state_df = long_df[["조사일자", "실측값", "생장상태", "기준값", "판정구간", "예측값"]].drop_duplicates("조사일자").sort_values("조사일자")
        date_options = state_df["조사일자"].tolist()
        selected_date = st.selectbox("상세 조사일", date_options, index=len(date_options)-1, format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"), key="tomato_balance_date_v292")
        day = long_df[pd.to_datetime(long_df["조사일자"]) == pd.to_datetime(selected_date)].copy()
        day = day.sort_values("Local SHAP", ascending=False)
        info = state_df[pd.to_datetime(state_df["조사일자"]) == pd.to_datetime(selected_date)].iloc[0]
        state = str(info["생장상태"])
        ref = float(info["기준값"])
        val = float(info["실측값"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(target_col, f"{val:.2f}")
        m2.metric("생장균형 상태", state)
        m3.metric("중심 기준", f"{ref:.1f}{' mm' if kind=='stem' else ' cm'}")
        m4.metric("판정구간", str(info["판정구간"]))

        # 시계열 상태 그래프
        render_panel_label("생장균형 상태 시계열")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=state_df["조사일자"], y=state_df["실측값"], mode="lines+markers", name=target_col))
        if kind == "stem":
            fig.add_hrect(y0=min(float(state_df["실측값"].min()), 0), y1=10, opacity=0.08, line_width=0, annotation_text="생식")
            fig.add_hrect(y0=10, y1=13, opacity=0.08, line_width=0, annotation_text="평균")
            fig.add_hrect(y0=13, y1=max(float(state_df["실측값"].max()), 14), opacity=0.08, line_width=0, annotation_text="영양")
            fig.add_hline(y=12, line_dash="dot", annotation_text="기준 12 mm")
            ytitle = "줄기굵기 (mm)"
        else:
            fig.add_hrect(y0=min(float(state_df["실측값"].min()), 0), y1=15, opacity=0.08, line_width=0, annotation_text="생식")
            fig.add_hrect(y0=15, y1=20, opacity=0.08, line_width=0, annotation_text="평균")
            fig.add_hrect(y0=20, y1=max(float(state_df["실측값"].max()), 21), opacity=0.08, line_width=0, annotation_text="영양")
            fig.add_hline(y=18, line_dash="dot", annotation_text="기준 18 cm")
            ytitle = "화방높이 (cm)"
        fig.update_layout(height=360, yaxis_title=ytitle, xaxis_title="조사일자", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.82)")
        display_plotly(fig)

        left, right = st.columns(2, gap="large")
        with left:
            render_panel_label(f"{state} 방향의 환경 GEI Local SHAP")
            plot_day = day.copy()
            fig2 = go.Figure(go.Bar(x=plot_day["환경"], y=plot_day["Local SHAP"], text=[f"{x:+.3f}" for x in plot_day["Local SHAP"]], textposition="outside"))
            fig2.add_hline(y=0, line_dash="dash")
            fig2.update_layout(height=360, yaxis_title="Local SHAP (+:영양 / -:생식)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.82)")
            display_plotly(fig2)
        with right:
            summary = _v292_build_direction_summary(long_df, target_col, kind)
            render_panel_label("전체 작기 영양/생식 방향 기여도 순위")
            st.dataframe(summary.round(4), use_container_width=True, hide_index=True, height=360)

        # 선택 상태에서 방향과 일치하는 최상위 원인
        if state == "영양생장":
            cand = day[pd.to_numeric(day["Local SHAP"], errors="coerce") > 0].sort_values("Local SHAP", ascending=False)
            desc = "영양생장"
        elif state == "생식생장":
            cand = day[pd.to_numeric(day["Local SHAP"], errors="coerce") < 0].sort_values("Local SHAP", ascending=True)
            desc = "생식생장"
        else:
            cand = day.reindex(day["Local SHAP"].abs().sort_values(ascending=False).index)
            desc = "평균생장 상태 변화"
        if not cand.empty:
            top = cand.iloc[0]
            reason = _v292_reason_sentence(gei_df, str(top["환경"]), target_col, desc, selected_date, float(top["Local SHAP"]))
            st.markdown(f'<div class="xai-insight-card"><b>자동 원인 해석</b><br>{reason}</div>', unsafe_allow_html=True)

        with st.expander("📋 조사일별 생장균형·환경기여 상세 데이터", expanded=False):
            st.dataframe(long_df.sort_values(["조사일자", "Local SHAP"]).round(4), use_container_width=True, hide_index=True, height=520)
        st.session_state["tomato_balance_attribution_v292"] = long_df
        st.session_state["tomato_balance_summary_v292"] = _v292_build_direction_summary(long_df, target_col, kind)
        return

    # 기타 지표: v29.1 원인분해 결과를 활용해 증가/감소 방향 분석
    selected_weights = st.session_state.get("gei_selected_weights", {})
    result = build_integrated_gei_growth_factor_attribution(
        gei_df=gei_df,
        target_col=target_col,
        response_basis=response_basis,
        model_name=model_name,
        selected_weights=selected_weights,
    )
    if len(result) != 7:
        st.info("선택 지표의 증가/감소 환경기여 분석에 필요한 데이터가 부족합니다.")
        return
    wide_df, long_df, component_map, response_label, model, summary_df, meta = result
    if long_df.empty:
        st.info("선택 지표의 증가/감소 환경기여 분석에 필요한 유효 조사일이 부족합니다.")
        return
    date_resp = long_df[["조사일자", "실제 반응"]].drop_duplicates("조사일자").sort_values("조사일자")
    selected_date = st.selectbox("상세 조사일", date_resp["조사일자"].tolist(), index=len(date_resp)-1, format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"), key="tomato_response_date_v292")
    day = long_df[pd.to_datetime(long_df["조사일자"]) == pd.to_datetime(selected_date)].copy()
    actual = float(pd.to_numeric(day["실제 반응"], errors="coerce").iloc[0])
    state = "증가" if actual > 0 else ("감소" if actual < 0 else "유지")

    a,b,c = st.columns(3)
    a.metric("선택 지표", target_col)
    b.metric("실제 반응", f"{actual:+.2f}")
    c.metric("반응 판정", state)

    left, right = st.columns(2, gap="large")
    with left:
        render_panel_label(f"{target_col} {state} 방향 환경 GEI 기여")
        plot_day = day.sort_values("Local SHAP")
        f = go.Figure(go.Bar(x=plot_day["환경"], y=plot_day["Local SHAP"], text=[f"{x:+.3f}" for x in plot_day["Local SHAP"]], textposition="outside"))
        f.add_hline(y=0, line_dash="dash")
        f.update_layout(height=360, yaxis_title="Local SHAP (+:증가 / -:감소)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.82)")
        display_plotly(f)
    with right:
        direction_summary = _v292_build_direction_summary(long_df, target_col, kind)
        render_panel_label("전체 작기 증가/감소 방향 기여도 순위")
        st.dataframe(direction_summary.round(4), use_container_width=True, hide_index=True, height=360)

    if state == "감소":
        cand = day[pd.to_numeric(day["Local SHAP"], errors="coerce") < 0].sort_values("Local SHAP", ascending=True)
    elif state == "증가":
        cand = day[pd.to_numeric(day["Local SHAP"], errors="coerce") > 0].sort_values("Local SHAP", ascending=False)
    else:
        cand = day.reindex(day["Local SHAP"].abs().sort_values(ascending=False).index)
    if not cand.empty:
        top = cand.iloc[0]
        reason = _v292_reason_sentence(gei_df, str(top["환경"]), target_col, state, selected_date, float(top["Local SHAP"]))
        st.markdown(f'<div class="xai-insight-card"><b>자동 원인 해석</b><br>{reason}</div>', unsafe_allow_html=True)

    # 수확특화 안내
    if kind in ["harvest_count", "harvest_weight", "avg_weight"]:
        st.caption("수확지표는 증가/감소 방향별 Local SHAP을 해석합니다. 수확은 환경반응 지연이 존재할 수 있으므로 Lag SHAP 및 1~7주 window 결과와 함께 확인하는 것을 권장합니다.")

    with st.expander("📋 조사일별 증가/감소·환경기여 상세 데이터", expanded=False):
        st.dataframe(long_df.sort_values(["조사일자", "Local SHAP"]).round(4), use_container_width=True, hide_index=True, height=520)

    st.session_state["tomato_response_attribution_v292"] = long_df
    st.session_state["tomato_response_summary_v292"] = _v292_build_direction_summary(long_df, target_col, kind)


try:
    render_tomato_growth_balance_and_direction_attribution_v292()
except Exception as _v292_runtime_error:
    st.warning(f"완숙토마토 생장균형·방향기여 분석 모듈 처리 오류: {_v292_runtime_error}")


# =============================================================
# v29.3 Calendar × Phenology 이중정렬 / 조건부 Reference 모듈
# =============================================================
V293_VERSION = "29.3"


def _v293_auto_season(month):
    try:
        m = int(month)
    except Exception:
        return "미지정"
    if m in (12, 1, 2):
        return "겨울"
    if m in (3, 4, 5):
        return "봄"
    if m in (6, 7, 8):
        return "여름"
    if m in (9, 10, 11):
        return "가을"
    return "미지정"


def _v293_wap_band(wap, width=4):
    try:
        w = int(np.floor(float(wap)))
    except Exception:
        return "미지정"
    if w < 1:
        return "정식전/초기"
    start = ((w - 1) // int(width)) * int(width) + 1
    end = start + int(width) - 1
    return f"WAP {start}-{end}"


def _v293_circular_month_distance(a, b):
    try:
        d = abs(int(a) - int(b)) % 12
        return min(d, 12 - d)
    except Exception:
        return np.nan


def _v293_norm_distance(current, reference, scale):
    c = _kb_to_float(current)
    r = _kb_to_float(reference)
    s = _kb_to_float(scale)
    if not (np.isfinite(c) and np.isfinite(r) and np.isfinite(s) and s > 0):
        return np.nan
    return abs(c - r) / s


def _v293_find_columns(df, include_terms, exclude_terms=()):
    if not isinstance(df, pd.DataFrame):
        return []
    result = []
    for col in df.columns:
        low = str(col).lower().replace(" ", "")
        if any(str(t).lower().replace(" ", "") in low for t in include_terms):
            if not any(str(t).lower().replace(" ", "") in low for t in exclude_terms):
                result.append(col)
    return result


def _v293_build_daily_sensor_context(sensor_df, date_col, gdd_temp_col, external_temp_col,
                                     external_solar_col, planting_date, base_temp=10.0):
    """원센서에서 일평균/GDD/7일 외기 맥락을 생성합니다."""
    if not isinstance(sensor_df, pd.DataFrame) or sensor_df.empty or date_col not in sensor_df.columns:
        return pd.DataFrame(), {"gdd_source": "없음", "coverage_start": None, "coverage_end": None}
    work = sensor_df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    if work.empty:
        return pd.DataFrame(), {"gdd_source": "없음", "coverage_start": None, "coverage_end": None}
    work["일자"] = work[date_col].dt.normalize()
    use_cols = []
    for c in [gdd_temp_col, external_temp_col, external_solar_col]:
        if c is not None and c in work.columns and c not in use_cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")
            use_cols.append(c)
    if not use_cols:
        return pd.DataFrame(), {"gdd_source": "없음", "coverage_start": None, "coverage_end": None}
    daily = work.groupby("일자", as_index=False)[use_cols].mean(numeric_only=True)
    daily = daily.sort_values("일자").reset_index(drop=True)
    planting_ts = pd.Timestamp(planting_date).normalize()
    daily["DAP_day"] = (daily["일자"] - planting_ts).dt.days
    daily = daily[daily["DAP_day"] >= 0].copy()
    if daily.empty:
        return daily, {"gdd_source": str(gdd_temp_col or "없음"), "coverage_start": None, "coverage_end": None}
    if gdd_temp_col is not None and gdd_temp_col in daily.columns:
        t = pd.to_numeric(daily[gdd_temp_col], errors="coerce")
        daily["GDD_day"] = np.maximum(t - float(base_temp), 0.0)
        daily["GDD_cum"] = daily["GDD_day"].fillna(0.0).cumsum()
    else:
        daily["GDD_day"] = np.nan
        daily["GDD_cum"] = np.nan
    if external_temp_col is not None and external_temp_col in daily.columns:
        daily["외기온_7일평균"] = pd.to_numeric(daily[external_temp_col], errors="coerce").rolling(7, min_periods=1).mean()
    else:
        daily["외기온_7일평균"] = np.nan
    if external_solar_col is not None and external_solar_col in daily.columns:
        daily["외부일사_7일평균"] = pd.to_numeric(daily[external_solar_col], errors="coerce").rolling(7, min_periods=1).mean()
    else:
        daily["외부일사_7일평균"] = np.nan
    info = {
        "gdd_source": str(gdd_temp_col or "없음"),
        "coverage_start": daily["일자"].min().date().isoformat() if not daily.empty else None,
        "coverage_end": daily["일자"].max().date().isoformat() if not daily.empty else None,
    }
    return daily, info


def _v293_prepare_response_by_date(gei_df, target_col):
    if not isinstance(gei_df, pd.DataFrame) or gei_df.empty or not target_col or target_col not in gei_df.columns:
        return pd.DataFrame()
    gei_feature = next((c for c in ["통합 GEI", "온도 GEI", "습도 GEI", "CO₂ GEI", "일사량 GEI"] if c in gei_df.columns), None)
    if gei_feature is None:
        return pd.DataFrame()
    try:
        resp, _ = build_gei_growth_response_curve(
            gei_df=gei_df,
            gei_feature=gei_feature,
            target_col=target_col,
            baseline_mode="생육단계 기대 변화량 대비(최종 권장)",
            stable_band_pct=2.0,
            danger_pct=-10.0,
        )
        if resp is None or resp.empty or "조사일자" not in resp.columns:
            return pd.DataFrame()
        cols = [c for c in ["조사일자", "변화량", "일평균 변화량", "7일환산 변화량", "반응률(%)"] if c in resp.columns]
        out = resp[cols].copy()
        out["조사일자"] = pd.to_datetime(out["조사일자"], errors="coerce").dt.normalize()
        return out.dropna(subset=["조사일자"]).drop_duplicates("조사일자", keep="last")
    except Exception:
        return pd.DataFrame()


def _v293_build_alignment_table(gei_df, yield_df, yield_date_col, planting_date,
                                daily_context, stage_col=None, ngr_target=None):
    if not isinstance(gei_df, pd.DataFrame) or gei_df.empty or "조사일자" not in gei_df.columns:
        return pd.DataFrame()
    out = gei_df.copy()
    out["조사일자"] = pd.to_datetime(out["조사일자"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["조사일자"]).sort_values("조사일자").reset_index(drop=True)
    planting_ts = pd.Timestamp(planting_date).normalize()
    out["DAP"] = (out["조사일자"] - planting_ts).dt.days
    out["WAP"] = np.floor(out["DAP"] / 7.0).astype("Int64") + 1
    out["달력월"] = out["조사일자"].dt.month.astype("Int64")
    out["달력계절"] = out["달력월"].apply(_v293_auto_season)
    out["WAP구간"] = out["WAP"].apply(_v293_wap_band)

    if isinstance(yield_df, pd.DataFrame) and yield_date_col in yield_df.columns and stage_col and stage_col in yield_df.columns:
        stage = yield_df[[yield_date_col, stage_col]].copy()
        stage[yield_date_col] = pd.to_datetime(stage[yield_date_col], errors="coerce").dt.normalize()
        stage = stage.dropna(subset=[yield_date_col]).drop_duplicates(yield_date_col, keep="last")
        stage = stage.rename(columns={yield_date_col: "조사일자", stage_col: "관측 생육단계"})
        out = out.merge(stage, on="조사일자", how="left")
    else:
        out["관측 생육단계"] = np.nan
    out["생육단계"] = out["관측 생육단계"].where(out["관측 생육단계"].notna(), out["WAP구간"])

    if isinstance(daily_context, pd.DataFrame) and not daily_context.empty:
        daily = daily_context.copy().sort_values("일자")
        keep = [c for c in ["일자", "GDD_cum", "외기온_7일평균", "외부일사_7일평균"] if c in daily.columns]
        out = pd.merge_asof(
            out.sort_values("조사일자"),
            daily[keep].sort_values("일자"),
            left_on="조사일자", right_on="일자", direction="backward"
        )
        if "일자" in out.columns:
            out = out.drop(columns=["일자"])
    else:
        out["GDD_cum"] = np.nan
        out["외기온_7일평균"] = np.nan
        out["외부일사_7일평균"] = np.nan

    response = _v293_prepare_response_by_date(gei_df, ngr_target)
    if not response.empty:
        out = out.merge(response, on="조사일자", how="left", suffixes=("", "_NGR"))
    return out.sort_values("조사일자").reset_index(drop=True)


def _v293_phase_profile(aligned_df):
    if not isinstance(aligned_df, pd.DataFrame) or aligned_df.empty:
        return []
    metrics = [
        "온도 GEI", "습도 GEI", "CO₂ GEI", "일사량 GEI", "통합 GEI",
        "반응률(%)", "GDD_cum", "외기온_7일평균", "외부일사_7일평균"
    ]
    metrics = [c for c in metrics if c in aligned_df.columns]
    rows = []
    for phase, g in aligned_df.groupby("WAP구간", dropna=False):
        if str(phase) in ("nan", "미지정"):
            continue
        row = {"WAP구간": str(phase), "관측수": int(len(g))}
        wap_vals = pd.to_numeric(g.get("WAP"), errors="coerce").dropna()
        if not wap_vals.empty:
            row["WAP중앙"] = float(wap_vals.median())
        seasons = g.get("달력계절", pd.Series(dtype=str)).dropna().astype(str)
        if not seasons.empty:
            row["주계절"] = seasons.mode().iloc[0]
        stages = g.get("생육단계", pd.Series(dtype=str)).dropna().astype(str)
        if not stages.empty:
            row["주생육단계"] = stages.mode().iloc[0]
        for m in metrics:
            s = pd.to_numeric(g[m], errors="coerce").dropna()
            if not s.empty:
                row[f"{m} 평균"] = float(s.mean())
                row[f"{m} Median"] = float(s.median())
        rows.append(row)
    return rows


def _v293_phase_lag_summary(aligned_df):
    matrix = np.asarray(st.session_state.get("v293_lag_sample_abs_matrix", np.empty((0, 0))), dtype=float)
    lags = list(st.session_state.get("v293_lag_values", []))
    dates = pd.to_datetime(st.session_state.get("v293_lag_test_dates", []), errors="coerce")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0 or len(lags) != matrix.shape[1] or len(dates) != matrix.shape[0]:
        return []
    if not isinstance(aligned_df, pd.DataFrame) or aligned_df.empty:
        return []
    key = aligned_df[["조사일자", "WAP구간", "달력계절", "생육단계"]].copy()
    key["조사일자"] = pd.to_datetime(key["조사일자"], errors="coerce").dt.normalize()
    sample = pd.DataFrame({"조사일자": pd.to_datetime(dates, errors="coerce").normalize()})
    sample = sample.merge(key.drop_duplicates("조사일자", keep="last"), on="조사일자", how="left")
    result = []
    for (phase, season), idx in sample.groupby(["WAP구간", "달력계절"], dropna=False).groups.items():
        indices = list(idx)
        if len(indices) < 1:
            continue
        sub = matrix[indices, :]
        mean_abs = np.nanmean(sub, axis=0)
        if not np.isfinite(mean_abs).any():
            continue
        peak_pos = int(np.nanargmax(mean_abs))
        total = float(np.nansum(mean_abs))
        result.append({
            "WAP구간": str(phase),
            "계절": str(season),
            "Lag SHAP Target": str(st.session_state.get("lag_shap_target", "")),
            "Peak Lag(주)": int(lags[peak_pos]),
            "Peak 비중(%)": float(mean_abs[peak_pos] / total * 100.0) if total > 0 else np.nan,
            "평가샘플수": int(len(indices)),
        })
    return result


def _v293_cycle_context_from_alignment(aligned_df, planting_date, base_temp, gdd_info, ext_temp_col, ext_solar_col):
    rec = {
        "정식일": pd.Timestamp(planting_date).date().isoformat(),
        "정식월": int(pd.Timestamp(planting_date).month),
        "정식계절(자동)": _v293_auto_season(pd.Timestamp(planting_date).month),
        "GDD Base(℃)": float(base_temp),
        "GDD 계산온도컬럼": str(gdd_info.get("gdd_source", "")),
        "외기온 컬럼": str(ext_temp_col or ""),
        "외부일사 컬럼": str(ext_solar_col or ""),
        "센서커버리지 시작": gdd_info.get("coverage_start"),
        "센서커버리지 종료": gdd_info.get("coverage_end"),
    }
    if isinstance(aligned_df, pd.DataFrame) and not aligned_df.empty:
        for c, key, agg in [
            ("DAP", "최종 DAP", "max"), ("WAP", "최종 WAP", "max"), ("GDD_cum", "최종 GDD", "max"),
            ("외기온_7일평균", "평균 외기온(℃)", "mean"), ("외부일사_7일평균", "평균 외부일사", "mean")
        ]:
            if c in aligned_df.columns:
                s = pd.to_numeric(aligned_df[c], errors="coerce").dropna()
                if not s.empty:
                    rec[key] = float(s.max() if agg == "max" else s.mean())
        dap = _kb_to_float(rec.get("최종 DAP", np.nan))
        gdd = _kb_to_float(rec.get("최종 GDD", np.nan))
        rec["평균 GDD/day"] = float(gdd / max(dap, 1.0)) if np.isfinite(gdd) and np.isfinite(dap) else np.nan
        rec["생육단계 프로파일"] = _v293_phase_profile(aligned_df)
        rec["Phase×Season Lag SHAP"] = _v293_phase_lag_summary(aligned_df)
    return rec


def _v293_save_context_into_cycle(db_path, cycle_id, context_record):
    records = load_crop_cycle_records(db_path)
    base = next((dict(r) for r in records if str(r.get("작기 ID", "")) == str(cycle_id)), None)
    if base is None:
        base = {
            "작기 ID": str(cycle_id), "농가 ID": "", "작물": "완숙토마토", "품종": "",
            "계절": context_record.get("정식계절(자동)", "미지정"),
            "재배 시작일": context_record.get("정식일", ""), "재배 종료일": ""
        }
    base.update(_kb_json_safe(context_record))
    base["Calendar×Phenology 버전"] = V293_VERSION
    save_crop_cycle_record(db_path, base)
    return base


def _v293_context_similarity(current_ctx, historical_record, weights):
    distances = []
    used_weights = []
    details = []
    # 정식월: 최대 6개월 거리 -> 0~1
    md = _v293_circular_month_distance(current_ctx.get("정식월"), historical_record.get("정식월"))
    if np.isfinite(md):
        distances.append(md / 6.0); used_weights.append(weights.get("month", 0.15)); details.append(("정식월", md / 6.0))
    # WAP 진행속도/작기 길이는 너무 강하게 쓰지 않고 GDD/day로 대체
    pairs = [
        ("평균 GDD/day", "gdd", max(abs(_kb_to_float(current_ctx.get("평균 GDD/day"), 10.0)) * 0.25, 1.0)),
        ("평균 외기온(℃)", "temp", 4.0),
        ("평균 외부일사", "solar", max(abs(_kb_to_float(current_ctx.get("평균 외부일사"), 100.0)) * 0.30, 1.0)),
    ]
    for key, wkey, scale in pairs:
        d = _v293_norm_distance(current_ctx.get(key), historical_record.get(key), scale)
        if np.isfinite(d):
            distances.append(d); used_weights.append(weights.get(wkey, 0.25)); details.append((key, d))
    cur_season = str(current_ctx.get("정식계절(자동)", ""))
    ref_season = str(historical_record.get("정식계절(자동)", historical_record.get("계절", ""))).replace("작기", "")
    if cur_season and ref_season and cur_season != "미지정" and ref_season != "미지정":
        distances.append(0.0 if cur_season in ref_season or ref_season in cur_season else 1.0)
        used_weights.append(weights.get("season", 0.15)); details.append(("계절", distances[-1]))
    if not distances or sum(used_weights) <= 0:
        return np.nan, 0.0, details
    w = np.asarray(used_weights, dtype=float)
    d = np.asarray(distances, dtype=float)
    weighted = float(np.sum(w * d) / np.sum(w))
    similarity = float(100.0 * np.exp(-weighted))
    coverage = float(np.sum(w) / max(sum(weights.values()), 1e-9) * 100.0)
    return similarity, min(coverage, 100.0), details


def _v293_find_similar_top_cycles(records_df, current_ctx, top_percent=30, top_k=5, weights=None):
    if records_df is None or records_df.empty:
        return pd.DataFrame()
    weights = weights or {"month": 0.15, "season": 0.15, "gdd": 0.30, "temp": 0.25, "solar": 0.15}
    ranked = compute_cycle_ranking(records_df, {"yield":0.35,"quality":0.20,"growth":0.20,"gei":0.15,"efficiency":0.10})
    if ranked.empty:
        ranked = records_df.copy()
        ranked["종합 생산성/안정성 점수"] = np.nan
    n_top = max(1, int(np.ceil(len(ranked) * float(top_percent) / 100.0)))
    candidates = ranked.head(n_top).copy()
    rows = []
    for _, row in candidates.iterrows():
        rec = row.to_dict()
        sim, cov, _ = _v293_context_similarity(current_ctx, rec, weights)
        rows.append({
            "작기 ID": rec.get("작기 ID", ""), "농가 ID": rec.get("농가 ID", ""), "품종": rec.get("품종", ""),
            "정식일": rec.get("정식일", rec.get("재배 시작일", "")),
            "정식계절": rec.get("정식계절(자동)", rec.get("계절", "")),
            "평균 GDD/day": rec.get("평균 GDD/day", np.nan), "평균 외기온(℃)": rec.get("평균 외기온(℃)", np.nan),
            "평균 외부일사": rec.get("평균 외부일사", np.nan), "종합점수": rec.get("종합 생산성/안정성 점수", np.nan),
            "환경·생육시점 유사도(%)": sim, "조건정보 충족률(%)": cov,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["환경·생육시점 유사도(%)", "종합점수"], ascending=[False, False]).head(int(top_k)).reset_index(drop=True)


def _v293_build_conditional_phase_reference(records, selected_cycle_ids, current_phase):
    rows = []
    for rec in records:
        if str(rec.get("작기 ID", "")) not in set(map(str, selected_cycle_ids)):
            continue
        profiles = rec.get("생육단계 프로파일", [])
        if not isinstance(profiles, list):
            continue
        for p in profiles:
            if isinstance(p, dict) and str(p.get("WAP구간", "")) == str(current_phase):
                row = dict(p); row["작기 ID"] = rec.get("작기 ID", "")
                rows.append(row)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c.endswith(" 평균") and any(k in c for k in ["GEI", "반응률", "GDD", "외기온", "외부일사"])]
    prof = []
    for c in metric_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        prof.append({"지표": c.replace(" 평균", ""), "Q25": float(s.quantile(.25)), "Median": float(s.median()), "Q75": float(s.quantile(.75)), "참조작기수": int(s.size)})
    return df, pd.DataFrame(prof)


def render_calendar_phenology_conditional_reference_v293():
    render_stylish_section(
        "🗓️ Calendar × Phenology 이중정렬 · 조건부 우수작기 Reference",
        "정식 시작일이 서로 다른 작기를 단순 WAP만으로 합치지 않고, DAP/WAP·달력계절·GDD·외기온·외부일사를 동시에 정렬합니다. 현재 작기와 환경·생육시점이 유사한 과거 상위작기만 선택해 조건부 Reference를 생성합니다.",
        kicker="V29.3 CALENDAR × PHENOLOGY",
    )
    gei_df = st.session_state.get("gei_growth_dataset", pd.DataFrame())
    sensor_df = st.session_state.get("v293_sensor_df", pd.DataFrame())
    yield_df = st.session_state.get("v293_yield_df", pd.DataFrame())
    sensor_date_col = st.session_state.get("v293_sensor_date_col")
    yield_date_col = st.session_state.get("v293_yield_date_col")
    indoor_temp_col = st.session_state.get("v293_indoor_temp_col")
    if not isinstance(gei_df, pd.DataFrame) or gei_df.empty:
        st.info("먼저 환경센서와 생육/수확 파일을 업로드하고 GEI 분석을 실행하세요.")
        return

    gei_dates = pd.to_datetime(gei_df["조사일자"], errors="coerce") if "조사일자" in gei_df.columns else pd.Series(dtype="datetime64[ns]")
    default_planting = (gei_dates.min() - pd.Timedelta(days=28)).date() if gei_dates.notna().any() else pd.Timestamp.today().date()
    db_path = st.session_state.get("crop_cycle_db_path", CYCLE_KB_DEFAULT_DB)
    try:
        resolved_db = init_crop_cycle_kb(db_path)
    except Exception as exc:
        st.error(f"작기 DB 연결 오류: {exc}"); return
    records = load_crop_cycle_records(resolved_db)
    records_df = _kb_records_to_dataframe(records)

    t1,t2,t3,t4 = st.columns(4)
    with t1:
        planting_date = st.date_input("정식일", value=default_planting, key="v293_planting_date")
        base_temp = st.number_input("GDD 기준온도(℃)", value=10.0, step=0.5, key="v293_gdd_base")
    sensor_cols = sensor_df.columns.tolist() if isinstance(sensor_df, pd.DataFrame) else []
    none="(없음)"
    ext_temp_candidates = _v293_find_columns(sensor_df, ["외기온","외부온","outdoor","outside","externaltemp"])
    ext_solar_candidates = _v293_find_columns(sensor_df, ["외부일사","외부광","outdoorsolar","outside","externalradiation"])
    with t2:
        ext_temp = st.selectbox("외기온 컬럼", [none]+sensor_cols, index=([none]+sensor_cols).index(ext_temp_candidates[0]) if ext_temp_candidates else 0, key="v293_ext_temp")
        ext_temp = None if ext_temp==none else ext_temp
        ext_solar = st.selectbox("외부일사 컬럼", [none]+sensor_cols, index=([none]+sensor_cols).index(ext_solar_candidates[0]) if ext_solar_candidates else 0, key="v293_ext_solar")
        ext_solar = None if ext_solar==none else ext_solar
    with t3:
        gdd_temp_options=[c for c in [ext_temp, indoor_temp_col] if c is not None]+[c for c in sensor_cols if c not in [ext_temp, indoor_temp_col]]
        gdd_temp_options=list(dict.fromkeys(gdd_temp_options))
        gdd_temp=st.selectbox("GDD 계산 온도", gdd_temp_options if gdd_temp_options else [none], key="v293_gdd_temp")
        gdd_temp=None if gdd_temp==none else gdd_temp
        stage_options=[none]+(yield_df.columns.tolist() if isinstance(yield_df,pd.DataFrame) else [])
        stage_col=st.selectbox("생육단계 컬럼(선택)", stage_options, key="v293_stage_col")
        stage_col=None if stage_col==none else stage_col
    with t4:
        numeric_targets=_kb_numeric_candidates(gei_df)
        ngr_candidates=[c for c in numeric_targets if "GEI" not in str(c) and "누적시간" not in str(c)]
        ngr_target=st.selectbox("Phase NGR 기준 지표", [none]+ngr_candidates, key="v293_ngr_target")
        ngr_target=None if ngr_target==none else ngr_target
        cycle_ids=[str(r.get("작기 ID","")) for r in records if str(r.get("작기 ID",""))]
        cycle_id=st.selectbox("저장/갱신할 작기 ID", ["(새 작기 ID 입력)"]+cycle_ids, key="v293_cycle_pick")
        if cycle_id=="(새 작기 ID 입력)":
            cycle_id=st.text_input("새 작기 ID", value=f"Cycle-{pd.Timestamp.today().year}", key="v293_new_cycle_id")

    daily, gdd_info = _v293_build_daily_sensor_context(sensor_df, sensor_date_col, gdd_temp, ext_temp, ext_solar, planting_date, base_temp)
    aligned = _v293_build_alignment_table(gei_df, yield_df, yield_date_col, planting_date, daily, stage_col=stage_col, ngr_target=ngr_target)
    ctx = _v293_cycle_context_from_alignment(aligned, planting_date, base_temp, gdd_info, ext_temp, ext_solar)

    if aligned.empty:
        st.warning("Calendar × Phenology 정렬 결과를 만들 수 없습니다."); return
    current_phase = str(aligned["WAP구간"].dropna().iloc[-1]) if aligned["WAP구간"].notna().any() else "미지정"
    current_stage = str(aligned["생육단계"].dropna().iloc[-1]) if aligned["생육단계"].notna().any() else current_phase

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("현재 DAP", f"{int(_kb_to_float(ctx.get('최종 DAP',0),0))}일")
    m2.metric("현재 WAP", f"{int(_kb_to_float(ctx.get('최종 WAP',0),0))}주")
    m3.metric("누적 GDD", f"{_kb_to_float(ctx.get('최종 GDD',np.nan)):.1f}" if np.isfinite(_kb_to_float(ctx.get('최종 GDD',np.nan))) else "N/A")
    m4.metric("현재 Phase", current_phase)
    m5.metric("정식계절", ctx.get("정식계절(자동)","미지정"))
    st.caption(f"현재 생육단계: {current_stage} · GDD source: {gdd_info.get('gdd_source','없음')} · 센서 coverage: {gdd_info.get('coverage_start')} ~ {gdd_info.get('coverage_end')}")

    left,right=st.columns([1.35,0.65], gap="large")
    with left:
        render_panel_label("조사일 × DAP/WAP × Calendar × Thermal time 정렬")
        show_cols=[c for c in ["조사일자","DAP","WAP","WAP구간","생육단계","달력월","달력계절","GDD_cum","외기온_7일평균","외부일사_7일평균","온도 GEI","습도 GEI","CO₂ GEI","일사량 GEI","통합 GEI","반응률(%)"] if c in aligned.columns]
        st.dataframe(aligned[show_cols].round(3), use_container_width=True, hide_index=True, height=420)
    with right:
        render_panel_label("현재 작기 Context 요약")
        ctx_view=pd.DataFrame([{"항목":k,"값":v} for k,v in ctx.items() if k not in ["생육단계 프로파일","Phase×Season Lag SHAP"]])
        st.dataframe(ctx_view, use_container_width=True, hide_index=True, height=420)

    phase_df=pd.DataFrame(ctx.get("생육단계 프로파일",[]))
    lag_phase_df=pd.DataFrame(ctx.get("Phase×Season Lag SHAP",[]))
    c1,c2=st.columns(2, gap="large")
    with c1:
        render_panel_label("WAP 4주 구간별 GEI·NGR·계절환경 Profile")
        if phase_df.empty: st.info("Phase profile 데이터가 부족합니다.")
        else: st.dataframe(phase_df.round(3), use_container_width=True, hide_index=True, height=340)
    with c2:
        render_panel_label("Phase × Season별 Lag SHAP Peak")
        if lag_phase_df.empty:
            st.info("현재 세션에 날짜가 연결된 Lag SHAP 평가샘플이 없습니다. Lag SHAP 분석을 먼저 실행하면 단계별 요약이 활성화됩니다.")
        else:
            st.dataframe(lag_phase_df.round(3), use_container_width=True, hide_index=True, height=340)

    if st.button("💾 Calendar × Phenology Context를 작기 DB에 저장/업데이트", type="primary", key="v293_save_context"):
        try:
            saved=_v293_save_context_into_cycle(resolved_db, cycle_id, ctx)
            st.success(f"{cycle_id}에 Calendar × Phenology context를 저장했습니다.")
            records=load_crop_cycle_records(resolved_db); records_df=_kb_records_to_dataframe(records)
        except Exception as exc:
            st.error(f"저장 실패: {exc}")

    st.markdown("---")
    render_panel_label("현재 작기와 가장 유사한 과거 상위생산 작기 자동 검색")
    q1,q2,q3=st.columns(3)
    with q1:
        top_percent=st.slider("성과 상위 후보군(%)", 10, 60, 30, 5, key="v293_top_pct")
    with q2:
        top_k=st.slider("유사 Reference 작기 수", 1, 10, 5, 1, key="v293_top_k")
    with q3:
        strict_same_cultivar=st.checkbox("동일 작물·품종 우선 필터", value=True, key="v293_same_cultivar")

    cohort=records_df.copy()
    if not cohort.empty and "작기 ID" in cohort.columns:
        cohort = cohort[cohort["작기 ID"].astype(str) != str(cycle_id)].copy()
    if not cohort.empty and strict_same_cultivar:
        current_rec=next((r for r in records if str(r.get("작기 ID",""))==str(cycle_id)), {})
        crop=current_rec.get("작물", "")
        cultivar=current_rec.get("품종", "")
        if crop and "작물" in cohort.columns:
            cohort=cohort[cohort["작물"].astype(str)==str(crop)]
        if cultivar and "품종" in cohort.columns:
            cohort=cohort[cohort["품종"].astype(str)==str(cultivar)]
    similar=_v293_find_similar_top_cycles(cohort, ctx, top_percent=top_percent, top_k=top_k)
    if similar.empty:
        st.info("비교 가능한 과거 작기가 아직 부족합니다. 여러 작기에 v29.3 context를 저장하면 자동 검색이 활성화됩니다.")
    else:
        st.dataframe(similar.round(3), use_container_width=True, hide_index=True)
        selected_ids=similar["작기 ID"].astype(str).tolist()
        phase_rows, conditional_ref=_v293_build_conditional_phase_reference(records, selected_ids, current_phase)
        r1,r2=st.columns([0.8,1.2], gap="large")
        with r1:
            render_panel_label(f"선택된 유사 상위작기의 {current_phase} 원자료")
            if phase_rows.empty: st.info("선택 작기들에 동일 WAP구간 profile이 아직 저장되지 않았습니다.")
            else: st.dataframe(phase_rows.round(3), use_container_width=True, hide_index=True, height=360)
        with r2:
            render_panel_label(f"조건부 Reference · {current_phase} · Q25 / Median / Q75")
            if conditional_ref.empty:
                st.info("조건부 Reference를 계산할 공통 지표가 부족합니다.")
            else:
                st.dataframe(conditional_ref.round(3), use_container_width=True, hide_index=True, height=360)
                current_row=aligned.iloc[-1]
                diag=[]
                for _,rr in conditional_ref.iterrows():
                    metric=str(rr["지표"])
                    current_col=metric
                    if current_col not in aligned.columns: continue
                    cur=_kb_to_float(current_row.get(current_col,np.nan))
                    if not np.isfinite(cur): continue
                    q25,q50,q75=[_kb_to_float(rr.get(k,np.nan)) for k in ["Q25","Median","Q75"]]
                    state="범위내"
                    if np.isfinite(q25) and cur<q25: state="하단이탈"
                    if np.isfinite(q75) and cur>q75: state="상단이탈"
                    diag.append({"지표":metric,"현재":cur,"Q25":q25,"Median":q50,"Q75":q75,"상태":state,"편차":cur-q50 if np.isfinite(q50) else np.nan})
                if diag:
                    st.markdown("**현재 조사시점 조건부 Reference 진단**")
                    st.dataframe(pd.DataFrame(diag).round(3), use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="xai-insight-card"><b>해석 원칙</b><br>'
        '같은 WAP라도 정식월·계절·GDD·외기온·외부일사가 다르면 동일 Reference로 간주하지 않습니다. '
        '현재 작기와 환경·thermal-time이 유사한 상위생산 작기를 먼저 고른 뒤, 동일 WAP구간의 GEI·NGR 분포를 비교합니다. '
        '따라서 9월 정식 작기의 기준이 11월 정식 작기에 기계적으로 적용되는 오류를 줄입니다.</div>',
        unsafe_allow_html=True,
    )
    st.session_state["v293_alignment_table"] = aligned
    st.session_state["v293_current_context"] = ctx


try:
    render_calendar_phenology_conditional_reference_v293()
except Exception as _v293_runtime_error:
    st.warning(f"Calendar × Phenology 조건부 Reference 모듈 처리 오류: {_v293_runtime_error}")
