import pickle
import streamlit as st
import Orange
import numpy as np

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="어린이 보호구역 교통사고 위험도 예측기",
    page_icon="🚦",
    layout="centered"
)

# 커스텀 CSS (신호등 및 안전 수칙 카드 스타일링)
st.markdown("""
    <style>
    .traffic-light-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .status-emoji {
        font-size: 80px;
        line-height: 1.2;
    }
    .status-title {
        font-size: 32px;
        font-weight: 800;
        margin-top: 10px;
    }
    .status-desc {
        font-size: 18px;
        margin-top: 8px;
    }
    .safety-card {
        background-color: #f8f9fa;
        border-left: 6px solid #4A90E2;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .safety-card h4 {
        margin-top: 0;
        color: #1f2937;
    }
    .safety-card ul {
        margin-bottom: 0;
        color: #374151;
        line-height: 1.7;
    }
    </style>
""", unsafe_allow_html=True)

# 2. 모델 로드 (캐싱을 통해 속도 최적화)
@st.cache_resource
def load_orange_model(model_path="traffic_casualty_model.pkcls"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_orange_model()
except FileNotFoundError:
    st.error("⚠️ `traffic_casualty_model.pkcls` 파일을 찾을 수 없습니다. 깃허브 저장소에 모델 파일을 함께 업로드해주세요.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 헤더 영역
st.title("🚦 어린이 보호구역 사고 위험도 예측")
st.write("시도별 교통사고 발생 규모에 따른 **예상 사망자 수**를 AI 모델로 실시간 예측하고 안전 수칙을 안내합니다.")
st.divider()

# 4. 사용자 입력 UI (시도 선택 및 슬라이더)
st.subheader("📊 사고 조건 설정")

col1, col2 = st.columns([1, 2])

with col1:
    # 모델 학습 도메인에 포함된 시도 목록
    sido_options = ["서울", "경기", "부산", "인천"]
    selected_sido = st.selectbox("📍 시도 선택", sido_options)

with col2:
    accident_count = st.slider("🚗 사고건수 (건)", min_value=1, max_value=150, value=30, step=1)
    minor_injury_count = st.slider("🩹 경상자수 (명)", min_value=0, max_value=150, value=20, step=1)

# 5. 실시간 예측 실행 함수
def predict_fatalities(orange_model, sido, accidents, minor_injuries):
    # 모델의 original_domain 피처 구조 확인
    # 피처 순서: ['시도', '사고건수', '사상자수', '경상자수', '부상신고자수']
    domain = orange_model.original_domain
    
    # 입력받지 않은 사상자수·부상신고자수는 입력 데이터 비율에 맞춰 추정치 자동 보정
    estimated_report_injuries = max(0, int(round(accidents * 0.1)))
    estimated_total_casualties = max(accidents, minor_injuries + estimated_report_injuries)
    
    # Orange3 Table 형태로 변환 (범주형 변수는 문자열 그대로 매핑 가능)
    input_row = [
        sido,
        float(accidents),
        float(estimated_total_casualties),
        float(minor_injuries),
        float(estimated_report_injuries)
    ]
    
    # 메타 컬럼이 있다면 빈 값 추가
    metas = [""] * len(domain.metas) if domain.metas else []
    
    # 예측용 Orange Table 생성 (class_vars는 결측치 NaN으로 설정)
    table = Orange.data.Table.from_list(
        domain,
        [input_row + [np.nan] + metas]
    )
    
    prediction = orange_model(table)[0]
    # 회귀 모델 특성상 음수가 나올 경우 0.0으로 보정
    return max(0.0, float(prediction))

predicted_deaths = predict_fatalities(model, selected_sido, accident_count, minor_injury_count)

# 6. 예측 결과 및 대형 신호등 표시
st.divider()
st.subheader("🔍 실시간 AI 예측 결과")

st.metric(
    label=f"{selected_sido} 지역 예상 사망자 수",
    value=f"{predicted_deaths:.2f} 명"
)

# 신호등 임계값 판별 (예측된 사망자 수 기준)
if predicted_deaths < 0.5:
    bg_color = "#E8F5E9"
    border_color = "#4CAF50"
    text_color = "#1B5E20"
    emoji = "🟢"
    status = "안전 (SAFE)"
    desc = "예상 사망자 발생 가능성이 낮은 수준입니다. 기본 안전 수칙을 준수해주세요."
    card_border = "#4CAF50"
    tips = [
        "어린이 보호구역 내 제한속도 <b>30km/h 이하</b>를 항상 준수하세요.",
        "횡단보도 앞에서는 보행자가 없더라도 <b>일시정지</b> 후 주변을 살피세요.",
        "학교 정문 및 통학로 주변 불법 주·정차를 절대 하지 마세요."
    ]
elif predicted_deaths < 1.5:
    bg_color = "#FFFDE7"
    border_color = "#FBC02D"
    text_color = "#F57F17"
    emoji = "🟡"
    status = "주의 (CAUTION)"
    desc = "사망 사고 발생 우려가 있습니다. 감속 운행과 각별한 전방 주시가 필요합니다."
    card_border = "#FBC02D"
    tips = [
        "등·하교 시간대에는 <b>서행(20km/h 이하)</b>하며 어린이의 돌발 행동을 대비하세요.",
        "주·정차된 차량 사이에서 어린이가 갑자기 튀어나올 수 있으니 시야 확보에 유의하세요.",
        "우회전 시 보행자 신호와 사각지대를 반드시 재확인한 후 진입하세요."
    ]
else:
    bg_color = "#FFEBEE"
    border_color = "#E53935"
    text_color = "#B71C1C"
    emoji = "🔴"
    status = "위험 (DANGER)"
    desc = "중대 인명피해 및 사망 사고 위험이 매우 높습니다! 즉각적인 안전 조치가 필요합니다."
    card_border = "#E53935"
    tips = [
        "<b>긴급 주의:</b> 스쿨존 내 과속 및 신호위반을 엄격히 단속하고 우회 도로 이용을 권장합니다.",
        "어린이 통학버스 승·하차 시 모든 차로의 차량은 반드시 <b>일시정지 후 안전 확인</b>을 거쳐야 합니다.",
        "안전 펜스, 과속방지턱, 무인단속카메라 등 교통안전 시설물 집중 점검이 필요합니다."
    ]

# 대형 신호등 UI 렌더링
st.markdown(f"""
    <div class="traffic-light-box" style="background-color: {bg_color}; border: 3px solid {border_color}; color: {text_color};">
        <div class="status-emoji">{emoji}</div>
        <div class="status-title">{status}</div>
        <div class="status-desc">{desc}</div>
    </div>
""", unsafe_allow_html=True)

# 7. 어린이 보호구역 안전 수칙 카드
tips_html = "".join([f"<li>{tip}</li>" for tip in tips])
st.markdown(f"""
    <div class="safety-card" style="border-left-color: {card_border};">
        <h4>🛡️ 어린이 보호구역 핵심 안전 수칙 ({selected_sido})</h4>
        <ul>
            {tips_html}
        </ul>
    </div>
""", unsafe_allow_html=True)