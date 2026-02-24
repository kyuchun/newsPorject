"""
뉴스 감성 분석 & 요약 — Streamlit 프론트엔드
FastAPI 백엔드(http://127.0.0.1:8000)를 호출하여 결과를 시각화합니다.
"""

import streamlit as st
import requests
import urllib3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# SSL 경고 억제
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── 설정 ─────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="📰 뉴스 감성 분석 & 요약", page_icon="📰", layout="wide")


# ── API 호출 ─────────────────────────────────────────────────────
def call_api(endpoint, method="GET", params=None, json_body=None):
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, params=params, timeout=30)
        else:
            resp = requests.post(url, json=json_body, timeout=30)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ API 서버에 연결할 수 없습니다."
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = str(e)
        return None, f"❌ API 오류: {detail}"
    except Exception as e:
        return None, f"❌ 오류: {e}"


def check_api_health():
    data, _ = call_api("/health")
    return data is not None


# ── 사이드바 ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")

    api_alive = check_api_health()
    if api_alive:
        st.success("🟢 API 서버 연결됨")
    else:
        st.error("🔴 API 서버 연결 안됨")

    st.divider()
    api_key = st.text_input("🔑 NewsAPI 키", type="password", help="https://newsapi.org/register").strip()
    st.divider()
    st.caption("v1.4 — 한국어 감성 사전 탑재")


# ══════════════════════════════════════════════════════════════════
# 뉴스 검색 & 분석
# ══════════════════════════════════════════════════════════════════
st.title("📰 뉴스 검색 & 감성 분석")

col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
with col1:
    query = st.text_input("🔎 검색 키워드", value="AI")
with col2:
    language = st.selectbox("🌐 언어", ["en", "ko", "ja", "de", "fr", "es"])
with col3:
    page_size = st.slider("📄 기사 수", 1, 20, 5)
with col4:
    summary_sentences = st.slider("📝 요약 문장", 1, 10, 3)

search_btn = st.button("🚀 뉴스 가져오기", type="primary", use_container_width=True)

if search_btn:
    if not api_key:
        st.error("⚠️ 사이드바에서 NewsAPI 키를 입력하세요.")
    elif not api_alive:
        st.error("⚠️ API 서버가 실행 중이 아닙니다.")
    else:
        with st.spinner("📡 뉴스를 가져오고 분석 중..."):
            data, err = call_api("/news", params={
                "api_key": api_key, "query": query, "language": language,
                "page_size": page_size, "summary_sentences": summary_sentences,
            })
        if err:
            st.error(err)
        elif data and data["total"] > 0:
            st.session_state["news_data"] = data
        else:
            st.warning("검색 결과가 없습니다.")

if "news_data" in st.session_state:
    data = st.session_state["news_data"]
    articles = data["articles"]
    ss = data["sentiment_summary"]
    total = data["total"]
    pos = ss.get("Positive 😊", 0)
    neu = ss.get("Neutral 😐", 0)
    neg = ss.get("Negative 😟", 0)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📰 전체", total)
    c2.metric("😊 긍정", pos)
    c3.metric("😐 중립", neu)
    c4.metric("😟 부정", neg)

    # 차트
    ch1, ch2 = st.columns(2)
    with ch1:
        fig = px.pie(values=[pos, neu, neg], names=["긍정", "중립", "부정"],
                     color_discrete_sequence=["#28a745", "#ffc107", "#dc3545"], hole=0.4, title="감성 분포")
        st.plotly_chart(fig, use_container_width=True)
    with ch2:
        titles = [a["title"][:30] + "…" if len(a["title"]) > 30 else a["title"] for a in articles]
        pols = [a["sentiment"]["polarity"] for a in articles]
        colors = ["#28a745" if p > 0.03 else "#dc3545" if p < -0.03 else "#ffc107" for p in pols]
        fig2 = go.Figure(go.Bar(x=pols, y=titles, orientation="h", marker_color=colors,
                                 text=[f"{p:.3f}" for p in pols], textposition="outside"))
        fig2.update_layout(title="기사별 감성 점수", yaxis=dict(autorange="reversed"), height=max(300, len(articles)*60))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # 기사 카드
    for i, a in enumerate(articles, 1):
        label = a["sentiment"]["label"]
        polarity = a["sentiment"]["polarity"]
        if "Positive" in label:
            color = "green"
        elif "Negative" in label:
            color = "red"
        else:
            color = "orange"

        with st.container():
            st.markdown(f"### {i}. {a['title']}")
            mc = st.columns([2, 2, 2])
            mc[0].markdown(f"**출처:** {a['source']}")
            mc[1].markdown(f"**날짜:** {a['published']}")
            mc[2].markdown(f"**감성:** :{color}[{label}] ({polarity:+.4f})")

            if a.get("image_url"):
                try:
                    st.image(a["image_url"], width=500)
                except Exception:
                    pass

            with st.expander("📝 요약 / 분석 상세"):
                st.write(f"**요약:** {a['summary']}")
                st.write(f"**분석된 텍스트:** {a['sentiment'].get('analyzed_text', '')}")
                st.write(f"**분석 방법:** {a['sentiment'].get('method', '')}")

            url = a['url']
            st.markdown(
                f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
                f'🔗 원문 보기 (새 탭)</a>',
                unsafe_allow_html=True,
            )
            st.divider()

    with st.expander("📊 데이터 테이블"):
        df = pd.DataFrame([{
            "제목": a["title"], "출처": a["source"], "날짜": a["published"],
            "감성": a["sentiment"]["label"], "점수": a["sentiment"]["polarity"],
        } for a in articles])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("📥 CSV", df.to_csv(index=False, encoding="utf-8-sig"),
                           "news_sentiment.csv", "text/csv")

