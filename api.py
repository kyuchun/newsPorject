"""
뉴스 감성 분석 & 요약 — FastAPI 백엔드 v1.4
- 한국어 감성 사전 + Google 번역 폴백
"""

import ssl
import os
from contextlib import asynccontextmanager

import nltk
import requests as http_requests
import urllib3
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# ── SSL 전역 우회 (기업 네트워크) ────────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

_orig_request = http_requests.Session.request
def _patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return _orig_request(self, *args, **kwargs)
http_requests.Session.request = _patched_request

from deep_translator import GoogleTranslator


# ── NLTK ─────────────────────────────────────────────────────────
def _ensure_nltk_data():
    for res in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            try:
                _c = ssl._create_default_https_context
                ssl._create_default_https_context = ssl._create_unverified_context
                nltk.download(res, quiet=True)
            finally:
                ssl._create_default_https_context = _c


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_nltk_data()
    yield


app = FastAPI(
    title="뉴스 감성 분석 & 요약 API",
    description="v1.4 — 한국어 감성 사전 + 번역 폴백",
    version="1.4.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SUMY_LANG_MAP = {"en": "english", "ko": "english", "ja": "japanese", "de": "german", "fr": "french", "es": "spanish"}


# ── 모델 ─────────────────────────────────────────────────────────
class SentimentResult(BaseModel):
    label: str
    polarity: float
    analyzed_text: str
    method: str


class ArticleResult(BaseModel):
    title: str
    description: str
    content: str
    url: str
    source: str
    published: str
    image_url: str | None
    sentiment: SentimentResult
    summary: str


class NewsResponse(BaseModel):
    total: int
    sentiment_summary: dict[str, int]
    articles: list[ArticleResult]


class SentimentRequest(BaseModel):
    text: str
    language: str = "auto"


class SummarizeRequest(BaseModel):
    text: str
    sentence_count: int = 3
    language: str = "en"


# ── 감성 사전 ────────────────────────────────────────────────────

EN_POSITIVE = {
    "good", "great", "best", "win", "success", "grow", "surge", "boost",
    "rise", "gain", "profit", "record", "breakthrough", "improve", "love",
    "happy", "excellent", "strong", "launch", "innovation", "support",
    "recover", "positive", "hope", "top", "high", "lead", "award",
    "joy", "wonderful", "fantastic", "amazing", "delight", "pleasure",
    "beautiful", "brilliant", "awesome", "celebrate", "glad", "cheerful",
    "excited", "thrilled", "optimistic", "proud", "grateful", "fortunate",
    "happiness", "satisfaction", "impressive", "remarkable", "outstanding",
}

EN_NEGATIVE = {
    "bad", "worst", "fail", "loss", "crash", "fall", "drop", "kill",
    "death", "war", "attack", "crisis", "fear", "threat", "decline",
    "down", "cut", "fire", "fraud", "scandal", "risk", "danger",
    "collapse", "destroy", "bomb", "terror", "recession", "layoff",
    "victim", "suffer", "wrong", "debt", "bankrupt", "negative",
    "sad", "terrible", "horrible", "awful", "pain", "grief", "angry",
    "hate", "misery", "sorrow", "tragedy", "disaster", "despair",
    "worried", "disappointed", "frustrated", "alarming", "devastating",
    "sadness", "anxiety", "depression", "loneliness", "regret",
}

# 한국어 감성 사전 — 부분 매칭(포함 여부)으로 검사
KO_POSITIVE = [
    "기쁨", "기쁘", "행복", "사랑", "좋다", "좋은", "좋아", "훌륭", "멋진", "멋지",
    "최고", "성공", "감사", "즐거", "즐겁", "축하", "승리", "희망", "응원", "대박",
    "우수", "탁월", "긍정", "만족", "환영", "감동", "뿌듯", "자랑", "신나", "신난",
    "웃음", "환호", "찬사", "보람", "설레", "기대", "잘했", "잘한", "최선", "발전",
    "상승", "호조", "흑자", "이익", "수익", "경사", "복", "건강", "평화", "화합",
]

KO_NEGATIVE = [
    "슬픔", "슬프", "분노", "실패", "나쁘", "나쁜", "최악", "위기", "전쟁", "사망",
    "죽음", "죽었", "공포", "불안", "걱정", "고통", "절망", "비극", "재난", "파괴",
    "손실", "하락", "폭락", "침체", "부정", "혐오", "증오", "괴로", "눈물", "후회",
    "실망", "패배", "좌절", "두려", "우울", "외로", "고독", "원망", "적자", "파산",
    "해고", "사고", "피해", "위험", "폭력", "범죄", "테러", "빚", "부채", "탄핵",
]


def _is_korean(text: str) -> bool:
    for ch in text:
        if "\uac00" <= ch <= "\ud7a3" or "\u3131" <= ch <= "\u3163":
            return True
    return False


def _ko_sentiment_score(text: str) -> tuple[float, int, int]:
    """한국어 감성 사전으로 직접 점수 계산 (부분 매칭)."""
    pos = sum(1 for w in KO_POSITIVE if w in text)
    neg = sum(1 for w in KO_NEGATIVE if w in text)
    score = (pos - neg) * 0.25
    return max(-1.0, min(1.0, score)), pos, neg


def _translate_to_english(text: str) -> tuple[str, bool]:
    """Google 번역 시도. 반환: (결과, 성공여부)"""
    try:
        result = GoogleTranslator(source="auto", target="en").translate(text)
        if result and result.strip() and result.strip().lower() != text.strip().lower():
            return result.strip(), True
        return text, False
    except Exception as e:
        print(f"[번역 실패] {e}")
        return text, False


def _is_valid_text(text: str) -> bool:
    if not text or not text.strip():
        return False
    return not any(m in text for m in ["[removed]", "[Removed]", "(제목 없음)"])


def _analyze_sentiment(text: str, language: str = "auto") -> SentimentResult:
    """
    감성 분석 — 3단계 전략:
      1) 한국어 감성 사전 (즉시)
      2) 영어 번역 + TextBlob + 영어 키워드 사전
      3) 두 결과 종합
    """
    if not _is_valid_text(text):
        return SentimentResult(label="Neutral 😐", polarity=0.0, analyzed_text="(분석 불가)", method="none")

    is_ko = _is_korean(text)
    method_used = ""

    # ── 1단계: 한국어 사전 분석 ──────────────────────────────────
    ko_score, ko_pos, ko_neg = 0.0, 0, 0
    if is_ko:
        ko_score, ko_pos, ko_neg = _ko_sentiment_score(text)
        print(f"[1-KO사전] '{text}' → 긍정={ko_pos}, 부정={ko_neg}, 점수={ko_score:.2f}")

    # ── 2단계: 영어 번역 + TextBlob ──────────────────────────────
    en_text = text
    translated = False
    tb_polarity = 0.0
    en_boost = 0.0

    if is_ko or language not in ("en",):
        en_text, translated = _translate_to_english(text)
        print(f"[2-번역] '{text}' → '{en_text}' (성공={translated})")

    if translated or not is_ko:
        blob = TextBlob(en_text)
        tb_polarity = blob.sentiment.polarity
        words = set(en_text.lower().split())
        en_pos = len(words & EN_POSITIVE)
        en_neg = len(words & EN_NEGATIVE)
        en_boost = (en_pos - en_neg) * 0.15
        print(f"[2-TextBlob] pol={tb_polarity:.4f}, EN(+{en_pos}/-{en_neg}), boost={en_boost:.2f}")

    # ── 3단계: 종합 ─────────────────────────────────────────────
    if is_ko and translated:
        final = ko_score * 0.4 + tb_polarity * 0.3 + en_boost * 0.3
        method_used = f"KO사전({ko_score:.2f})+번역TextBlob({tb_polarity:.2f})+EN키워드({en_boost:.2f})"
    elif is_ko and not translated:
        final = ko_score
        method_used = f"KO사전 only({ko_score:.2f})"
    else:
        final = tb_polarity * 0.6 + en_boost * 0.4
        method_used = f"TextBlob({tb_polarity:.2f})+EN키워드({en_boost:.2f})"

    final = max(-1.0, min(1.0, final))
    print(f"[최종] '{text}' → {final:.4f} ({method_used})")

    if final > 0.03:
        label = "Positive 😊"
    elif final < -0.03:
        label = "Negative 😟"
    else:
        label = "Neutral 😐"

    display_text = en_text if translated else text
    return SentimentResult(label=label, polarity=round(final, 4), analyzed_text=display_text[:100], method=method_used)


def _summarize_text(text: str, sentence_count: int = 3, lang: str = "en") -> str:
    if not text or len(text.split()) < 10:
        return text or "(본문 없음)"
    sumy_lang = SUMY_LANG_MAP.get(lang, "english")
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(sumy_lang))
        stemmer = Stemmer(sumy_lang)
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(sumy_lang)
        summary = summarizer(parser.document, sentence_count)
        return " ".join(str(s) for s in summary) or "(요약 생성 실패)"
    except Exception as e:
        return f"(요약 중 오류: {e})"


def _fetch_news(api_key: str, query: str, language: str, page_size: int) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": language, "pageSize": page_size, "sortBy": "publishedAt", "apiKey": api_key}
    resp = http_requests.get(url, params=params, timeout=15, verify=False)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        raise ValueError(data.get("message", "알 수 없는 오류"))
    return data.get("articles", [])


# ── 엔드포인트 ───────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.4.0"}


@app.get("/news", response_model=NewsResponse)
def get_news(
    api_key: str = Query(..., description="NewsAPI 키"),
    query: str = Query("technology", description="검색 키워드"),
    language: str = Query("en", description="언어 코드 (en, ko, ja, de, fr, es)"),
    page_size: int = Query(5, ge=1, le=20, description="기사 수"),
    summary_sentences: int = Query(3, ge=1, le=10, description="요약 문장 수"),
):
    """뉴스를 가져오고, 감성 분석 및 본문 요약을 수행합니다."""
    try:
        raw = _fetch_news(api_key, query, language, page_size)
    except Exception as e:
        raise HTTPException(502, detail=f"뉴스 가져오기 실패: {e}")

    if not raw:
        return NewsResponse(total=0, sentiment_summary={}, articles=[])

    counts = {"Positive 😊": 0, "Neutral 😐": 0, "Negative 😟": 0}
    results = []

    for a in raw:
        title = a.get("title") or "(제목 없음)"
        desc = a.get("description") or ""
        content = a.get("content") or desc
        analysis = title + (f". {desc}" if desc and _is_valid_text(desc) else "")

        sent = _analyze_sentiment(analysis, language)
        counts[sent.label] = counts.get(sent.label, 0) + 1

        results.append(ArticleResult(
            title=title, description=desc, content=content,
            url=a.get("url", "#"), source=a.get("source", {}).get("name", ""),
            published=(a.get("publishedAt") or "")[:10], image_url=a.get("urlToImage"),
            sentiment=sent, summary=_summarize_text(content, summary_sentences, language),
        ))

    return NewsResponse(total=len(results), sentiment_summary=counts, articles=results)


@app.post("/sentiment", response_model=SentimentResult)
def sentiment(body: SentimentRequest):
    """단일 텍스트의 감성을 분석합니다."""
    if not body.text.strip():
        raise HTTPException(400, "텍스트가 비어 있습니다.")
    return _analyze_sentiment(body.text, body.language)


@app.post("/summarize")
def summarize(body: SummarizeRequest):
    """단일 텍스트를 요약합니다."""
    if not body.text.strip():
        raise HTTPException(400, "텍스트가 비어 있습니다.")
    return {"summary": _summarize_text(body.text, body.sentence_count, body.language)}
