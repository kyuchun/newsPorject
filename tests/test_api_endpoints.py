"""
FastAPI 엔드포인트 통합 테스트
- TestClient를 사용한 HTTP 요청/응답 검증
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════
# 1. Health Check
# ═══════════════════════════════════════════════════════════════════
class TestHealthEndpoint:

    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_body(self):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# ═══════════════════════════════════════════════════════════════════
# 2. Root redirect
# ═══════════════════════════════════════════════════════════════════
class TestRootEndpoint:

    def test_root_redirects(self):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/docs" in resp.headers.get("location", "")


# ═══════════════════════════════════════════════════════════════════
# 3. POST /sentiment
# ═══════════════════════════════════════════════════════════════════
class TestSentimentEndpoint:

    def test_positive_korean(self):
        resp = client.post("/sentiment", json={"text": "기쁨", "language": "auto"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "Positive 😊"
        assert data["polarity"] > 0

    def test_negative_korean(self):
        resp = client.post("/sentiment", json={"text": "슬픔", "language": "auto"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "Negative 😟"
        assert data["polarity"] < 0

    def test_neutral_korean(self):
        resp = client.post("/sentiment", json={"text": "회의실 예약", "language": "auto"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "Neutral 😐"

    def test_positive_english(self):
        resp = client.post("/sentiment", json={"text": "This is great and wonderful", "language": "en"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "Positive 😊"

    def test_negative_english(self):
        resp = client.post("/sentiment", json={"text": "terrible disaster and crisis", "language": "en"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "Negative 😟"

    def test_empty_text_returns_400(self):
        resp = client.post("/sentiment", json={"text": "   ", "language": "auto"})
        assert resp.status_code == 400

    def test_response_has_required_fields(self):
        resp = client.post("/sentiment", json={"text": "test"})
        data = resp.json()
        assert "label" in data
        assert "polarity" in data
        assert "analyzed_text" in data
        assert "method" in data

    def test_default_language_is_auto(self):
        """language 미전달 시 기본값 auto"""
        resp = client.post("/sentiment", json={"text": "기쁨"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "Positive 😊"


# ═══════════════════════════════════════════════════════════════════
# 4. POST /summarize
# ═══════════════════════════════════════════════════════════════════
class TestSummarizeEndpoint:

    def test_summarize_short_text(self):
        resp = client.post("/summarize", json={"text": "Short text.", "language": "en"})
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data

    def test_summarize_long_text(self):
        long_text = (
            "Artificial intelligence has transformed the technology industry. "
            "Machine learning algorithms are being used in healthcare. "
            "Deep learning models can now recognize images. "
            "Companies are investing billions in AI research. "
            "The impact of AI on jobs is widely debated. "
            "Experts predict AI will evolve rapidly. "
            "Governments are developing regulations for AI. "
            "AI tools are becoming more accessible."
        )
        resp = client.post("/summarize", json={"text": long_text, "sentence_count": 2, "language": "en"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] != "(본문 없음)"
        assert len(data["summary"]) < len(long_text)

    def test_empty_text_returns_400(self):
        resp = client.post("/summarize", json={"text": "  ", "language": "en"})
        assert resp.status_code == 400

    def test_summarize_response_structure(self):
        resp = client.post("/summarize", json={"text": "Some text here for testing."})
        data = resp.json()
        assert "summary" in data
        assert isinstance(data["summary"], str)


# ═══════════════════════════════════════════════════════════════════
# 5. GET /news (NewsAPI mock)
# ═══════════════════════════════════════════════════════════════════
MOCK_ARTICLES = [
    {
        "title": "Samsung reports record profits this quarter",
        "description": "Samsung Electronics achieved its best quarterly results.",
        "content": "Samsung Electronics reported record-breaking profits driven by strong chip demand. "
                   "The company saw gains across all divisions. Revenue surged past expectations. "
                   "Analysts praised the excellent performance. Investors celebrated the results.",
        "url": "https://example.com/1",
        "source": {"name": "TechNews"},
        "publishedAt": "2026-02-20T10:00:00Z",
        "urlToImage": "https://example.com/img.jpg",
    },
    {
        "title": "Market crash fears grow amid global crisis",
        "description": "Economic recession warnings intensify.",
        "content": "Global markets face collapse as trade wars escalate. "
                   "Investors fear devastating losses and bankruptcy risks. "
                   "The financial crisis threatens jobs worldwide. "
                   "Analysts warn of terrible consequences ahead.",
        "url": "https://example.com/2",
        "source": {"name": "FinanceDaily"},
        "publishedAt": "2026-02-21T08:00:00Z",
        "urlToImage": None,
    },
]


class TestNewsEndpoint:

    @patch("api._fetch_news", return_value=MOCK_ARTICLES)
    def test_news_returns_articles(self, mock_fetch):
        resp = client.get("/news", params={"api_key": "test_key", "query": "tech", "page_size": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["articles"]) == 2

    @patch("api._fetch_news", return_value=MOCK_ARTICLES)
    def test_news_has_sentiment_summary(self, mock_fetch):
        resp = client.get("/news", params={"api_key": "test_key", "query": "tech"})
        data = resp.json()
        assert "sentiment_summary" in data
        summary = data["sentiment_summary"]
        total = sum(summary.values())
        assert total == data["total"]

    @patch("api._fetch_news", return_value=MOCK_ARTICLES)
    def test_news_article_has_required_fields(self, mock_fetch):
        resp = client.get("/news", params={"api_key": "test_key", "query": "tech"})
        article = resp.json()["articles"][0]
        for field in ["title", "description", "content", "url", "source",
                      "published", "sentiment", "summary"]:
            assert field in article, f"'{field}' 필드 누락"

    @patch("api._fetch_news", return_value=MOCK_ARTICLES)
    def test_news_sentiment_is_valid(self, mock_fetch):
        resp = client.get("/news", params={"api_key": "test_key", "query": "tech"})
        for article in resp.json()["articles"]:
            sent = article["sentiment"]
            assert sent["label"] in ("Positive 😊", "Neutral 😐", "Negative 😟")
            assert -1.0 <= sent["polarity"] <= 1.0

    @patch("api._fetch_news", return_value=[])
    def test_news_empty_result(self, mock_fetch):
        resp = client.get("/news", params={"api_key": "test_key", "query": "없는키워드"})
        data = resp.json()
        assert data["total"] == 0
        assert data["articles"] == []

    @patch("api._fetch_news", side_effect=Exception("API 에러"))
    def test_news_api_error_returns_502(self, mock_fetch):
        resp = client.get("/news", params={"api_key": "test_key", "query": "test"})
        assert resp.status_code == 502

    def test_news_missing_api_key_returns_422(self):
        """api_key 미전달 시 422 Validation Error"""
        resp = client.get("/news", params={"query": "test"})
        assert resp.status_code == 422

    @patch("api._fetch_news", return_value=MOCK_ARTICLES)
    def test_news_article_has_summary(self, mock_fetch):
        resp = client.get("/news", params={"api_key": "test_key", "query": "tech"})
        for article in resp.json()["articles"]:
            assert article["summary"] != ""
            assert article["summary"] is not None
