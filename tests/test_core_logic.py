"""
핵심 로직 단위 테스트 — api.py의 순수 함수들
- 한국어 감지 (_is_korean)
- 한국어 감성 사전 점수 (_ko_sentiment_score)
- 텍스트 유효성 검사 (_is_valid_text)
- 감성 분석 통합 (_analyze_sentiment)
- 텍스트 요약 (_summarize_text)
"""

import pytest
from unittest.mock import patch, MagicMock

# api 모듈에서 테스트 대상 함수 임포트
from api import (
    _is_korean,
    _ko_sentiment_score,
    _is_valid_text,
    _analyze_sentiment,
    _summarize_text,
    KO_POSITIVE,
    KO_NEGATIVE,
    EN_POSITIVE,
    EN_NEGATIVE,
    SentimentResult,
)


# ═══════════════════════════════════════════════════════════════════
# 1. _is_korean 테스트
# ═══════════════════════════════════════════════════════════════════
class TestIsKorean:
    """한국어 텍스트 감지 함수 테스트"""

    def test_korean_word(self):
        assert _is_korean("기쁨") is True

    def test_korean_sentence(self):
        assert _is_korean("오늘 날씨가 좋다") is True

    def test_korean_mixed_with_english(self):
        assert _is_korean("Hello 안녕") is True

    def test_english_only(self):
        assert _is_korean("Hello world") is False

    def test_numbers_only(self):
        assert _is_korean("12345") is False

    def test_empty_string(self):
        assert _is_korean("") is False

    def test_japanese(self):
        """일본어 히라가나/가타카나는 한국어가 아님"""
        assert _is_korean("こんにちは") is False

    def test_korean_jamo(self):
        """한글 자모(ㄱ, ㅏ 등)도 한국어로 감지"""
        assert _is_korean("ㅋㅋㅋ") is True

    def test_special_characters(self):
        assert _is_korean("!@#$%^&*()") is False


# ═══════════════════════════════════════════════════════════════════
# 2. _ko_sentiment_score 테스트
# ═══════════════════════════════════════════════════════════════════
class TestKoSentimentScore:
    """한국어 감성 사전 점수 계산 테스트"""

    def test_positive_word(self):
        score, pos, neg = _ko_sentiment_score("기쁨")
        assert pos == 1
        assert neg == 0
        assert score == 0.25

    def test_negative_word(self):
        score, pos, neg = _ko_sentiment_score("슬픔")
        assert pos == 0
        assert neg == 1
        assert score == -0.25

    def test_neutral_text(self):
        score, pos, neg = _ko_sentiment_score("회의실 예약")
        assert pos == 0
        assert neg == 0
        assert score == 0.0

    def test_multiple_positive_words(self):
        score, pos, neg = _ko_sentiment_score("행복하고 기쁨이 넘치는 성공적인 하루")
        assert pos >= 3  # 행복, 기쁨, 성공
        assert score > 0

    def test_multiple_negative_words(self):
        score, pos, neg = _ko_sentiment_score("전쟁과 공포 속에 절망적인 위기")
        assert neg >= 3  # 전쟁, 공포, 절망, 위기
        assert score < 0

    def test_mixed_sentiment(self):
        """긍정+부정 혼합 시 상계 확인"""
        score, pos, neg = _ko_sentiment_score("성공과 실패")
        assert pos >= 1
        assert neg >= 1

    def test_score_clamped_to_minus_one(self):
        """점수가 -1.0 이하로 내려가지 않음"""
        # 부정 단어가 매우 많은 텍스트
        all_neg = " ".join(KO_NEGATIVE)
        score, _, _ = _ko_sentiment_score(all_neg)
        assert score >= -1.0

    def test_score_clamped_to_plus_one(self):
        """점수가 1.0 이상으로 올라가지 않음"""
        all_pos = " ".join(KO_POSITIVE)
        score, _, _ = _ko_sentiment_score(all_pos)
        assert score <= 1.0

    def test_substring_matching(self):
        """부분 매칭: '기쁘다'에서 '기쁘'가 매칭되어야 함"""
        score, pos, neg = _ko_sentiment_score("기쁘다")
        assert pos >= 1  # '기쁘'가 '기쁘다'에 포함

    def test_empty_string(self):
        score, pos, neg = _ko_sentiment_score("")
        assert score == 0.0
        assert pos == 0
        assert neg == 0


# ═══════════════════════════════════════════════════════════════════
# 3. _is_valid_text 테스트
# ═══════════════════════════════════════════════════════════════════
class TestIsValidText:
    """텍스트 유효성 검사 함수 테스트"""

    def test_normal_text(self):
        assert _is_valid_text("정상적인 텍스트입니다") is True

    def test_empty_string(self):
        assert _is_valid_text("") is False

    def test_whitespace_only(self):
        assert _is_valid_text("   ") is False

    def test_none(self):
        assert _is_valid_text(None) is False

    def test_removed_marker(self):
        assert _is_valid_text("[removed]") is False

    def test_removed_capital(self):
        assert _is_valid_text("[Removed]") is False

    def test_no_title_marker(self):
        assert _is_valid_text("(제목 없음)") is False

    def test_removed_in_longer_text(self):
        assert _is_valid_text("기사 내용 [removed] 나머지") is False

    def test_valid_english(self):
        assert _is_valid_text("This is a valid article title") is True


# ═══════════════════════════════════════════════════════════════════
# 4. _analyze_sentiment 통합 테스트
# ═══════════════════════════════════════════════════════════════════
class TestAnalyzeSentiment:
    """감성 분석 통합 함수 테스트 (번역은 mock 처리)"""

    def test_invalid_text_returns_neutral(self):
        result = _analyze_sentiment("")
        assert result.label == "Neutral 😐"
        assert result.method == "none"

    def test_removed_text_returns_neutral(self):
        result = _analyze_sentiment("[removed]")
        assert result.label == "Neutral 😐"

    # ── 한국어 긍정 ──────────────────────────────────────────────
    @patch("api._translate_to_english", return_value=("joy", True))
    def test_korean_positive_joy(self, mock_translate):
        result = _analyze_sentiment("기쁨", "auto")
        assert result.label == "Positive 😊"
        assert result.polarity > 0

    @patch("api._translate_to_english", return_value=("happiness", True))
    def test_korean_positive_happiness(self, mock_translate):
        result = _analyze_sentiment("행복", "auto")
        assert result.label == "Positive 😊"
        assert result.polarity > 0

    @patch("api._translate_to_english", return_value=("love", True))
    def test_korean_positive_love(self, mock_translate):
        result = _analyze_sentiment("사랑", "auto")
        assert result.label == "Positive 😊"

    # ── 한국어 부정 ──────────────────────────────────────────────
    @patch("api._translate_to_english", return_value=("sadness", True))
    def test_korean_negative_sadness(self, mock_translate):
        result = _analyze_sentiment("슬픔", "auto")
        assert result.label == "Negative 😟"
        assert result.polarity < 0

    @patch("api._translate_to_english", return_value=("anger", True))
    def test_korean_negative_anger(self, mock_translate):
        result = _analyze_sentiment("분노", "auto")
        assert result.label == "Negative 😟"

    @patch("api._translate_to_english", return_value=("failure and crisis", True))
    def test_korean_negative_compound(self, mock_translate):
        result = _analyze_sentiment("실패와 위기", "auto")
        assert result.label == "Negative 😟"

    # ── 한국어 중립 ──────────────────────────────────────────────
    @patch("api._translate_to_english", return_value=("reserve a meeting room", True))
    def test_korean_neutral(self, mock_translate):
        result = _analyze_sentiment("회의실 예약", "auto")
        assert result.label == "Neutral 😐"

    # ── 영어 감성 분석 ───────────────────────────────────────────
    def test_english_positive(self):
        result = _analyze_sentiment("This is a great and wonderful achievement", "en")
        assert result.label == "Positive 😊"
        assert result.polarity > 0

    def test_english_negative(self):
        result = _analyze_sentiment("This is terrible and horrible news", "en")
        assert result.label == "Negative 😟"
        assert result.polarity < 0

    def test_english_neutral(self):
        result = _analyze_sentiment("The meeting is scheduled for Monday", "en")
        assert result.label == "Neutral 😐"

    # ── 반환 구조 확인 ───────────────────────────────────────────
    def test_result_is_sentiment_result(self):
        result = _analyze_sentiment("test text", "en")
        assert isinstance(result, SentimentResult)

    def test_result_has_method_field(self):
        result = _analyze_sentiment("good news", "en")
        assert result.method != ""
        assert result.method != "none"

    def test_polarity_range(self):
        """polarity는 항상 [-1, 1] 범위"""
        for text in ["기쁨", "슬픔", "great", "terrible", "회의실"]:
            result = _analyze_sentiment(text)
            assert -1.0 <= result.polarity <= 1.0

    def test_analyzed_text_truncated(self):
        """analyzed_text는 최대 100자"""
        long_text = "good " * 200
        result = _analyze_sentiment(long_text, "en")
        assert len(result.analyzed_text) <= 100


# ═══════════════════════════════════════════════════════════════════
# 5. _summarize_text 테스트
# ═══════════════════════════════════════════════════════════════════
class TestSummarizeText:
    """텍스트 요약 함수 테스트"""

    def test_short_text_returned_as_is(self):
        """10단어 미만 텍스트는 그대로 반환"""
        short = "This is short."
        assert _summarize_text(short) == short

    def test_empty_text(self):
        result = _summarize_text("")
        assert result == "(본문 없음)"

    def test_none_text(self):
        result = _summarize_text(None)
        assert result == "(본문 없음)"

    def test_long_text_summarized(self):
        """실제 요약이 동작하는지 확인"""
        long_text = (
            "Artificial intelligence has transformed the technology industry. "
            "Machine learning algorithms are being used in healthcare, finance, and education. "
            "Deep learning models can now recognize images and understand natural language. "
            "Companies like Google, Microsoft, and OpenAI are investing billions in AI research. "
            "The impact of AI on jobs and society is a topic of ongoing debate. "
            "Experts predict that AI will continue to evolve rapidly in the coming years. "
            "Governments around the world are developing regulations for AI systems. "
            "The ethical implications of AI are being studied by researchers and policymakers. "
            "AI-powered tools are becoming more accessible to everyday users. "
            "The future of AI holds both great promise and significant challenges."
        )
        result = _summarize_text(long_text, sentence_count=2, lang="en")
        assert result != "(본문 없음)"
        assert result != "(요약 생성 실패)"
        assert len(result) < len(long_text)  # 요약이므로 원본보다 짧아야 함

    def test_sentence_count_respected(self):
        """요약 문장 수 제한 확인"""
        text = (
            "First sentence here. Second sentence goes here. Third sentence is this one. "
            "Fourth sentence follows. Fifth sentence ends it. Sixth sentence too. "
            "Seventh sentence. Eighth sentence keeps going. Ninth sentence. Tenth sentence."
        )
        result = _summarize_text(text, sentence_count=2, lang="en")
        # 문장 수가 대략 요청한 수를 넘지 않아야 함
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        assert len(sentences) <= 4  # sumy 특성상 약간 넘을 수 있으므로 여유 줌

    def test_korean_language_fallback(self):
        """한국어는 내부적으로 english tokenizer를 사용"""
        text = (
            "인공지능이 기술 산업을 변화시키고 있다. "
            "머신러닝 알고리즘은 의료, 금융, 교육 분야에서 활용되고 있다. "
            "딥러닝 모델은 이미지 인식과 자연어 이해가 가능하다. "
            "전 세계 정부들이 AI 규제를 개발하고 있다. "
            "AI의 윤리적 함의를 연구자들이 연구하고 있다."
        )
        result = _summarize_text(text, sentence_count=2, lang="ko")
        assert "(요약 중 오류" not in result

    def test_invalid_language_fallback(self):
        """지원하지 않는 언어 코드는 english로 폴백"""
        text = "This is a sample text. " * 20
        result = _summarize_text(text, lang="zz")
        assert "(요약 중 오류" not in result or result == text


# ═══════════════════════════════════════════════════════════════════
# 6. 감성 사전 무결성 테스트
# ═══════════════════════════════════════════════════════════════════
class TestSentimentDictionaries:
    """감성 사전 데이터 무결성 검사"""

    def test_ko_positive_not_empty(self):
        assert len(KO_POSITIVE) > 0

    def test_ko_negative_not_empty(self):
        assert len(KO_NEGATIVE) > 0

    def test_en_positive_not_empty(self):
        assert len(EN_POSITIVE) > 0

    def test_en_negative_not_empty(self):
        assert len(EN_NEGATIVE) > 0

    def test_ko_positive_no_duplicates(self):
        assert len(KO_POSITIVE) == len(set(KO_POSITIVE))

    def test_ko_negative_no_duplicates(self):
        assert len(KO_NEGATIVE) == len(set(KO_NEGATIVE))

    def test_no_overlap_ko_positive_negative(self):
        """긍정/부정 사전에 중복 단어 없어야 함"""
        overlap = set(KO_POSITIVE) & set(KO_NEGATIVE)
        assert len(overlap) == 0, f"중복: {overlap}"

    def test_no_overlap_en_positive_negative(self):
        overlap = EN_POSITIVE & EN_NEGATIVE
        assert len(overlap) == 0, f"중복: {overlap}"

    def test_ko_positive_has_key_words(self):
        """핵심 긍정 단어가 사전에 포함"""
        for w in ["기쁨", "행복", "사랑", "성공", "희망"]:
            assert w in KO_POSITIVE, f"'{w}' 누락"

    def test_ko_negative_has_key_words(self):
        """핵심 부정 단어가 사전에 포함"""
        for w in ["슬픔", "분노", "실패", "전쟁", "위기"]:
            assert w in KO_NEGATIVE, f"'{w}' 누락"
