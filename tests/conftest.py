"""pytest 전역 설정 — 패키지 호환성 경고 억제"""
import warnings

# requests 패키지가 import 시점에 발생시키는 경고를 전역 억제
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
