"""Diversity spec: topic × environment × register × age × background."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

TARGET_COUNT = 10000
HALFPAGE_COUNT = 1000  # alias: same size as the 1-page set
ONEPAGE_COUNT = 1000
FIVEPAGE_COUNT = 1000

ENVIRONMENTS = ("online", "offline")
REGISTERS = ("casual", "formal", "professional")

AGES = (17, 19, 22, 25, 28, 31, 34, 37, 41, 45, 49, 53, 58, 63, 68)

# Plausible speaker age for each background (inclusive).
BACKGROUND_AGES: dict[str, tuple[int, int]] = {
    "고등학생": (16, 19),
    "대학생": (19, 27),
    "취업준비생": (22, 32),
    "신입 사무직": (23, 31),
    "중견 회사원": (30, 48),
    "팀장급 관리자": (36, 55),
    "공무원": (25, 58),
    "초등학교 교사": (24, 55),
    "고등학교 교사": (25, 58),
    "간호사": (23, 55),
    "병원 원무과": (24, 55),
    "소프트웨어 개발자": (24, 45),
    "디자이너": (24, 42),
    "자영업 사장님": (30, 62),
    "카페 알바": (18, 28),
    "배달 라이더": (20, 40),
    "현장 기능직": (25, 55),
    "프리랜서 작가": (26, 50),
    "육아 중인 부모": (28, 45),
    "전업주부": (30, 55),
    "은퇴 후 자원봉사자": (60, 75),
    "농촌 거주 농업인": (40, 70),
    "부동산 중개사": (30, 60),
    "학원 강사": (24, 45),
    "사회복지사": (25, 50),
}

BACKGROUNDS = (
    "고등학생",
    "대학생",
    "취업준비생",
    "신입 사무직",
    "중견 회사원",
    "팀장급 관리자",
    "공무원",
    "초등학교 교사",
    "고등학교 교사",
    "간호사",
    "병원 원무과",
    "소프트웨어 개발자",
    "디자이너",
    "자영업 사장님",
    "카페 알바",
    "배달 라이더",
    "현장 기능직",
    "프리랜서 작가",
    "육아 중인 부모",
    "전업주부",
    "은퇴 후 자원봉사자",
    "농촌 거주 농업인",
    "부동산 중개사",
    "학원 강사",
    "사회복지사",
)

TOPICS = (
    "동네 분식집",
    "배달 음식 후기",
    "카페 자리 다툼",
    "회식 자리",
    "집밥과 반찬",
    "편의점 야식",
    "전통시장 장보기",
    "출근길 지하철",
    "야근과 잔업",
    "주간 업무 보고",
    "코드 인수인계",
    "채용 면접",
    "이직 고민",
    "프리랜서 계약",
    "성과 평가",
    "중간고사 대비",
    "졸업 논문",
    "학부모 상담",
    "동아리 모임",
    "수능 전날",
    "명절 차례",
    "어린이집 하원",
    "부모님 병원",
    "형제 모임",
    "결혼 준비",
    "이사와 짐",
    "전세 갱신",
    "월세 인상",
    "층간소음",
    "관리비 고지",
    "부동산 매물",
    "감기와 약국",
    "헬스장 등록",
    "건강검진 결과",
    "한의원 침",
    "기차 연착",
    "제주 여행",
    "주말 등산",
    "해외여행 준비",
    "당근마켓 거래",
    "쿠팡 환불",
    "백화점 세일",
    "아파트 반상회",
    "동호회 정기모임",
    "지역 축제",
    "스마트폰 고장",
    "인터넷 장애",
    "게임 업데이트",
    "선거 투표",
    "태풍 대비",
    "버스 파업",
    "반려동물 병원",
    "독서 모임",
    "축구 동호회",
    "요리 클래스",
    "은행 창구",
    "주민센터 민원",
    "법원 서류",
    "보험 청구",
    "세금 신고",
    "아파트 주차",
    "학교 급식",
    "과외 상담",
    "군대 휴가",
    "알바 면접",
    "창업 아이디어",
    "재택근무",
    "회의록 정리",
    "고객 클레임",
    "제품 출시",
    "연구 노트",
    "실험실 안전",
    "학회 발표",
    "병문안",
    "조문",
    "결혼식 축사",
    "동창회",
    "고속버스",
    "김장",
    "장마 침수",
)


# Inter-generational speech situations. speech_level is banmal | jondaet.
RELATIONS = (
    ("child_to_parent", "jondaet", "younger_to_older", "부모", 28),
    ("parent_to_child", "banmal", "older_to_younger", "자녀", -26),
    ("grandchild_to_grandparent", "jondaet", "younger_to_older", "조부모", 48),
    ("grandparent_to_grandchild", "banmal", "older_to_younger", "손주", -48),
    ("junior_to_senior", "jondaet", "younger_to_older", "직장 선배", 9),
    ("senior_to_junior", "banmal", "older_to_younger", "직장 후배", -9),
    ("student_to_teacher", "jondaet", "younger_to_older", "선생님", 22),
    ("teacher_to_student", "banmal", "older_to_younger", "학생", -20),
    ("peers_close", "banmal", "peers", "친구", 1),
    ("peers_distant", "jondaet", "peers", "처음 만난 또래", 0),
    ("in_law_younger", "jondaet", "younger_to_older", "시어른·장인장모", 30),
    ("staff_to_customer", "jondaet", "service", "손님", 8),
)


@dataclass(frozen=True)
class DiversitySpec:
    doc_id: int
    topic: str
    environment: str
    register: str
    age: int
    background: str
    relation: str
    speech_level: str
    generation: str
    addressee: str
    addressee_age: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp_age(n: int) -> int:
    return max(8, min(85, n))


def _age_for_background(background: str, doc_id: int) -> int:
    lo, hi = BACKGROUND_AGES.get(background, (22, 60))
    candidates = [a for a in AGES if lo <= a <= hi]
    if not candidates:
        return max(lo, min(hi, 35))
    return candidates[doc_id % len(candidates)]


def _rel(name: str) -> tuple:
    return next(r for r in RELATIONS if r[0] == name)


def _fit_relation(age: int, rel: tuple) -> tuple:
    """Drop relations that contradict the speaker's age."""
    rid = rel[0]
    if rid == "parent_to_child" and age < 28:
        return _rel("peers_close")
    if rid == "grandparent_to_grandchild" and age < 55:
        return _rel("senior_to_junior") if age < 28 else _rel("parent_to_child")
    if rid == "grandchild_to_grandparent" and age > 45:
        return _rel("child_to_parent")
    if rid == "child_to_parent" and age > 50:
        return _rel("peers_close")
    if rid == "student_to_teacher" and age > 35:
        return _rel("junior_to_senior")
    if rid == "teacher_to_student" and age < 24:
        return _rel("peers_close")
    return rel


def spec_for_id(doc_id: int) -> DiversitySpec:
    """Deterministic unique-enough assignment covering every axis."""
    if doc_id < 0:
        raise ValueError("doc_id must be >= 0")
    topic = TOPICS[doc_id % len(TOPICS)]
    environment = ENVIRONMENTS[(doc_id // len(TOPICS)) % len(ENVIRONMENTS)]
    register = REGISTERS[(doc_id // (len(TOPICS) * len(ENVIRONMENTS))) % len(REGISTERS)]
    background = BACKGROUNDS[(doc_id * 13) % len(BACKGROUNDS)]
    age = _age_for_background(background, doc_id * 7)
    rel = _fit_relation(age, RELATIONS[(doc_id * 11) % len(RELATIONS)])
    # parent_to_child fallback when age is too low after remap
    if rel[0] == "parent_to_child" and age < 28:
        rel = next(r for r in RELATIONS if r[0] == "peers_close")
    if rel[0] == "grandparent_to_grandchild" and age < 55:
        rel = next(r for r in RELATIONS if r[0] == "senior_to_junior")
    rel_id, speech_level, generation, addressee, delta = rel
    return DiversitySpec(
        doc_id=doc_id,
        topic=topic,
        environment=environment,
        register=register,
        age=age,
        background=background,
        relation=rel_id,
        speech_level=speech_level,
        generation=generation,
        addressee=addressee,
        addressee_age=_clamp_age(age + delta),
    )


def histogram(records: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {
        "topic": {},
        "environment": {},
        "register": {},
        "age": {},
        "background": {},
        "speech_level": {},
        "relation": {},
        "generation": {},
    }
    for rec in records:
        for key in out:
            val = str(rec[key])
            out[key][val] = out[key].get(val, 0) + 1
    return out
