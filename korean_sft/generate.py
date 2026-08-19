"""Agent writers: emit 10k raw Korean documents with diversity metadata.

Generation follows fluent-korean (complete sentences, particles kept, no
em-dash). Drafts are intentionally seeded with AI-tell so the polish path
has work to do. Resume-safe JSONL writer.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Callable, Iterable

from .diversity import (
    FIVEPAGE_COUNT,
    HALFPAGE_COUNT,
    ONEPAGE_COUNT,
    TARGET_COUNT,
    DiversitySpec,
    spec_for_id,
)
from .paths import FIVEPAGE_RAW_PATH, HALFPAGE_RAW_PATH, ONEPAGE_RAW_PATH, RAW_PATH

HANGUL_BASE = 0xAC00


def jongseong(word: str) -> int:
    """Hangul jongseong index (0 = no batchim). ㄹ is 8."""
    for ch in reversed(word):
        code = ord(ch) - HANGUL_BASE
        if 0 <= code <= 11171:
            return code % 28
    return 0


def has_batchim(word: str) -> bool:
    return jongseong(word) != 0


def eul_reul(word: str) -> str:
    return word + ("을" if has_batchim(word) else "를")


def i_ga(word: str) -> str:
    return word + ("이" if has_batchim(word) else "가")


def eun_neun(word: str) -> str:
    return word + ("은" if has_batchim(word) else "는")


def euro(word: str) -> str:
    """으로 after batchim, except ㄹ (길로, not 길으로)."""
    js = jongseong(word)
    if js == 0 or js == 8:
        return word + "로"
    return word + "으로"


PLACES = (
    "신림역 2번 출구", "홍대입구 9번", "강남역 지하상가", "수원역 로데오",
    "부산 서면 젊음의 거리", "대전 은행동", "광주 충장로", "인천 구월동",
    "일산 라페스타", "분당 서현역", "노원역 롯데백화점 앞", "건대입구 커먼그라운드",
    "혜화 마로니에 공원", "이태원 경리단길", "성수 연무장길", "연남동 경의선숲길",
    "해운대 해변", "전주 한옥마을", "경주 황리단길", "강릉 안목해변",
    "춘천 명동", "청주 성안길", "울산 삼산동", "창원 상남동",
    "제주 동문시장", "속초 중앙시장", "안동 구시장", "포항 죽도시장",
    "우리 아파트 상가", "회사 앞 골목", "학교 정문 건너편", "주민센터 옆",
)

GIVEN = (
    "민준", "서연", "하준", "지아", "도윤", "하윤", "은우", "서준",
    "수아", "예준", "지우", "시우", "채원", "주원", "윤서", "건우",
    "지호", "수빈", "현우", "소율", "연우", "지민", "우진", "하은",
    "선우", "다은", "준서", "예은", "시윤", "유진",
)

DAYS = (
    "지난 월요일", "화요일 저녁", "수요일 점심", "목요일 아침", "금요일 밤",
    "토요일 오후", "일요일 이른 아침", "그저께", "어제", "오늘 낮",
    "이번 주 수요일", "저번 주말", "보름 전", "한 달 전",
)

TIMES = (
    "일곱 시 십분", "여덟 시 반", "아홉 시 정각", "열 시 사십분",
    "열한 시 조금 넘어서", "낮 열두 시", "오후 세 시", "저녁 여섯 시",
    "밤 아홉 시", "자정 무렵",
)

NUMBERS = (2, 3, 4, 5, 6, 7, 8, 12, 15, 20, 24, 30, 45, 50, 80, 120)
PRICES = (1800, 3500, 4500, 7900, 12000, 15800, 23000, 35000, 48000, 67000, 89000, 135000)
STATIONS = (
    "2호선", "3호선", "4호선", "5호선", "7호선", "9호선", "분당선", "경의중앙선",
    "신분당선", "수인분당선",
)

# Topic cluster → concrete nouns used as unique facts
TOPIC_NOUNS: dict[str, tuple[str, ...]] = {
    "동네 분식집": ("김밥", "떡볶이", "라면", "순대", "튀김", "어묵"),
    "배달 음식 후기": ("치킨", "족발", "짜장면", "짬뽕", "피자", "초밥"),
    "카페 자리 다툼": ("아메리카노", "라떼", "케이크", "콘센트 자리", "창가 테이블", "디카페인"),
    "회식 자리": ("삼겹살", "소주", "된장찌개", "회", "막걸리", "곱창"),
    "집밥과 반찬": ("된장찌개", "멸치볶음", "김치", "계란말이", "시금치나물", "고등어구이"),
    "편의점 야식": ("삼각김밥", "컵라면", "도시락", "바나나우유", "핫바", "김밥"),
    "전통시장 장보기": ("고등어", "배추", "대파", "두부", "오징어", "사과"),
    "출근길 지하철": ("환승", "빈자리", "지연 방송", "스크린도어", "출근 인파", "막차"),
    "야근과 잔업": ("마감", "야근 식대", "막차", "로그", "배포", "야근"),
    "주간 업무 보고": ("진행률", "리스크", "일정", "산출물", "담당자", "마일스톤"),
    "코드 인수인계": ("저장소", "배포 스크립트", "환경 변수", "테스트", "권한", "문서"),
    "채용 면접": ("자기소개", "포트폴리오", "코딩 과제", "연봉", "출근일", "직무"),
    "이직 고민": ("연봉", "팀 분위기", "출퇴근", "성장", "복지", "이직"),
    "프리랜서 계약": ("계약서", "선금", "수정 횟수", "납기", "세금계산서", "용역"),
    "성과 평가": ("목표", "실적", "동료 평가", "등급", "면담", "피드백"),
    "중간고사 대비": ("범위", "족보", "스터디", "쪽지 시험", "필기", "암기"),
    "졸업 논문": ("지도교수", "설문", "참고문헌", "초록", "심사", "각주"),
    "학부모 상담": ("성적표", "생활기록", "방과후", "급식", "친구 관계", "진로"),
    "동아리 모임": ("회비", "장소", "신입", "발표", "뒷풀이", "일정"),
    "수능 전날": ("수험표", "도시락", "시계", "숙소", "준비물", "좌석"),
    "명절 차례": ("전", "과일", "지방", "제사", "세배", "차례상"),
    "어린이집 하원": ("가방", "알림장", "간식", "낮잠", "열", "원복"),
    "부모님 병원": ("접수", "처방전", "대기", "검사", "보호자", "진료"),
    "형제 모임": ("부모님", "용돈", "집", "여행", "명절", "형제"),
    "결혼 준비": ("예식장", "스드메", "청첩장", "혼수", "신혼집", "하객"),
    "이사와 짐": ("박스", "포장", "엘리베이터", "입주", "청소", "짐차"),
    "전세 갱신": ("보증금", "계약서", "중개사", "확정일자", "특약", "갱신"),
    "월세 인상": ("관리비", "월세", "집주인", "계약", "통보", "인상"),
    "층간소음": ("아이들", "청소기", "항의", "관리실", "매트", "소음"),
    "관리비 고지": ("고지서", "전기", "수도", "장기수선충당금", "이의", "납부"),
    "부동산 매물": ("전용면적", "역세권", "학군", "융자", "중개수수료", "시세"),
    "감기와 약국": ("타이레놀", "가글", "처방", "보험", "증상", "휴일지킴이"),
    "헬스장 등록": ("PT", "락커", "등록비", "운동화", "인바디", "트레이너"),
    "건강검진 결과": ("콜레스테롤", "혈압", "위내시경", "재검", "수치", "소견"),
    "한의원 침": ("침", "부항", "추나", "보험", "예약", "한약"),
    "기차 연착": ("KTX", "지연", "환승", "좌석", "차내식", "승무원"),
    "제주 여행": ("렌터카", "흑돼지", "올레길", "공항", "숙소", "해수욕장"),
    "주말 등산": ("북한산", "도시락", "스틱", "정상", "하산", "주차장"),
    "해외여행 준비": ("여권", "환전", "보험", "수하물", "비자", "어댑터"),
    "당근마켓 거래": ("직거래", "가격", "채팅", "네고", "보관", "거래"),
    "쿠팡 환불": ("송장", "박스", "교환", "고객센터", "적립금", "반품"),
    "백화점 세일": ("쿠폰", "피팅룸", "주차", "적립", "세일", "대기"),
    "아파트 반상회": ("안건", "입대의", "도색", "주차", "소음", "의결"),
    "동호회 정기모임": ("회비", "장소", "신입", "회장", "뒷풀이", "공지"),
    "지역 축제": ("부스", "공연", "주차", "먹거리", "혼잡", "안내"),
    "스마트폰 고장": ("액정", "배터리", "백업", "수리", "보증", "초기화"),
    "인터넷 장애": ("모뎀", "기사", "약정", "속도", "장애", "재부팅"),
    "게임 업데이트": ("패치", "서버", "점검", "과금", "길드", "렉"),
    "선거 투표": ("투표소", "신분증", "기표", "사전투표", "공보물", "투표용지"),
    "태풍 대비": ("창문", "배수구", "생수", "손전등", "대피", "일기예보"),
    "버스 파업": ("대체 노선", "출근", "택시", "안내 방송", "파업", "배차"),
    "반려동물 병원": ("접종", "중성화", "사료", "예약", "수술", "펫보험"),
    "독서 모임": ("발제", "서평", "다음 책", "장소", "지각", "토론"),
    "축구 동호회": ("구장", "유니폼", "심판", "회비", "부상", "경기"),
    "요리 클래스": ("레시피", "재료", "앞치마", "수강료", "시식", "강사"),
    "은행 창구": ("번호표", "통장", "이체", "수수료", "창구", "대기"),
    "주민센터 민원": ("등본", "인감", "대기표", "무인발급", "수수료", "신청"),
    "법원 서류": ("소장", "송달", "인지대", "기일", "대리인", "보정"),
    "보험 청구": ("진단서", "약제비", "피보험자", "접수", "심사", "지급"),
    "세금 신고": ("홈택스", "원천징수", "공제", "기한", "세무사", "신고서"),
    "아파트 주차": ("방문증", "이중주차", "차단기", "과태료", "전기차", "배정"),
    "학교 급식": ("알레르기", "잔반", "식단표", "배식", "영양사", "급식비"),
    "과외 상담": ("시급", "교재", "진도", "학부모", "시험", "화상"),
    "군대 휴가": ("휴가증", "터미널", "복귀", "보급", "면회", "열차"),
    "알바 면접": ("시급", "주휴", "근무표", "유니폼", "교육", "근로계약"),
    "창업 아이디어": ("임대료", "초기 비용", "상권", "메뉴", "허가", "손익"),
    "재택근무": ("화상회의", "슬랙", "아이 소리", "코어타임", "VPN", "자리"),
    "회의록 정리": ("안건", "결정", "액션아이템", "참석자", "차기", "배포"),
    "고객 클레임": ("교환", "환불", "상담원", "대기", "보상", "접수번호"),
    "제품 출시": ("출시일", "재고", "가격", "리뷰", "프로모션", "초도"),
    "연구 노트": ("실험", "시약", "재현", "오차", "기록", "프로토콜"),
    "실험실 안전": ("후드", "보호구", "폐기", "교육", "사고", "MSDS"),
    "학회 발표": ("포스터", "질의", "초록", "세션", "좌장", "슬라이드"),
    "병문안": ("병실", "면회", "과일", "면회증", "보호자", "면회시간"),
    "조문": ("근조", "향", "부의금", "빈소", "장지", "상주"),
    "결혼식 축사": ("신랑", "신부", "주례", "하객", "축의금", "사회"),
    "동창회": ("졸업앨범", "회비", "장소", "짝꿍", "근황", "단체사진"),
    "고속버스": ("터미널", "좌석", "휴게소", "지연", "짐칸", "예매"),
    "김장": ("배추", "고춧가루", "새우젓", "대야", "이웃", "김장통"),
    "장마 침수": ("빗물", "배수", "모래주머니", "지하", "대피", "보험"),
}


def _nouns(topic: str) -> tuple[str, ...]:
    return TOPIC_NOUNS.get(topic, (topic, "일", "자리", "사람", "이야기", "시간"))


def _pick(rng: random.Random, seq: Iterable) -> object:
    seq = tuple(seq)
    return seq[rng.randrange(len(seq))]


def _facts(spec: DiversitySpec, rng: random.Random) -> dict[str, str]:
    nouns = _nouns(spec.topic)
    noun = str(_pick(rng, nouns))
    other = str(_pick(rng, [n for n in nouns if n != noun] or nouns))
    place = str(_pick(rng, PLACES))
    name = str(_pick(rng, GIVEN))
    day = str(_pick(rng, DAYS))
    time = str(_pick(rng, TIMES))
    n = int(_pick(rng, NUMBERS))
    price = int(_pick(rng, PRICES))
    station = str(_pick(rng, STATIONS))
    return {
        "topic": spec.topic,
        "noun": noun,
        "other": other,
        "place": place,
        "name": name,
        "day": day,
        "time": time,
        "n": str(n),
        "price": f"{price:,}".replace(",", ","),
        "station": station,
        "age": str(spec.age),
        "background": spec.background,
        "noun_obj": eul_reul(noun),
        "noun_subj": i_ga(noun),
        "noun_top": eun_neun(noun),
        "other_obj": eul_reul(other),
        "place_euro": euro(place),
        "name_subj": i_ga(name),
        "topic_obj": eul_reul(spec.topic),
        "topic_top": eun_neun(spec.topic),
        "place2": str(_pick(rng, [p for p in PLACES if p != place] or PLACES)),
        "name2": str(_pick(rng, [g for g in GIVEN if g != name] or GIVEN)),
        "n2": str(int(_pick(rng, NUMBERS))),
        "price2": f"{int(_pick(rng, PRICES)):,}",
        "place3": str(_pick(rng, PLACES)),
        "name3": str(_pick(rng, GIVEN)),
        "n3": str(int(_pick(rng, NUMBERS))),
        "day2": str(_pick(rng, DAYS)),
        "time2": str(_pick(rng, TIMES)),
        "noun_and": noun + ("이랑" if has_batchim(noun) else "랑"),
        "name_voc": name + ("아" if has_batchim(name) else "야"),
        "addressee": spec.addressee,
        "addressee_age": str(spec.addressee_age),
        "relation": spec.relation,
        "speech_level": spec.speech_level,
    }


# ---------------------------------------------------------------------------
# Register-specific sentence banks. Placeholders match _facts keys.
# fluent-korean: full sentences with particles and finite endings.
# ---------------------------------------------------------------------------

CASUAL_OPEN = (
    "{day} {place}에서 {noun_obj} 샀어. {price}원 나왔는데 생각보다 괜찮더라.",
    "{day} {name}이랑 {place}에서 만났어. {noun} 이야기하다가 시간 가는 줄 몰랐어.",
    "나 {age}살 {background}인데, {topic} 때문에 {day} 진짜 정신없었어.",
    "{time}에 {place} 갔더니 {noun_subj} 이미 동이 났어.",
    "{day} {station} 타는데 {noun} 생각나서 {name}한테 바로 톡 보냈어.",
    "야, {place} {noun} 먹어봤어? 나 {day} 처음 가봤는데 {n}분이나 기다렸어.",
    "{background}로 일하면서 {topic} 겪으니까 그냥 넘기기가 어렵더라.",
    "{day} {place}에서 {name}이 {noun_obj} 가져왔는데 양이 {n}인분이었어.",
)

CASUAL_MID = (
    "{name_subj} {other_obj} 먼저 시키길래 나도 따라서 시켰어.",
    "옆 테이블에서 {noun} 이야기를 하는데 우리 집이랑 상황이 비슷했어.",
    "가격이 {price}원이라 좀 망설였는데, {other}까지 생각하면 납득이 가더라.",
    "{n}번이나 다시 확인했는데도 {noun} 때문에 마음이 안 놓였어.",
    "원래는 {other}만 하려고 했는데 {noun}까지 엮이니까 일이 커졌어.",
    "요즘 {age}살에 {background} 하면서 {topic} 이런 거 자주 생겨.",
    "{place_euro} 가는 길이 {station}이라 환승이 한 번 더 있어.",
    "솔직히 {noun_top} 기대보다 별로였고 {other_obj} 더 잘했어.",
    "{name} 말로는 다음에도 {place}에서 {noun_obj} 하자더라.",
    "중간에 {time}쯤 비가 와서 {noun} 계획이 좀 꼬였어.",
)

CASUAL_CLOSE = (
    "다음에 또 {place} 가면 {name}한테 먼저 연락할게.",
    "일단 오늘은 여기까지 하고 내일 {noun} 다시 볼게.",
    "너는 이런 상황에서 어떻게 했어? {other}도 같이 생각해 줘.",
    "나는 그냥 {noun_obj} 기준으로 결정하려고.",
    "이 정도면 {topic} 치고는 괜찮은 편인 것 같아.",
    "나중에 {day} 다시 이야기하자. 지금은 좀 피곤해.",
)

CASUAL_ONLINE_EXTRA = (
    "방금 사진 올렸어. {place} {noun}이야.",
    "댓글에 {name}이 먼저 알아보더라. 신기하지.",
    "오픈채팅에도 같은 {topic} 글이 올라와서 링크 공유했어.",
    "당근 채팅으로 {price}원에 네고했는데 아직 답 없어.",
    "카톡 공지에 {time}이라고 적혀 있는데 사람마다 다르게 읽더라고.",
)

CASUAL_OFFLINE_EXTRA = (
    "직접 가서 보니까 사진이랑 {noun} 상태가 달랐어.",
    "창구에서 {n}번 불릴 때까지 서서 기다렸어.",
    "옆 사람이 {name} 아는 척해서 잠깐 인사하고 지나갔어.",
    "현장에서 {other_obj} 보여주니까 담당자가 바로 알아듣더라.",
    "자리에 앉아서 {noun} 이야길 꺼내니까 분위기가 좀 진지해졌어.",
)

FORMAL_OPEN = (
    "{day} {place}에서 {topic_obj} 확인했습니다. {noun_subj} {time} 기준으로 준비되어 있었습니다.",
    "안녕하세요. {age}세 {background} {name}입니다. {topic} 관련하여 상황을 말씀드립니다.",
    "{day} {station}을 이용하여 {place_euro} 이동하였고, {noun_obj} 점검하였습니다.",
    "{topic_top} {n}건이 접수되어 우선순위를 정리하였습니다.",
    "{place}에서 {name} 담당자를 만나 {noun_obj} 확인한 결과를 공유합니다.",
    "금일 {time}에 {topic} 건으로 유선 문의가 있었습니다.",
)

FORMAL_MID = (
    "{noun_top} {price}원으로 안내받았으며, {other} 항목은 별도 확인이 필요합니다.",
    "{name_subj} {n}시까지 {place}에 도착한다고 하였습니다.",
    "대기 인원이 {n}명이었고, 창구에서는 순번대로 처리하고 있었습니다.",
    "{other_obj} 먼저 처리한 뒤에 {noun_obj} 진행하는 편이 안전합니다.",
    "관련 서류는 {day} 오전에 제출하였고, 접수번호는 {n}번대입니다.",
    "{station} 지연으로 {time}보다 {n}분 늦게 도착하였습니다.",
    "문의 내용은 {noun} 일정과 {other} 비용이 중심이었습니다.",
    "연령대와 직무를 고려하면 {background} 입장에서 {topic_top} 부담이 됩니다.",
)

FORMAL_CLOSE = (
    "추가 확인이 끝나는 대로 다시 연락드리겠습니다.",
    "내일 {time}까지 {noun} 결과를 회신하겠습니다.",
    "검토 의견을 주시면 {other} 일정에 반영하겠습니다.",
    "이상입니다. 잘못된 부분이 있으면 지적해 주시기 바랍니다.",
    "필요하신 서류가 더 있으면 말씀해 주십시오.",
)

FORMAL_ONLINE_EXTRA = (
    "메일로 보낸 첨부파일을 확인해 주시면 감사하겠습니다.",
    "단체 대화방에 {time} 일정을 올려 두었습니다.",
    "온라인 신청 화면에서 {noun} 항목이 비활성화되어 있었습니다.",
    "고객센터 채팅으로 {n}번 문의하였고, 상담원 연결까지 {n}분 걸렸습니다.",
)

FORMAL_OFFLINE_EXTRA = (
    "창구에서 신분증을 확인하고 {noun_obj} 접수하였습니다.",
    "현장 안내에 따라 {place} {n}번 창구로 이동하였습니다.",
    "대면 상담에서 {name} 담당자가 {other_obj} 설명하였습니다.",
    "대기실에서 번호표를 받고 {time}에 호출되었습니다.",
)

PRO_OPEN = (
    "{topic} 진행 현황을 {day} {time} 기준으로 정리한다.",
    "{place} 현장에서 {noun_obj} 점검한 결과를 기록한다.",
    "본 메모는 {background} 관점에서 {topic_obj} 한정한다.",
    "{name} 담당과 {n}건의 {noun} 항목을 대조하였다.",
    "{station} 구간 이동 후 {place}에서 {other_obj} 확인하였다.",
)

PRO_MID = (
    "{noun_top} {price}원으로 산정되었고, {other}는 미측정 상태이다.",
    "일정은 {day}에서 {n}일 순연되었다.",
    "리스크는 {noun} 지연이며, 대응은 {other} 일정 재배치이다.",
    "참석자는 {name} 외 {n}명이었고 결정은 {time}에 확정되었다.",
    "산출물은 {noun} 점검표와 {other} 대조표이다.",
    "이전 공유 수치와 산출 기준이 달라졌으므로 기준 변경을 명시한다.",
    "{place_euro} 이동하는 데 {station} 환승이 필요하다.",
    "연령 {age}세 {background} 실무자가 현장 기록을 작성하였다.",
)

PRO_CLOSE = (
    "차기 점검은 {day} {time}으로 잡는다.",
    "미확인 항목은 {other}이며 추가 실측이 필요하다.",
    "결정이 필요하면 {name} 담당에게 회신한다.",
    "본 기록은 {topic} 범위로 한정한다.",
    "EOD",
)

PRO_ONLINE_EXTRA = (
    "슬랙 스레드에 {noun} 로그와 재현 절차를 남겼다.",
    "이슈 번호 {n}번에 {other} 재현 조건을 첨부하였다.",
    "화상회의 녹화본은 {day} {time} 일정과 함께 공유한다.",
    "원격 접속 권한은 {name}에게 한시적으로 부여한다.",
)

PRO_OFFLINE_EXTRA = (
    "현장 화이트보드에 {noun} 수치를 적고 사진으로 남겼다.",
    "대면 인수인계에서 {other} 캐비닛 위치를 확인하였다.",
    "회의실에서 출력본을 기준으로 항목을 대조하였다.",
    "실험실 출입 시 보호구를 착용하고 {noun_obj} 점검하였다.",
)

# Extra paragraphs for half-page (a few paragraphs) documents.
CASUAL_P2 = (
    "그 뒤로 {name}이랑 다시 연락했는데 {noun} 건이 생각보다 길어졌어. {other_obj} 먼저 정리하고 나니까 숨이 트이더라. {age}살에 {background} 하면서 이런 일이 한두 번이 아니야.",
    "집에 와서 {name2}한테 전화로 설명했는데 말이 잘 안 풀렸어. {place2}에서 본 {noun_and} 분위기가 달랐거든. 그래서 사진을 다시 보내 달라고 했어.",
    "사실 {topic} 자체보다 사람 대응이 더 힘들었어. {name_subj} 바쁘다 하고 {n2}시간이나 답이 없더라. 나는 그냥 {price2}원 기준으로 다시 셈했어.",
    "중간에 {name2}가 {place2} 근처라고 해서 잠깐 들렀어. {other} 이야기를 꺼내니까 표정이 좀 굳더라고. 그날은 더 밀어붙이지 않기로 했어.",
)

CASUAL_P3 = (
    "지금은 {noun_obj} 한 번 더 보고 결정하려고. {station} 타고 {place_euro} 가는 길에 메모해 둘게. 너도 {other} 쪽 생각 있으면 알려 줘.",
    "다음 주에는 {name}이랑 {place2}에서 다시 만나기로 했어. {topic}이 한 번에 끝나진 않을 것 같아. 그래도 오늘은 이 정도로 충분해.",
    "나중에 생각하면 {noun}보다 {other}가 더 기억에 남아. {background}로 살다 보면 이런 장면이 계속 겹치거든. 일단 오늘은 쉬려고.",
    "{day} 일을 {name2}한테도 짧게 공유했어. 반응은 담담했고 나도 더 보태지 않았어. {noun_top} 내일 아침에 다시 보면 될 것 같아.",
)

FORMAL_P2 = (
    "이후 {name2} 담당과 {place2}에서 추가로 확인하였습니다. {noun} 일정은 {n2}일로 조정되었고, {other} 비용은 {price2}원으로 안내받았습니다. 관련 자료는 메일로 먼저 보내 두었습니다.",
    "대기 과정에서 {topic} 관련 안내문을 다시 읽었습니다. {station} 이용 시 {time} 전후가 혼잡하였고, {n}분 정도 여유가 필요했습니다. {background} 일정과 겹치지 않도록 시간을 옮겼습니다.",
    "{name} 담당자는 {noun_obj} 우선 처리하자고 하였고, 저는 {other} 확인을 요청하였습니다. 두 항목을 같은 날 끝내기는 어렵다는 설명이었습니다. 내일 {time}에 이어서 진행하기로 하였습니다.",
)

FORMAL_P3 = (
    "현재까지 확인한 내용은 위와 같습니다. {place}에서 받은 번호와 {n2}번 접수 내역을 대조해 주시면 감사하겠습니다. 추가 서류가 필요하면 {name2}에게 전달하겠습니다.",
    "정리하면 {noun} 건은 진행 중이고 {other} 건은 대기입니다. {age}세 {background} 입장에서 일정 조율이 가장 급합니다. 회신 주시면 바로 반영하겠습니다.",
    "내일 {place2}에 다시 방문할 예정입니다. {topic_obj} 마감한 뒤에 결과만 짧게 공유하겠습니다. 잘못된 부분이 있으면 지적해 주시기 바랍니다.",
)

PRO_P2 = (
    "현장 메모를 기준으로 {noun} 수치와 {other} 기록을 다시 대조하였다. {place2} 구간은 {n2}분 지연이 있었고 산출 기준은 {day} 공유분과 다르다. 기준 변경을 별도로 명시한다.",
    "{name2}가 작성한 점검표에 {noun} 항목 {n}건이 빠져 있었다. 누락분은 {price2}원 산정에 영향을 주므로 재작성한다. 권한은 {name}에게 한시적으로 둔다.",
    "리스크는 {noun} 일정 순연과 {other} 미측정이다. 대응은 {place2} 재방문과 {n2}일 내 실측이다. 이전 공유 수치와 직접 비교하지 않는다.",
)

PRO_P3 = (
    "차기 점검은 {day} {time}, 장소는 {place2}로 잡는다. 참석자는 {name}·{name2}이며 산출물은 {noun} 점검표 개정본이다. 본 기록은 {topic} 범위로 한정한다.",
    "미확인은 {other}뿐이며 추가 실측 전에는 확정치로 쓰지 않는다. 결정이 필요하면 {name} 담당에게 회신한다. EOD",
    "{background} 실무 기록으로 {age}세 기준으로 작성하였다. {station} 이동 시간은 산정에서 제외한다. 다음 공유 때 기준을 한 줄로 반복한다.",
)

CASUAL_P4 = (
    "{day2}에는 {place3}에도 가 봤어. {name3}이 {noun_obj} 알고 있더라고. {n3}분 이야기하고 헤어졌어.",
    "밤에 {name2}랑 통화하면서 {topic}을 다시 짚었어. {time2}까지 얘기했는데도 결론은 안 났어. 그래도 {other_obj} 빼면 큰일은 아니야.",
    "돈을 다시 세어 보니 {price}원이 맞고 {price2}원은 다른 항목이었어. 헷갈려서 {name}한테 한 번 더 물어봤어.",
    "{station}에서 {name3}을 우연히 만났는데 {place2} 이야길 먼저 꺼내더라. 내가 {noun} 말한 줄 알고 당황했어.",
)

CASUAL_P5 = (
    "정리하면 {topic}은 하루로 안 끝났어. {noun_obj} 기준으로 내일 다시 보고, {other}는 천천히 볼게.",
    "주변에서는 {place}가 편하대. 나는 {place2}가 더 익숙해서 거기로 갈 것 같아. {name}만 시간 되면 돼.",
    "이런 일을 {age}살에 또 겪을 줄은 몰랐어. {background}로 사는 동안 {topic}은 몇 번이고 반복되는 느낌이야.",
    "마지막으로 {n2}시 전에 {noun} 사진만 받아 두면 돼. 나머지는 {day2}에 이어서 하자.",
)

FORMAL_P4 = (
    "{day2} {time2}에 {name3}과 유선으로 재확인하였습니다. {noun} 건은 {n3}건으로 정정되었고 {other}는 변동이 없었습니다.",
    "{place3} 창구에서도 같은 안내를 들었습니다. 수수료는 {price2}원이었고 처리 시간은 {n3}분 정도로 보였습니다.",
    "첨부한 메모에는 {topic} 일정과 {name2} 연락처를 적어 두었습니다. {time} 이후로는 연락이 어려울 수 있습니다.",
)

FORMAL_P5 = (
    "금일 확인분과 {day2} 확인분을 구분해 기록해 두었습니다. 혼동이 있으면 {name}에게 먼저 문의 부탁드립니다.",
    "{background} 일정과 겹치는 구간은 {n2}시 전후입니다. 가능하면 그 시간을 피해 주시면 감사하겠습니다.",
    "추가 방문은 {place3}를 우선으로 하겠습니다. {station} 이용이 어려우면 다른 경로를 알려 주십시오.",
)

PRO_P4 = (
    "{day2} 재측정에서 {noun} 값은 {n3}으로 나왔고 {other}는 여전히 미측정이다. 두 회차의 기준이 다르므로 표를 분리한다.",
    "{name3}이 남긴 로그에 {place3} 구간 {n2}분 지연이 있다. 이동 시간은 본 산정에서 제외한다.",
    "권한 회수는 {time2}로 잡는다. 그 전까지 {name2}만 {noun} 항목을 수정한다.",
)

PRO_P5 = (
    "공유본에는 {topic} 범위와 제외 항목을 한 줄로 적는다. {other} 추정치는 넣지 않는다.",
    "차기 자료는 {place3} 점검표와 {name} 회신이다. 배포는 {day2} {time2} 이후이다.",
    "본 메모의 수치는 {day} 기준이다. {day2} 재산출이 있으면 기준 변경을 명시한다.",
)

CASUAL_SECTION = (
    "{topic}, 처음",
    "그날 이어서",
    "다른 사람",
    "숫자와 일정",
    "다음에 할 일",
)
FORMAL_SECTION = (
    "방문 경위",
    "현장 확인",
    "비용과 일정",
    "추가 협의",
    "이후 계획",
)
PRO_SECTION = (
    "경위",
    "현장 점검",
    "수치 대조",
    "리스크",
    "차기",
)

# Spoken lines that encode 반말/존댓말 between generations.
DIALOGUES: dict[str, tuple[str, ...]] = {
    "child_to_parent": (
        "엄마, {noun} 때문에 늦었어요. {place}에서 {other_obj} 보고 갈게요.",
        "아버지, {addressee_age}세까지 일하시면 힘드시죠. 오늘은 제가 {noun_obj} 할게요.",
        "말씀하신 {noun} 확인했어요. {time}까지는 들어갈게요.",
    ),
    "parent_to_child": (
        "{name_voc}, {noun}은 네가 먼저 해. 엄마는 {other} 볼게.",
        "너는 {age}살이야. {topic} 정도는 혼자 해도 돼.",
        "{name_voc}, {time} 전에 들어와. {noun}은 내일 해도 늦지 않아.",
    ),
    "grandchild_to_grandparent": (
        "할머니, 저 {place}에서 {noun_obj} 사 왔어요. 따뜻할 때 드세요.",
        "할아버지, 계단 조심하세요. {noun}은 제가 들게요.",
        "오늘 {topic} 때문에 늦었어요. 먼저 앉아서 쉬세요.",
    ),
    "grandparent_to_grandchild": (
        "그래, {name_voc}. {noun}은 여기 두고 손부터 씻어.",
        "천천히 해. 할아버지는 {other}만 보면 돼.",
        "{name_voc}, {n}시만 넘기지 마. 밥은 남겨 둘게.",
    ),
    "junior_to_senior": (
        "선배, {noun} 건 제가 먼저 볼게요. 검토해 주시겠어요?",
        "팀장님, {time}까지 {other_obj} 올려 두었습니다. 확인 부탁드립니다.",
        "{addressee}께 {noun} 일정만 여쭤봐도 될까요?",
    ),
    "senior_to_junior": (
        "{name_voc}, {noun}은 네가 맡아. 막히면 그때 와.",
        "이번엔 {other_obj} 먼저 해. 보고는 {time}에 주면 돼.",
        "괜찮아, 그 정도면 됐어. {noun}만 내일 이어서 하자.",
    ),
    "student_to_teacher": (
        "선생님, {noun} 범위가 {n}번까지 맞나요? 다시 여쭤볼게요.",
        "교수님, {topic} 관련해서 질문 하나 드려도 될까요?",
        "과제 {noun}은 오늘 안에 제출할게요.",
    ),
    "teacher_to_student": (
        "{name_voc}, {noun}부터 다시 해 봐. {other}는 다음에 하자.",
        "질문은 좋은데, {n}쪽을 먼저 읽고 와.",
        "오늘은 이 정도면 됐어. 내일 {noun} 확인하자.",
    ),
    "peers_close": (
        "야, {noun} 진짜 그거 맞아? 나 {place}에서 다르게 봤는데.",
        "{name_voc}, {time}에 {place2}에서 보자. {other}는 그때 얘기해.",
        "그냥 {noun_obj} 기준으로 가자. 너무 오래 끌지 마.",
    ),
    "peers_distant": (
        "처음 뵙는데요, {noun} 관련해서 여쭤봐도 될까요?",
        "같은 {topic} 보시는 거죠? {place}에서 잠깐 이야기할까요?",
        "제 이름은 {name}이에요. {other}는 나중에 나눠요.",
    ),
    "in_law_younger": (
        "어머님, {noun}은 제가 할게요. 앉아서 좀 쉬세요.",
        "장인어른, {place} 교통이 복잡해요. {time}에 출발하시는 게 좋겠어요.",
        "{addressee}께 {topic} 일정만 미리 말씀드릴게요.",
    ),
    "staff_to_customer": (
        "손님, {noun}은 이쪽으로 오시면 됩니다. {n}번 창구입니다.",
        "잠시만 기다려 주세요. {other} 확인하는 데 {n2}분 걸립니다.",
        "영수증과 {noun} 내역을 함께 드리겠습니다.",
    ),
}


def to_haeyo(text: str) -> str:
    """Lift 반말 sentence endings to 해요체. Do not restack already-polite forms."""
    pairs = (
        ("아니야", "아니에요"),
        ("해야 돼", "해야 돼요"),
        ("더라고", "더라고요"),
        ("거든", "거든요"),
        ("거야", "거예요"),
        ("을게", "을게요"),
        ("할게", "할게요"),
        ("줄게", "줄게요"),
        ("볼게", "볼게요"),
        ("갈게", "갈게요"),
        ("었어", "었어요"),
        ("았어", "았어요"),
        ("였어", "였어요"),
        ("졌어", "졌어요"),
        ("갔어", "갔어요"),
        ("났어", "났어요"),
        ("됐어", "됐어요"),
        ("쳤어", "쳤어요"),
        ("렸어", "렸어요"),
        ("랐어", "랐어요"),
        ("왔어", "왔어요"),
        ("봤어", "봤어요"),
        ("같아", "같아요"),
        ("많아", "많아요"),
        ("좋아", "좋아요"),
        ("싫어", "싫어요"),
        ("힘들어", "힘들어요"),
        ("괜찮아", "괜찮아요"),
        ("줘.", "주세요."),
        ("줘,", "주세요,"),
        ("마.", "마세요."),
        ("봐.", "보세요."),
        ("와.", "오세요."),
        ("지.", "죠."),
        ("지?", "죠?"),
        ("이야.", "이에요."),
        ("이야,", "이에요,"),
    )
    out = text
    for src, dst in sorted(pairs, key=lambda kv: -len(kv[0])):
        if dst.startswith(src):
            out = re.sub(re.escape(src) + r"(?!요)", dst, out)
        else:
            out = out.replace(src, dst)
    out = re.sub(r"(?<![요세])해(?=[.!?])", "해요", out)
    out = re.sub(r"(?<![요])돼(?=[.!?])", "돼요", out)
    return out


def dialogue_for(spec: DiversitySpec, facts: dict[str, str], rng: random.Random) -> str:
    bank = DIALOGUES.get(spec.relation)
    if not bank:
        return ""
    line = _fill(str(_pick(rng, bank)), facts)
    if spec.speech_level == "jondaet" and spec.register == "casual":
        # already written in 해요체 in the bank
        return line
    return line

AI_TELL_PREFIXES = (
    "결론적으로, ",
    "요약하면, ",
    "또한, ",
    "따라서, ",
    "이를 통해 ",
)
def _ai_tell_clause(noun: str, kind: int) -> str:
    kind = kind % 5
    if kind == 0:
        return f"{eul_reul(noun)} 논의해 볼 필요가 있다."
    if kind == 1:
        return f"{eun_neun(noun)} 매우 중요하다고 할 수 있다."
    if kind == 2:
        return f"{eun_neun(noun)} 시사하는 바가 크다."
    if kind == 3:
        return f"{i_ga(noun)} 판단되어진다."
    return f"{eun_neun(noun)} 강력한 경쟁력을 가지고 있다."
AI_TELL_EMDASH = " — "


def apply_speech_level(text: str, spec: DiversitySpec) -> str:
    """Casual 존댓말 → 해요체. Formal/professional keep written register."""
    if spec.register == "casual" and spec.speech_level == "jondaet":
        return to_haeyo(text)
    return text


def _join(parts: list[str]) -> str:
    text = " ".join(p.strip() for p in parts if p and p.strip())
    return text.replace("  ", " ").strip()


def _fill(template: str, facts: dict[str, str]) -> str:
    return template.format(**facts)


def _voice_age(text: str, spec: DiversitySpec, rng: random.Random) -> str:
    """Light lexical color by age; keep particles and finite endings."""
    if spec.register != "casual":
        return text
    if spec.age <= 22:
        extras = (
            " 진짜 그랬어.",
            " 완전 당황했어.",
            " 요즘 그런 일이 많아.",
        )
    elif spec.age <= 40:
        extras = (
            " 요즘 이런 일이 잦다.",
            " 일단 기록해 둘게.",
            " 시간이 빠듯했어.",
        )
    elif spec.age <= 55:
        extras = (
            " 예전하고는 분위기가 다르다.",
            " 아이들 일정까지 겹쳐서 힘들었어.",
            " 크게 문제는 아니야.",
        )
    else:
        extras = (
            " 천천히 다시 보면 이해가 가.",
            " 젊은 친구들 말이 빨라서 한 번 더 물었어.",
            " 크게 서두를 일은 아니야.",
        )
    if rng.random() < 0.55:
        text = text + extras[rng.randrange(len(extras))]
    return text


def _inject_ai_tell(text: str, spec: DiversitySpec, rng: random.Random) -> str:
    """Seed translationese / AI-tell for the polish stage to remove."""
    prefix = AI_TELL_PREFIXES[spec.doc_id % len(AI_TELL_PREFIXES)]
    noun = _nouns(spec.topic)[spec.doc_id % len(_nouns(spec.topic))]
    sentences = [s.strip() for s in text.replace("!", ".").split(".") if s.strip()]
    if not sentences:
        return prefix + text
    mid = sentences[len(sentences) // 2]
    # Standalone AI-tell sentence for polish to drop. Identity lives in JSON `id`.
    tell = _ai_tell_clause(noun, spec.doc_id // 3)
    if spec.register in ("formal", "professional"):
        # Seed endings lint.py --fix will rewrite (됐다/했다).
        tell = tell + " 확인됐다. 정리했다."
        if spec.register == "professional":
            tell = "본 문서는 " + eul_reul(spec.topic) + " 다룬다. " + tell
    glued = prefix + mid + "."
    if rng.random() < 0.7:
        glued = glued + AI_TELL_EMDASH + tell
    else:
        glued = glued + " " + tell
    out = []
    for i, s in enumerate(sentences):
        if i == len(sentences) // 2:
            out.append(glued.rstrip("."))
        else:
            out.append(s)
    return ". ".join(out).replace("..", ".") + "."


class WriterAgent:
    name = "base"

    def openings(self, spec: DiversitySpec) -> tuple[str, ...]:
        raise NotImplementedError

    def middles(self, spec: DiversitySpec) -> tuple[str, ...]:
        raise NotImplementedError

    def closings(self, spec: DiversitySpec) -> tuple[str, ...]:
        raise NotImplementedError

    def extras(self, spec: DiversitySpec) -> tuple[str, ...]:
        return ()

    def mid_paragraphs(self, spec: DiversitySpec) -> tuple[tuple[str, ...], ...]:
        return ((), ())

    def section_titles(self) -> tuple[str, ...]:
        return ()

    def write(self, spec: DiversitySpec, rng: random.Random, form: str = "short") -> str:
        facts = _facts(spec, rng)
        if form in ("halfpage", "onepage", "fivepage"):
            return self._write_longform(spec, rng, facts, form)
        n_mid = 2 + (spec.doc_id % 3)
        opening = _fill(str(_pick(rng, self.openings(spec))), facts)
        mids = []
        pool = list(self.middles(spec))
        rng.shuffle(pool)
        for tmpl in pool[:n_mid]:
            mids.append(_fill(tmpl, facts))
        extra_pool = list(self.extras(spec))
        if extra_pool:
            mids.append(_fill(str(_pick(rng, extra_pool)), facts))
        closing = _fill(str(_pick(rng, self.closings(spec))), facts)
        body = _join([opening] + mids + [closing])
        body = _voice_age(body, spec, rng)
        spoken = dialogue_for(spec, facts, rng)
        if spoken:
            body = body + " " + spoken
        body = _inject_ai_tell(body, spec, rng)
        body = apply_speech_level(body, spec)
        if not body.endswith(("다", "다.", "요", "요.", "까", "까.", "어", "어.", "지", "지.", "게", "게.", "D")):
            if not body.endswith("."):
                body += "."
        return body

    def _write_longform(
        self, spec: DiversitySpec, rng: random.Random, facts: dict, form: str
    ) -> str:
        """onepage ≈ 8–12 paras; fivepage ≈ 5 sections × several paras."""
        if form == "fivepage":
            n_paras, n_sections = 34 + (spec.doc_id % 6), 5
        else:
            # halfpage is now a 1-page document
            n_paras, n_sections = 10 + (spec.doc_id % 4), 1

        pool = list(self.middles(spec))
        rng.shuffle(pool)
        extras = list(self.extras(spec))
        banks = [b for b in self.mid_paragraphs(spec) if b]
        filled: list[str] = []
        filled.append(
            _join(
                [
                    _fill(str(_pick(rng, self.openings(spec))), facts),
                    _fill(pool[0], facts) if pool else "",
                    _fill(pool[1], facts) if len(pool) > 1 else "",
                ]
            )
        )
        # Rotate banks and leftover middles so long docs stay distinct.
        source: list[str] = []
        for bank in banks:
            source.extend(_fill(t, facts) for t in bank)
        source.extend(_fill(t, facts) for t in extras)
        source.extend(_fill(t, facts) for t in pool[2:])
        rng.shuffle(source)

        def _take(n: int = 2) -> str:
            bits: list[str] = []
            for _ in range(n):
                if not source:
                    refill = [
                        _fill(str(_pick(rng, bank)), facts)
                        for bank in banks
                        if bank
                    ]
                    if extras:
                        refill.append(_fill(str(_pick(rng, extras)), facts))
                    if pool:
                        refill.append(_fill(str(_pick(rng, pool)), facts))
                    rng.shuffle(refill)
                    source.extend(refill)
                if source:
                    bits.append(source.pop())
            return _join(bits)

        while len(filled) < n_paras - 1:
            filled.append(_take(3 if form == "fivepage" else 2))
        close = _voice_age(_fill(str(_pick(rng, self.closings(spec))), facts), spec, rng)
        filled.append(close)
        filled = [p for p in filled if p]
        spoken = dialogue_for(spec, facts, rng)
        if spoken:
            filled.insert(min(2, len(filled)), spoken)
        mid_i = min(2, len(filled) - 1)
        filled[mid_i] = _inject_ai_tell(filled[mid_i], spec, rng)
        filled = [apply_speech_level(p, spec) for p in filled]

        titles = self.section_titles()
        if n_sections > 1 and titles:
            chunks: list[list[str]] = [[] for _ in range(n_sections)]
            for i, para in enumerate(filled):
                chunks[i % n_sections].append(para)
            blocks = []
            for i, chunk in enumerate(chunks):
                title = titles[i % len(titles)]
                if spec.register == "casual":
                    head = title
                elif spec.register == "formal":
                    head = title
                else:
                    head = title
                blocks.append(head + "\n\n" + "\n\n".join(chunk))
            return "\n\n".join(blocks)
        return "\n\n".join(filled)


class CasualWriterAgent(WriterAgent):
    name = "casual"

    def openings(self, spec: DiversitySpec) -> tuple[str, ...]:
        return CASUAL_OPEN

    def middles(self, spec: DiversitySpec) -> tuple[str, ...]:
        return CASUAL_MID

    def closings(self, spec: DiversitySpec) -> tuple[str, ...]:
        return CASUAL_CLOSE

    def extras(self, spec: DiversitySpec) -> tuple[str, ...]:
        return CASUAL_ONLINE_EXTRA if spec.environment == "online" else CASUAL_OFFLINE_EXTRA

    def mid_paragraphs(self, spec: DiversitySpec) -> tuple[tuple[str, ...], ...]:
        return CASUAL_P2, CASUAL_P3, CASUAL_P4, CASUAL_P5

    def section_titles(self) -> tuple[str, ...]:
        return CASUAL_SECTION


class FormalWriterAgent(WriterAgent):
    name = "formal"

    def openings(self, spec: DiversitySpec) -> tuple[str, ...]:
        return FORMAL_OPEN

    def middles(self, spec: DiversitySpec) -> tuple[str, ...]:
        return FORMAL_MID

    def closings(self, spec: DiversitySpec) -> tuple[str, ...]:
        return FORMAL_CLOSE

    def extras(self, spec: DiversitySpec) -> tuple[str, ...]:
        return FORMAL_ONLINE_EXTRA if spec.environment == "online" else FORMAL_OFFLINE_EXTRA

    def mid_paragraphs(self, spec: DiversitySpec) -> tuple[tuple[str, ...], ...]:
        return FORMAL_P2, FORMAL_P3, FORMAL_P4, FORMAL_P5

    def section_titles(self) -> tuple[str, ...]:
        return FORMAL_SECTION


class ProfessionalWriterAgent(WriterAgent):
    name = "professional"

    def openings(self, spec: DiversitySpec) -> tuple[str, ...]:
        return PRO_OPEN

    def middles(self, spec: DiversitySpec) -> tuple[str, ...]:
        return PRO_MID

    def closings(self, spec: DiversitySpec) -> tuple[str, ...]:
        return PRO_CLOSE

    def extras(self, spec: DiversitySpec) -> tuple[str, ...]:
        return PRO_ONLINE_EXTRA if spec.environment == "online" else PRO_OFFLINE_EXTRA

    def mid_paragraphs(self, spec: DiversitySpec) -> tuple[tuple[str, ...], ...]:
        return PRO_P2, PRO_P3, PRO_P4, PRO_P5

    def section_titles(self) -> tuple[str, ...]:
        return PRO_SECTION


AGENTS: dict[str, WriterAgent] = {
    "casual": CasualWriterAgent(),
    "formal": FormalWriterAgent(),
    "professional": ProfessionalWriterAgent(),
}


def instruction_for(spec: DiversitySpec, form: str = "short") -> str:
    env_ko = "온라인" if spec.environment == "online" else "오프라인"
    reg_ko = {"casual": "캐주얼", "formal": "격식", "professional": "업무·보고"}[spec.register]
    length = ""
    if form in ("halfpage", "onepage"):
        length = "A4 한 페이지 분량으로, 문단 여덟 개에서 열두 개 사이로 써라. "
    elif form == "fivepage":
        length = "A4 다섯 페이지 분량으로, 절을 다섯 개로 나누어 써라. "
    level_ko = "반말" if spec.speech_level == "banmal" else "존댓말"
    return (
        f"다음 조건에 맞는 한국어 글을 작성하라. "
        f"주제는 {spec.topic}이고, 환경은 {env_ko}이며, "
        f"문체는 {reg_ko}이다. 화자는 {spec.age}세 {spec.background}이다. "
        f"상대는 {spec.addressee_age}세 {spec.addressee}이고, "
        f"관계는 {spec.relation}이며 {level_ko}로 말하라. "
        f"{length}"
        f"원어민이 쓴 것처럼 자연스럽게 쓰되, 조건을 지키라."
    )


def generate_document(doc_id: int, form: str = "short") -> dict:
    spec = spec_for_id(doc_id)
    seed_base = {"fivepage": 80_000, "onepage": 50_000, "halfpage": 50_000}.get(
        form, 10_000
    )
    seed = seed_base + doc_id * 17
    rng = random.Random(seed)
    agent = AGENTS[spec.register]
    body = agent.write(spec, rng, form=form)
    return {
        "id": doc_id,
        "topic": spec.topic,
        "environment": spec.environment,
        "register": spec.register,
        "age": spec.age,
        "background": spec.background,
        "agent": agent.name,
        "form": form,
        "relation": spec.relation,
        "speech_level": spec.speech_level,
        "generation": spec.generation,
        "addressee": spec.addressee,
        "addressee_age": spec.addressee_age,
        "instruction": instruction_for(spec, form=form),
        "body": body,
    }


def _load_existing(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    found: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            found[int(rec["id"])] = rec
    return found


def generate_corpus(
    path: Path = RAW_PATH,
    count: int = TARGET_COUNT,
    resume: bool = True,
    progress: Callable[[int, int], None] | None = None,
    form: str = "short",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(path) if resume else {}
    missing = [i for i in range(count) if i not in existing]
    if not missing and len(existing) >= count:
        return path
    # Rewrite in id order so resume stays compact and ordered.
    records = dict(existing)
    for i, doc_id in enumerate(missing):
        records[doc_id] = generate_document(doc_id, form=form)
        if progress:
            progress(i + 1, len(missing))
    with path.open("w", encoding="utf-8") as fh:
        for doc_id in range(count):
            fh.write(json.dumps(records[doc_id], ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Generate the raw Korean corpus")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--count", type=int, default=None)
    p.add_argument(
        "--length",
        choices=("short", "onepage", "fivepage", "halfpage"),
        default="short",
        help="halfpage is an alias of onepage",
    )
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)
    form = "onepage" if args.length == "halfpage" else args.length
    defaults = {
        "onepage": (ONEPAGE_RAW_PATH, ONEPAGE_COUNT),
        "fivepage": (FIVEPAGE_RAW_PATH, FIVEPAGE_COUNT),
        "short": (RAW_PATH, TARGET_COUNT),
    }
    default_out, default_count = defaults[form]
    out = args.out or default_out
    count = args.count if args.count is not None else default_count
    generate_corpus(path=out, count=count, resume=not args.no_resume, form=form)
    print(f"wrote {count} {form} documents to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
