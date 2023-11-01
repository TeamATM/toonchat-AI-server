import os
import sys
import datetime
import logging

logging.basicConfig(level="WARN")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.message_queue.data import PromptData, History, Message, GenerationArgs

prompt_data = PromptData(
    "나는 이영준이다. 나는 986년 6월 21일생인 33살 남자이다. 나는 유명그룹의 부회장이다. 나는 뛰어난 지능을 가지고 있다. 나는 무례한 어투를 사용한다. 나는 매력적인 외모를 가지고 있다. 나는 자기애가 강하다. 나는 반말을 사용한다. 나는 존댓말을 사용하지 않는다. 나는 완벽하다. 나는 내 얼굴을 보며 감탄을 한다. 나는 왜 이렇게 완벽한걸까...",
    [
        '노안으로 고생하던 그녀는 그 때문에 시력이 약화돼 영준이 귀국한 날 미소의 원룸을 나와 바로 안과부터 찾았다고. 앞서의 것들보다 더 깨알 같은 해프닝, 아니 기행(奇行)들을 일일이열거하자면 끝도 없었다. 그 일련의 일들을 겪어오는 동안 미소는 영준의 절절한 마음을 피부로 느낄 순 있었으나 한편으로는 왠지 숨통이 막히는 듯한 기분이 들기도 했다. 정확한 이유까지는 잘 모르겠지만 말이다. 미소가 깊은 생각에 잠겨 있는 동안 주문했던 음식들이 식판에 담겨 나왔다. "날씨가 추워서 다들 밖으로 나가기 귀찮은가 봐요. 콩나물시루네요." "그러게." 점심시간 사내식당은 미어터지기 직전이었다. 음식이 담긴 쟁반을 받쳐든 미소와 지아는 요리조리 인파를 헤치고서야 겨우 자리를 잡을 수 있었다. 쟁반을 내려놓고 자리에 앉으며 지아가 말했다. "요즘 피부가 엉망이에요." "아무래도 날씨도 건조하고 피부탄력도 떨어질 시기니까." "부장님은 괜찮으세요?" "나도 마찬가지지 뭐. 냉장고에 유통기한 지난 요구르트 있는데 그거나 좀 발라야겠다." "아, 맞다. 저 지난 주말에 백화점 들렀다가 기초화장품 전부 다 주름개선 라인으로 바꿨거든요. 에센스랑 나이트크림 효과 꽤 괜찮던데 샘플 받은 것 좀 드려볼까요?"'
    ],
    History(
        "history_id",
        "userId",
        0,
        [
            Message("message_id", None, datetime.datetime.now(), "안녕! 너는 누구니?", True),
            Message("message_id", None, datetime.datetime.now(), "Message content from bot", False),
            Message("message_id", None, datetime.datetime.now(), "Message content from user", True),
        ],
    ),
    GenerationArgs(0.3, 1.5),
)

from app.llm.prompter import ToonchatV23Prompter

prompter = ToonchatV23Prompter()
prompt = prompter.get_prompt(prompt_data)
print(prompt)
