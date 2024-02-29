import re

seasons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def first_word(text, deli=" "):
    for i, t in enumerate(text):
        if t == deli:
            break
    return text[:i]


def remove_pattern(pattern, full_log):
    for s in re.finditer(pattern, full_log):
        a, b = s.span()
        full_log = (full_log[:a] + full_log[b:]).strip()
    return full_log


def refine_data(full_log):
    t = first_word(full_log)
    # 첫 단어가 연도를 나타낼때,
    if len(t) == 4 and t.isdigit() and t[:2] in ("20", "21"):
        # 실제로 달라지는 부분은 WinEvtLog: 다음부분(idx=5)
        full_log = full_log[5:].strip()

    t = first_word(full_log)
    # 첫 단어가 12개월의 약자일 때,
    if len(t) == 3 and t in seasons:
        # 실제로 달라지는 부분은 localhost 다음부분(idx=4)
        full_log = full_log[4:].strip()

        t = first_word(full_log)
        if t.isdigit():
            full_log = full_log[len(t) + 1: ].strip()

    # 00:00:00 형식의 시간 이면?
    if re.match(r"\d{2}:\d{2}:\d{2}", full_log):
        full_log = full_log[9:].strip()

    if full_log.startswith("localhost"):
        full_log = full_log[10:].strip()

    # @timestamp: "~~~~Z"
    full_log = re.sub(r'"@timestamp"\s?:\s?"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?",?', '', full_log)

    # "pid": "4567"
    full_log = re.sub(r'"pid"\s?:\s?\d+,?', 'pid', full_log)
    # [pid]
    full_log = re.sub(r"\[\d+\]", 'pid', full_log)
    # PID=4567 or pid=4567
    full_log = re.sub(r'(PID|pid)\s?=\s?\d+', 'pid', full_log)

    # 시간 : 15:00
    full_log = re.sub(r'\d{1:2}:\d{1:2}', '', full_log)
    # =숫자 제거
    full_log = re.sub(r'=\s?\d+', '', full_log)
    # 줄바꿈 문자 - corpus에서 \\n이 \n 처럼 쓰이는 것을 확인(Jan, Feb, Mar 등등)
    full_log = re.sub(r'\n|\\n',' ',full_log)


    # 띄어쓰기를 한번으로 통일
    full_log = re.sub(r"\s+", " ", full_log)

    return full_log.lower()