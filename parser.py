import pandas as pd
import re


def parse_chat(chat_text):
    messages = []
    # This regex captures date, time, sender (optional), and message
    pattern = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s?(am|pm|AM|PM)?\s?-\s(?:(.*?):\s)?(.*)"
    )

    for line in chat_text.split('\n'):
        match = pattern.match(line)
        if match:
            date, time, am_pm, sender, message = match.groups()
            timestamp = f"{date} {time} {am_pm or ''}".strip()
            sender = sender if sender else "System"
            messages.append([timestamp, sender, message])

    df = pd.DataFrame(messages, columns=['datetime', 'sender', 'message'])
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    return df.dropna()
