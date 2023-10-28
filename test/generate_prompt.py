import os
import sys
import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.data import PromptData, History, Message, GenerationArgs

prompt_data = PromptData(
    "This is a Persona",
    ["Reference text 1", "Reference text 2"],
    History(
        "history_id",
        "userId",
        0,
        [
            Message("message_id", None, datetime.datetime.now(), "Message content from user", True),
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
