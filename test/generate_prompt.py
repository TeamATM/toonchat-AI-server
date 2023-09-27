import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.llm.conversations import get_conv_template

conv = get_conv_template("Toonchat_v2.1")
conv.append_message(conv.roles[0], "hi")
conv.append_message(conv.roles[1], "hi, how can i help you?")
conv.append_message(conv.roles[0], "hi")
conv.append_message(conv.roles[1], "hi, how can i help you?")
conv.append_message(conv.roles[2], "system message")
conv.append_message(conv.roles[0], "hi")
conv.append_message(conv.roles[1], "hi, how can i help you?")

print(conv.get_prompt())
