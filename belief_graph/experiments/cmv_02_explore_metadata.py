# pip install convokit
import json
import pandas as pd
from convokit import Corpus, download

# Download and load the complete r/changemyview dataset
name = "winning-args-corpus"
corpus = Corpus(filename=download(name))

# Get the first conversation and its root utterance
first_convo = list(corpus.iter_conversations())[0]
sample_utt = first_convo.get_utterance(first_convo.id)

print("--- Utterance Attributes ---")
print(f"ID: {sample_utt.id}")
print(f"Speaker: {sample_utt.speaker.id}")
print(f"Timestamp property: {sample_utt.timestamp}")

print("\n--- Available Keys in Utterance __dict__ ---")
print(sample_utt.__dict__.keys())

print("\n--- Content of Utterance .meta ---")
print(json.dumps(sample_utt.meta, indent=4))

print("--- Inspecting the first 3 utterances from convo.iter_utterances() ---")

for i, utt in enumerate(first_convo.iter_utterances()):
    if i >= 3:
        break
    print(f"\n[Utterance #{i+1}]")
    print(f"ID: {utt.id}")
    print(f"Speaker: {utt.speaker.id}")
    print(f"Timestamp: {utt.timestamp}")
    print(f"Reply to: {utt.reply_to}")
    print(f"Text preview: {utt.text[:200]}...")
    print("-" * 40)