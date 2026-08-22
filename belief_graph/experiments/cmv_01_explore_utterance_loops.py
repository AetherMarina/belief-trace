# pip install convokit
from convokit import Corpus, download

# Downloads and loads the complete r/changemyview dataset
name = "winning-args-corpus"
corpus = Corpus(filename=download(name))

# print(corpus.meta)
# print(f"Number of conversations: {len(corpus.get_conversation_ids())}")
# print("Utterances:", len(corpus.get_utterance_ids()))
# # Inspect one utterance
# utt = next(corpus.iter_utterances())
# print("\nExample utterance:")
# print("Text:", utt.text[:500])
# print("Metadata:", utt.meta)

# Takes first two conversations from corpus
for_review = list(corpus.iter_conversations())[:2]

for i, convo in enumerate(for_review, 1):
    # The root comment (the first post) has the same ID as the entire conversation
    root_utt = convo.get_utterance(convo.id)
    pravi_op_id = root_utt.speaker.id

    print(f"\n========================================================================")
    print(f"   START OF POST #{i} (ID Topic: {convo.id}) | Author (OP): {pravi_op_id}")
    print(f"========================================================================")
    print(f"📝 MAIN POST (T1):\n{root_utt.text}")
    print(f"------------------------------------------------------------------------")
    print(f"💬 ALL COMMENTS INSIDE THIS TOPIC (Total: {len(list(convo.iter_utterances())) - 1}):")
    print(f"------------------------------------------------------------------------")

    # Go through all the other comments in this conversation
    for utt in convo.iter_utterances():
        if utt.id == convo.id:
            continue  # Skipping the main post because it is already printed above

        role = "OP" if utt.speaker.id == pravi_op_id else "Critic"
        success = "Assigned Delta (Success)" if utt.meta.get('success') == 1 else "No Delte"

        print(f"\n[ID: {utt.id}] -> Response on: {utt.reply_to}")
        print(f"👤 Author: {utt.speaker.id} ({role}) | Label: {success}")
        print(f"📄 Comment text:\n{utt.text}")
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print(f"\n========================================================================")
    print(f"  End of POST #{i}")
    print(f"========================================================================\n")