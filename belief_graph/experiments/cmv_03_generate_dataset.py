# pip install convokit
import json
import pandas as pd
from convokit import Corpus, download

# the logic that downloads the ConvoKit corpus, calculates the Processing_Time_Minutes,
# and exports cmv_longitudinal_dataset.csv

# Download and load the complete r/changemyview dataset
name = "winning-args-corpus"
corpus = Corpus(filename=download(name))


def create_longitudinal_dataset_with_delta_time(corpus):
    longitudinal_data = []

    for convo in corpus.iter_conversations():
        try:
            root_utt = convo.get_utterance(convo.id)
            t1_text = root_utt.text
            real_op_id = root_utt.speaker.id
        except KeyError:
            continue

        for search_utt in convo.iter_utterances():
            if search_utt.meta.get("success") == 1:
                catalyst_id = search_utt.id
                catalyst_text = search_utt.text
                catalyst_time = search_utt.timestamp  # This is a valid Unix timestamp
                catalyst_speaker = search_utt.speaker.id

                if catalyst_speaker == real_op_id:
                    continue

                t2_text = None
                t2_time = None  # This is a valid Unix timestamp
                for reply in convo.iter_utterances():
                    if (
                        reply.reply_to == catalyst_id
                        and reply.speaker.id == real_op_id
                    ):
                        t2_text = reply.text
                        t2_time = reply.timestamp
                        break

                if t1_text and catalyst_text and t2_text:
                    longitudinal_data.append(
                        {
                            "Thread_ID": convo.id,
                            "OP_User_ID": real_op_id,
                            "Critic_User_ID": catalyst_speaker,
                            "T1_Original_Belief": t1_text,
                            "Catalyst_Timestamp": catalyst_time,
                            "Catalyst_Argument": catalyst_text,
                            "T2_Timestamp": t2_time,
                            "T2_Actual_Explanation": t2_text,
                        }
                    )
                    break

    df = pd.DataFrame(longitudinal_data)

    # Convert available timestamps to readable datetime objects
    df["Catalyst_Timestamp"] = pd.to_datetime(
        df["Catalyst_Timestamp"], unit="s"
    )
    df["T2_Timestamp"] = pd.to_datetime(df["T2_Timestamp"], unit="s")

    # Calculate cognitive processing time (minutes between the argument and the OP's change of mind)
    df["Processing_Time_Minutes"] = (
        df["T2_Timestamp"] - df["Catalyst_Timestamp"]
    ).dt.total_seconds() / 60

    print("Saving temporal dataset to CSV file...")
    df.to_csv("data/cmv_longitudinal_dataset.csv", index=False)
    return df


print("Processing dataset...")
clean_dataset = create_longitudinal_dataset_with_delta_time(corpus)
print(
    f"\nSuccessfully generated dataset with {len(clean_dataset)} rows containing processing times!"
)

# Display dataset structure
if not clean_dataset.empty:
    for i in range(2):
        print(f"\nRow {i} preview:")
        # 1. Extract the row as a dictionary first
        row = (
            clean_dataset.iloc[i][
                [
                    "T1_Original_Belief",
                    "Catalyst_Timestamp",
                    "Catalyst_Argument",
                    "T2_Timestamp",
                    "T2_Actual_Explanation",
                    "Processing_Time_Minutes",
                ]
            ]
            .to_dict()
        )

        # 2. Convert Pandas Timestamps to strings so json.dumps doesn't crash
        row["Catalyst_Timestamp"] = str(row["Catalyst_Timestamp"])
        row["T2_Timestamp"] = str(row["T2_Timestamp"])

        # 3. Safe to dump to JSON now
        print(json.dumps(row, indent=4))
        print(f"\n==========================================")
        print(f"   END OF POST #{i}")
        print(f"==========================================\n")