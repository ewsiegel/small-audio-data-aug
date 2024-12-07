import pandas as pd

def get_emotion_to_wav():
    dir = "train_small"
    df = pd.read_csv(f"{dir}/train_small.csv")
    emotion_to_wav = {}
    for _, row in df.iterrows():
        emotion = row['Emotion']
        filename = row['filename']
        if emotion not in emotion_to_wav:
            emotion_to_wav[emotion] = []
        emotion_to_wav[emotion].append(f"{dir}/{filename}")
    return emotion_to_wav