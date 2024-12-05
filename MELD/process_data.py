import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from typing import List

class MELD:
    def __init__(self, test_csv_name: str, train_csv_name: str, test_mp4_name: str, train_mp4_name: str):
        self.file_pattern = r"dia(\d+)_utt(\d+)"  # filename regex

        # csv file names
        self.test_csv = test_csv_name
        self.train_csv = train_csv_name

        # folder names
        self.test_mp4 = test_mp4_name
        self.train_mp4 = train_mp4_name

    def process(self) -> List[pd.DataFrame]:
        train_df = self._csv_to_parsed_df(self.train_csv)
        test_df = self._csv_to_parsed_df(self.test_csv)

        self._match_mp4_binary_to_df(train_df, self.train_mp4)
        self._match_mp4_binary_to_df(test_df, self.test_mp4)
        return [train_df, test_df]
    
    def _csv_to_parsed_df(self, fname): 
        df = pd.read_csv(fname)
        # List of columns to keep
        columns_to_keep = ["Emotion", "Dialogue_ID", "Utterance_ID", "Utterance"]

        # Drop columns not in columns_to_keep
        df = df.drop(
            [col for col in df.columns if col not in columns_to_keep],
            axis=1
        )
        df.groupby("Emotion").apply(lambda x: x)
        df = df.set_index(["Emotion"])
        df.sort_values(by=["Emotion", "Dialogue_ID", "Utterance_ID"], inplace=True)
        return df

    def _match_mp4_binary_to_df(self, df, mp4_folder):
        df["MP4_fname"] = None
        for file in os.listdir(mp4_folder):
            match = re.search(self.file_pattern, file)
            if match:
                dialogue_id = int(match.group(1))
                utterance_id = int(match.group(2))
                
                # locate the matching row in the DataFrame
                row_index = df[(df["Dialogue_ID"] == dialogue_id) & (df["Utterance_ID"] == utterance_id)].index
                if len(row_index) > 0:
                    df.loc[row_index, "MP4_fname"] = file
        df = df.drop(["Dialogue_ID", "Utterance_ID"], axis=1, inplace=True)

if __name__ == "__main__":
    # fill in with relevant file names
    meld_data = MELD("MELD/MELD.Raw/test_sent_emo.csv", "MELD/MELD.Raw/train_sent_emo.csv", "MELD/train_splits", "MELD/output_repeated_splits_test")
    [train_df, test_df] = meld_data.process()
    print(train_df, test_df)