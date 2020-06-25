import numpy as np
import pandas as pd

predictions_y_test = np.load("data/predictions_y_test.npy")
sample_submission = pd.read_csv("data/sample_submission.csv")

submission = pd.DataFrame({"ID": np.arange(len(predictions_y_test)), "item_cnt_month": predictions_y_test})

if sample_submission.shape[0] != submission.shape[0]:
    print(f"The number of observations in the sample submission does not match the number of samples in the submissions file: {sample_submission.shape[0]} != {submission.shape[0]} ")


submission.to_csv("data/submission.csv", index=False)

