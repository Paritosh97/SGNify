import json
import pickle
import numpy as np
import scipy
from pathlib import Path
from sklearn.impute import SimpleImputer
from utils.rps import find_rps

def compute_sign_class(*, result_folder, openpose_folder, segment_path, sign_class_path):
    find_rps(result_path=result_folder, sign_class="-1", out_prefix="sign_classification", segment_path=segment_path)

    with segment_path.open() as json_file:
        segment = json.load(json_file)

    keypoints = []
    for j in range(segment["t1"], segment["t4"] + 1):
        with openpose_folder.joinpath(f"{j:03d}_keypoints.json").open() as json_file:
            json_data = json.load(json_file)
        if json_data["people"]:
            keypoints.append(json_data["people"][0]["pose_keypoints_2d"])

    x, y, c = np.transpose(np.reshape(keypoints, (-1, 25, 3)), (2, 1, 0))
    c_r = ~np.isclose(c[4], 0)
    c_l = ~np.isclose(c[7], 0)
    h = np.median(y[1] - y[0])
    y_r = y[4][c_r] / h
    y_l = y[7][c_l] / h

    v1 = min(np.ptp(y_r) if len(y_r) else 0, np.ptp(y_l) if len(y_l) else 0)

    base_folder = result_folder.joinpath("rps", "sign_classification")
    with base_folder.joinpath("ref_right_1.pkl").open("rb") as file:
        right_1 = pickle.load(file)["right_hand_pose"][0]
    with base_folder.joinpath("ref_right_2.pkl").open("rb") as file:
        right_2 = pickle.load(file)["right_hand_pose"][0]
    with base_folder.joinpath("ref_left_1.pkl").open("rb") as file:
        left_1 = pickle.load(file)["left_hand_pose"][0]
    with base_folder.joinpath("ref_left_2.pkl").open("rb") as file:
        left_2 = pickle.load(file)["left_hand_pose"][0]

    v3 = scipy.spatial.distance.cosine(right_1, left_1)

    v5 = max(scipy.spatial.distance.cosine(right_1, right_2), scipy.spatial.distance.cosine(left_1, left_2))

    # Preprocessing step: Handle NaN or infinite values
    input_data = np.array([[v1, v3, v5]])

    # Check for NaN or infinite values
    if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
        raise ValueError("Input contains NaN or infinite values")

    # Optionally, impute NaN values with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    input_data = imputer.fit_transform(input_data)

    # Optionally, scale the data if values are too large
    input_data = np.clip(input_data, a_min=-1e5, a_max=1e5)

    with open("data/sign_classifier.pkl", "rb") as file:
        clf = pickle.load(file)

    sign_class_path.write_text(clf.predict(input_data)[0][:2])
