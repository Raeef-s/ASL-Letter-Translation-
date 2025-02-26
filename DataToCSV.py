import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

from mediapipe.tasks.python.vision import HandLandmarker
from pandas import DataFrame

print(np.__version__)
print(cv.__version__)
print(mp.__version__)

dataset_path = "/Dataset/asl_alphabet_train/asl_alphabet_train"

training_paths = [dataset_path + "/A",
                  dataset_path + "/B",
                  dataset_path + "/C",
                  dataset_path + "/D",
                  dataset_path + "/E",
                  dataset_path + "/F",
                  dataset_path + "/G",
                  dataset_path + "/H",
                  dataset_path + "/I",
                  dataset_path + "/J",
                  dataset_path + "/K",
                  dataset_path + "/L",
                  dataset_path + "/M",
                  dataset_path + "/N",
                  dataset_path + "/O",
                  dataset_path + "/P",
                  dataset_path + "/Q",
                  dataset_path + "/R",
                  dataset_path + "/S",
                  dataset_path + "/T",
                  dataset_path + "/U",
                  dataset_path + "/V",
                  dataset_path + "/W",
                  dataset_path + "/X",
                  dataset_path + "/Y",
                  dataset_path + "/Z",
                  dataset_path + "/del",
                  dataset_path + "/space"]

path_to_classification = {}
# assigns each training path as a key to its classification
for i in range(len(training_paths)-2):
    path_to_classification[training_paths[i]] = training_paths[i][-1]
path_to_classification[training_paths[26]] = "DELETE"
path_to_classification[training_paths[27]] = "SPACE"

# Creating HandLandmarker object
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)


def draw_hand_landmarks(image: np.ndarray, hand_landmarker_result) -> np.ndarray:
    """
    Draws hand landmarks on the given image.

    Parameters:
        image (np.ndarray): The input image.
        hand_landmarker_result: The hand landmark detection result from MediaPipe.

    Returns:
        np.ndarray: The image with hand landmarks drawn.
    """
    if not hand_landmarker_result.hand_landmarks:
        return image  # Return the original image if no hands are detected

    annotated_image = image.copy()

    # Define landmark connections (MediaPipe Hand Connections)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm connections
    ]

    for hand_landmarks in hand_landmarker_result.hand_landmarks:
        landmark_points = []

        # Convert landmark coordinates to image scale
        for landmark in hand_landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            landmark_points.append((x, y))

            # Draw each landmark as a circle
            cv.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

        # Draw connections between landmarks
        for start_idx, end_idx in connections:
            start_point = landmark_points[start_idx]
            end_point = landmark_points[end_idx]
            cv.line(annotated_image, start_point, end_point, (255, 0, 0), 2)

    return annotated_image





"""
with HandLandmarker.create_from_options(options) as landmarker:
    image = mp.Image.create_from_file(
        "C:/Users/raeef/OneDrive/Desktop/ML Projects/ASL Letter Translator Mediapipe/Dataset/asl_alphabet_train/asl_alphabet_train/A/A660.jpg")
    detection_result = landmarker.detect(image)

    for hand in detection_result.hand_landmarks:  # Each hand
        for landmark in hand:  # Each landmark
            print(i)
            print(landmark)


    annotated_image = draw_hand_landmarks(image.numpy_view(), detection_result)
    cv.imshow("Annotated A660", annotated_image)
    cv.waitKey(0)
"""



dataColumns = []
for i in range(106):
    dataColumns.append([])

with HandLandmarker.create_from_options(options) as landmarker:



    for training_path in training_paths:
        print(training_path)
        for filename in os.listdir(training_path):
            img_path = os.path.join(training_path, filename)
            mp_image = mp.Image.create_from_file(img_path)
            hand_landmarker_result = detector.detect(mp_image)

            # adds each landmark's x, y, and z to dataColumns,
            row = [None] * 106
            for hand in hand_landmarker_result.hand_landmarks: # Each hand

                for j, landmark in enumerate(hand): # Each landmark
                    base = j * 5
                    row[base] = landmark.x
                    row[base+1] = landmark.y
                    row[base+2] = landmark.z
                    row[base+3] = landmark.visibility
                    row[base+4] = landmark.presence
                    if landmark.visibility != 0.0:
                        print(landmark.visibility)
                        print(landmark.presence)
                    #print(i)

            # adds classification of image to 105th column
            row[105] = path_to_classification[training_path]
            #print(str(i) + "one hand finished")
            for j, cole in enumerate(row):
                dataColumns[j].append(cole)

            #print(dataColumns)





data = DataFrame({"Wrist X Position":dataColumns[0],
                  "Wrist Y Position":dataColumns[1],
                  "Wrist Z Position":dataColumns[2],
                  "Wrist Visibility":dataColumns[3],
                  "Wrist Presence":dataColumns[4],
                  "Thumb CMC X Position":dataColumns[5],
                  "Thumb CMC Y Position":dataColumns[6],
                  "Thumb CMC Z Position":dataColumns[7],
                  "Thumb CMC Visibility":dataColumns[8],
                  "Thumb CMC Presence":dataColumns[9],
                  "Thumb MCP X Position":dataColumns[10],
                  "Thumb MCP Y Position":dataColumns[11],
                  "Thumb MCP Z Position":dataColumns[12],
                  "Thumb MCP Visibility":dataColumns[13],
                  "Thumb MCP Presence":dataColumns[14],
                  "Thumb IP X Position":dataColumns[15],
                  "Thumb IP Y Position":dataColumns[16],
                  "Thumb IP Z Position":dataColumns[17],
                  "Thumb IP Visibility":dataColumns[18],
                  "Thumb IP Presence":dataColumns[19],
                  "Thumb TIP X Position":dataColumns[20],
                  "Thumb TIP Y Position":dataColumns[21],
                  "Thumb TIP Z Position":dataColumns[22],
                  "Thumb TIP Visibility":dataColumns[23],
                  "Thumb TIP Presence":dataColumns[24],
                  "Index Finger MCP X Position":dataColumns[25],
                  "Index Finger MCP Y Position":dataColumns[26],
                  "Index Finger MCP Z Position":dataColumns[27],
                  "Index Finger MCP Visibility":dataColumns[28],
                  "Index Finger MCP Presence":dataColumns[29],
                  "Index Finger PIP X Position":dataColumns[30],
                  "Index Finger PIP Y Position":dataColumns[31],
                  "Index Finger PIP Z Position":dataColumns[32],
                  "Index Finger PIP Visibility":dataColumns[33],
                  "Index Finger PIP Presence":dataColumns[34],
                  "Index Finger DIP X Position":dataColumns[35],
                  "Index Finger DIP Y Position":dataColumns[36],
                  "Index Finger DIP Z Position":dataColumns[37],
                  "Index Finger DIP Visibility":dataColumns[38],
                  "Index Finger DIP Presence":dataColumns[39],
                  "Index Finger TIP X Position":dataColumns[40],
                  "Index Finger TIP Y Position":dataColumns[41],
                  "Index Finger TIP Z Position":dataColumns[42],
                  "Index Finger TIP Visibility":dataColumns[43],
                  "Index Finger TIP Presence":dataColumns[44],
                  "Middle Finger MCP X Position":dataColumns[45],
                  "Middle Finger MCP Y Position":dataColumns[46],
                  "Middle Finger MCP Z Position":dataColumns[47],
                  "Middle Finger MCP Visibility":dataColumns[48],
                  "Middle Finger MCP Presence":dataColumns[49],
                  "Middle Finger PIP X Position":dataColumns[50],
                  "Middle Finger PIP Y Position":dataColumns[51],
                  "Middle Finger PIP Z Position":dataColumns[52],
                  "Middle Finger PIP Visibility":dataColumns[53],
                  "Middle Finger PIP Presence":dataColumns[54],
                  "Middle Finger DIP X Position":dataColumns[55],
                  "Middle Finger DIP Y Position":dataColumns[56],
                  "Middle Finger DIP Z Position":dataColumns[57],
                  "Middle Finger DIP Visibility":dataColumns[58],
                  "Middle Finger DIP Presence":dataColumns[59],
                  "Middle Finger TIP X Position":dataColumns[60],
                  "Middle Finger TIP Y Position":dataColumns[61],
                  "Middle Finger TIP Z Position":dataColumns[62],
                  "Middle Finger TIP Visibility":dataColumns[63],
                  "Middle Finger TIP Presence":dataColumns[64],
                  "Ring Finger MCP X Position":dataColumns[65],
                  "Ring Finger MCP Y Position":dataColumns[66],
                  "Ring Finger MCP Z Position":dataColumns[67],
                  "Ring Finger MCP Visibility":dataColumns[68],
                  "Ring Finger MCP Presence":dataColumns[69],
                  "Ring Finger PIP X Position":dataColumns[70],
                  "Ring Finger PIP Y Position":dataColumns[71],
                  "Ring Finger PIP Z Position":dataColumns[72],
                  "Ring Finger PIP Visibility":dataColumns[73],
                  "Ring Finger PIP Presence":dataColumns[74],
                  "Ring Finger DIP X Position":dataColumns[75],
                  "Ring Finger DIP Y Position":dataColumns[76],
                  "Ring Finger DIP Z Position":dataColumns[77],
                  "Ring Finger DIP Visibility":dataColumns[78],
                  "Ring Finger DIP Presence":dataColumns[79],
                  "Ring Finger TIP X Position":dataColumns[80],
                  "Ring Finger TIP Y Position":dataColumns[81],
                  "Ring Finger TIP Z Position":dataColumns[82],
                  "Ring Finger TIP Visibility":dataColumns[83],
                  "Ring Finger TIP Presence":dataColumns[84],
                  "Pinky MCP X Position":dataColumns[85],
                  "Pinky MCP Y Position":dataColumns[86],
                  "Pinky MCP Z Position":dataColumns[87],
                  "Pinky MCP Visibility":dataColumns[88],
                  "Pinky MCP Presence":dataColumns[89],
                  "Pinky PIP X Position":dataColumns[90],
                  "Pinky PIP Y Position":dataColumns[91],
                  "Pinky PIP Z Position":dataColumns[92],
                  "Pinky PIP Visibility":dataColumns[93],
                  "Pinky PIP Presence":dataColumns[94],
                  "Pinky DIP X Position":dataColumns[95],
                  "Pinky DIP Y Position":dataColumns[96],
                  "Pinky DIP Z Position":dataColumns[97],
                  "Pinky DIP Visibility":dataColumns[98],
                  "Pinky DIP Presence":dataColumns[99],
                  "Pinky TIP X Position":dataColumns[100],
                  "Pinky TIP Y Position":dataColumns[101],
                  "Pinky TIP Z Position":dataColumns[102],
                  "Pinky TIP Visibility":dataColumns[103],
                  "Pinky TIP Presence":dataColumns[104],
                  "Classification":dataColumns[105]
                  })

data.head(8)

data.to_csv("MediapipePredictions.csv", index=False)
