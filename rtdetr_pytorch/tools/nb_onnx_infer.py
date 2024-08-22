#%%
import cv2
import numpy as np

import torch
import onnxruntime as ort
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
#%%
# Define consts
MODEL_PATH = 'models/rt-detr-20240820.onnx'
OUTPUT_TMP_FOLDER = '/mnt/ssd/tmp/rt-detr/out/'
SCORE_THRESHOLD = 0.5
#%%
# Define file paths to be processed
video_path = "/mnt/ssd/vis_tool/data/raw-videos/nz.fol_id.as-7776-0152.a.a.spring_river-2024_07-31_08_00_01.mp4"
img_path = "/home/irisdc01/training/dataset/detection-images/nz.fol_id.as-7775-0038.a/nz.fol_id.as-7775-0038.a.a.katoa-2022_11-15_05_00_14_frame_10879.png"
# %%
# Define functions
#%%
## Inference function
def infer(session, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    size = torch.tensor([[640, 640]])
    outputs = session.run(None, {'images': image, "orig_target_sizes": size.data.numpy()})
    return outputs

#%%
## Postprocess function
def postprocess(outputs, score_threshold):
    labels = outputs[0]
    boxes = outputs[1]
    scores = outputs[2]
    # print(outputs)
    threshold = score_threshold
    filtered_boxes = boxes[scores > threshold]
    filtered_labels = labels[scores > threshold]
    filtered_scores = scores[scores > threshold]
    return filtered_boxes, filtered_labels, filtered_scores

#%%
## Simple version of a tracker class
class SimpleTracker:
    def __init__(self):
        self.tracks = []

    def update(self, boxes, labels, scores):
        new_tracks = []
        for box, label, score in zip(boxes, labels, scores):
            track = {'box': box, 'label': label, 'score': score}
            new_tracks.append(track)
        self.tracks = new_tracks


#%%
## Function to draw boxes on an image
def drawBoxInImage(img_path, model_path=MODEL_PATH, ouput_path=OUTPUT_TMP_FOLDER+"output.jpg", score_threshold=SCORE_THRESHOLD):
    frame = cv2.imread(img_path)
    # inference
    session = ort.InferenceSession(model_path)
    outputs = infer(session, frame)
    r_width = frame.shape[1]
    r_high = frame.shape[0]
    # post processing
    boxes, labels, scores = postprocess(outputs, score_threshold)
    # update track
    tracker = SimpleTracker()
    tracker.update(boxes, labels, scores)
    # output
    for track in tracker.tracks:
        # x1:wid  y1:high
        x1, y1, x2, y2 = map(int, track['box'])
        x1 = int(x1 * (r_width / 640))
        y1 = int(y1 * (r_high / 640))
        x2 = int(x2 * (r_width / 640))
        y2 = int(y2 * (r_high / 640))
        print("=========", x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            str(track['label']) + " : " + str('{:.2f}'.format(track['score'])),
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    cv2.imwrite(ouput_path, frame)

#%%
## Function to draw boxes on video
def drawBoxInVideo(video_path, model_path=MODEL_PATH, ouput_path=OUTPUT_TMP_FOLDER+"output.mp4", score_threshold=SCORE_THRESHOLD):
    session = ort.InferenceSession(model_path)

    tracker = SimpleTracker()

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # inference
        outputs = infer(session, frame)
        r_width = frame.shape[1]
        r_high = frame.shape[0]
        # post processing
        boxes, labels, scores = postprocess(outputs, score_threshold)
        # update track
        tracker.update(boxes, labels, scores)
        #print(tracker.tracks)
        # config output vidoe
        for track in tracker.tracks:
            # print(track['box'])
            # x1:wid  y1:high
            x1, y1, x2, y2 = map(int, track['box'])
            x1 = int(x1 * (r_width / 640))
            y1 = int(y1 * (r_high / 640))
            x2 = int(x2 * (r_width / 640))
            y2 = int(y2 * (r_high / 640))
            #print("=========", x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                str(track['label']) + " : " + str('{:.2f}'.format(track['score'])),
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    cap.release()
# %%
# draw bounding boxes on the sample image
drawBoxInImage(img_path)
# %%
