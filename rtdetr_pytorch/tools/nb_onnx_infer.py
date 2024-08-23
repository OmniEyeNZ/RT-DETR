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
## Output the box information through init the picture and run model
def model_infer(session, image):
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
# Record boxes that match the scores in one picture
class BoxSet:
    def __init__(self):
        self.boxes = []

    def update(self, boxes, labels, scores, actual_high=1080, actual_width=1920):
        new_boxes = []
        for box, label, score in zip(boxes, labels, scores):
            box = box.astype(int)
            box[0] = int(box[0] * (actual_width / 640))
            box[1] = int(box[1] * (actual_high / 640))
            box[2] = int(box[2] * (actual_width / 640))
            box[3] = int(box[3] * (actual_high / 640))
            #print(type(box), ":", box)
            box_info = {'box': box, 'label': label, 'score': score}
            new_boxes.append(box_info)
        self.boxes = new_boxes

    def clear(self):
        self.boxes = []

#%%
# Inference the picture and return all boxes
def inference(session, frame, box_set: BoxSet, score_threshold=SCORE_THRESHOLD) -> BoxSet:
    outputs = model_infer(session, frame)
    actual_high = frame.shape[0]
    actual_width = frame.shape[1]

    # post processing
    boxes, labels, scores = postprocess(outputs, score_threshold)
    # update box_set
    box_set.update(boxes, labels, scores, actual_high, actual_width)

    return box_set
#%%
## Function to draw boxes on an image
def drawBoxInImage(img_path, model_path=MODEL_PATH, ouput_path=OUTPUT_TMP_FOLDER+"output.jpg"):
    session = ort.InferenceSession(MODEL_PATH)
    frame = cv2.imread(img_path)
    box_set = BoxSet()
    box_set_out = inference(session, frame, box_set)  # get all boxes
    # output
    for box_info in box_set_out.boxes:
        # x1:wid  y1:high
        x1, y1, x2, y2 = map(int, box_info['box'])
        #print("=========", x1, y1, x2, y2, "-----", track['score'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            str('{:.2f}'.format(box_info['score'])),
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3
        )
    cv2.imwrite(ouput_path, frame)
#%%
## Function to draw boxes on video
def drawBoxInVideo(video_path, model_path=MODEL_PATH, ouput_path=OUTPUT_TMP_FOLDER+"output.mp4"):
    session = ort.InferenceSession(model_path)
    box_set = BoxSet()
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"video_path:{video_path} {fps} {video_height} {video_width}")

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size = (video_width, video_height)
    video = cv2.VideoWriter(ouput_path, fourcc, fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # inference
        box_set_out = inference(session, frame, box_set)  # get all boxes
        for box_info in box_set_out.boxes:
            x1, y1, x2, y2 = map(int, box_info['box'])
            #print("=========", x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                str('{:.2f}'.format(box_info['score'])),
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3
            )
        video.write(frame)
        # cv2.imshow('Tracking', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    video.release()

# %%
# draw bounding boxes on the sample image
drawBoxInImage(img_path)
# %%
# draw bounding boxes on the sample video
drawBoxInVideo(video_path)
# %%
