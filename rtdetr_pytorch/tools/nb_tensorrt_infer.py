#%%
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
 
import cv2
import math
import random
import json
#%%
IMG_PATH = "/home/irisdc01/training/dataset/detection-images/nz.fol_id.as-7775-0038.a/nz.fol_id.as-7775-0038.a.a.katoa-2022_11-15_05_00_14_frame_10879.png"
TRT_MODEL_PATH = "models/2024-09-13.r18vd.trt.engine"
OUT_PATH = "/mnt/ssd/tmp/rt-detr/out/"
SCORE_THRESHOLD = 0.5
#%%
img_folder = "/home/irisdc01/training/dataset/"
img_json_path = "/home/irisdc01/training/dataset/label-validation.json"
video_path1="/home/irisdc01/training/dataset/cvat_video/nz.fol_id.hu-7385-0262.a/nz.fol_id.hu-7385-0262.a.a.peaks-paradise-2024_08-19_06_00_01.mp4"
video_path2="/home/irisdc01/training/dataset/cvat_video/nz.fol_id.wm-7979-0063.a/nz.fol_id.wm-7979-0063.a.a.Tawai-2024_08-19_08_30_03.mp4"
output1 = OUT_PATH + "output/" + "nz.fol_id.hu-7385-0262.a/nz.fol_id.hu-7385-0262.a.a.peaks-paradise-2024_08-19_06_00_01.mp4"
output2 = OUT_PATH + "output/" + "nz.fol_id.wm-7979-0063.a/nz.fol_id.wm-7979-0063.a.a.Tawai-2024_08-19_08_30_03.mp4"
#%%
# Storage of the mapping of host memory and GPU memory
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()
#%%
# Request memory based on the inputs and outputs size of the model
def alloc_buf_N(engine,image):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in engine:
        size = abs(trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size)
        #print("----------",engine.max_batch_size, binding, size, engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
#%%
# Perform inferences on the input data
def do_inference_v2(context, inputs, bindings, outputs, stream, image):
    """
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    context.set_binding_shape(0, image.shape)
    # Run inference.
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
 
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
 
    # Writes the contents of the system buffers back to disk to ensure data synchronization.
    stream.synchronize()
 
    # Return only the host outputs.
    return [out.host for out in outputs]

#%%
# Load tensorrt model
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

#%%
# Img preprocess
def img_preprocess(image):
    image_h, image_w = image.shape[:2]
    ratio_h = 640 / image_h
    ratio_w = 640 / image_w

    img = cv2.resize(image, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=2)
    img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img[np.newaxis], dtype=np.int32)
    return img

#%%
# Perform inferences
def tensorrt_inference(engine, image):
    context = engine.create_execution_context()
    inputs_alloc_buf, outputs_alloc_buf, bindings_alloc_buf, stream_alloc_buf = alloc_buf_N(engine,image)
    # print("-------intputs array---------")
    # print(inputs_alloc_buf)
    # print("-------ontputs array---------")
    # print(outputs_alloc_buf)
    # while True:
    #     pass
    # Load the image into memory
    inputs_alloc_buf[0].host = image.reshape(-1)
    # inference, return outpuuts
    net_outputs = do_inference_v2(context, inputs_alloc_buf, bindings_alloc_buf, outputs_alloc_buf,stream_alloc_buf, image)
    return net_outputs

#%%
# Convert the output one-dimensional array into a bbox array in groups of 4
def group_list(lst, group_size=4):
    return [lst[i:i + group_size].tolist() for i in range(0, len(lst), group_size)]

#%%
# Filter for boxes that match the scores
def postprocess(outputs, score_threshold):
    labels = outputs[2]
    boxes = outputs[1]
    group_boxes = group_list(boxes)
    scores = outputs[0]
    filter_list = scores > score_threshold
    filtered_boxes = []
    filter_index = 0
    for filter_result in filter_list:
        if filter_result == True:
            filtered_boxes.append(group_boxes[filter_index])
            filter_index = filter_index + 1
    filtered_labels = labels
    filtered_scores = scores[filter_list]
    return filtered_boxes, filtered_labels, filtered_scores

#%%
# Store bbox information
class BoxSet:
    def __init__(self):
        self.boxes = []

    def update(self, boxes, labels, scores, actual_high=1080, actual_width=1920):
        new_boxes = []
        for box, score in zip(boxes, scores):
            box[0] = int(box[0] * actual_width)
            box[1] = int(box[1] * actual_high)
            box[2] = int(box[2] * actual_width)
            box[3] = int(box[3] * actual_high)
            #print(type(box), ":", box)
            box_info = {'box': box, 'label': 'dairy-cow', 'score': score}
            new_boxes.append(box_info)
        self.boxes = new_boxes

    def clear(self):
        self.boxes = []

#%%
# Inference about the image and draw box
def draw_box_in_image(img_path = IMG_PATH, engine_path = TRT_MODEL_PATH, out_path = OUT_PATH + "output.jpg", score_threshold = SCORE_THRESHOLD):
    # load image
    image = cv2.imread(img_path)
    # load image shape
    img_high = image.shape[0]
    img_width = image.shape[1]
    # iamge preprocess
    preprocess_image = img_preprocess(image)
    # load tensorrt model
    engine = load_engine(engine_path)
    # inference
    outputs = tensorrt_inference(engine, preprocess_image)
    # postprocess for bbox information
    boxes, labels, scores = postprocess(outputs, score_threshold)
    box_set = BoxSet()
    box_set.update(boxes, labels, scores, img_high, img_width)
    print(box_set.boxes)

    # Draw box on the image
    for box in box_set.boxes:
        # x1:wid  y1:high
        x1, y1, x2, y2 = box['box']
        #print("=========", x1, y1, x2, y2, "-----", box['score'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            image,
            str('{:.2f}'.format(box['score'])),
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3
        )
    # save image
    cv2.imwrite(out_path, image)

#%%

def draw_box_in_video(video_path, output_path, engine_path=TRT_MODEL_PATH, score_threshold = SCORE_THRESHOLD):
    # load tensorrt model
    engine = load_engine(engine_path)
    # Record all box, label and score information
    box_set = BoxSet()
    # load video
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"video_path:{video_path}, {fps}, {video_height}, {video_width}")

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size = (video_width, video_height)
    video = cv2.VideoWriter(output_path, fourcc, fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("load frame from video error")
            break
        # iamge preprocess
        preprocess_image = img_preprocess(frame)
        # inference
        outputs = tensorrt_inference(engine, preprocess_image)

        # postprocess for bbox information
        boxes, labels, scores = postprocess(outputs, score_threshold)
        box_set.update(boxes, labels, scores, video_height, video_width)
        #print(box_set.boxes)  # get all boxes

        for box_info in box_set.boxes:
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
    cap.release()
    video.release()
    print("video inference end")


#%%
# inference image to json
def infer_img_2_json(img_path, engine_path=TRT_MODEL_PATH, img_id=-1, score_threshold = SCORE_THRESHOLD)->dict:
    print(f"img_path:{img_path}")
    # load image
    image = cv2.imread(img_path)
    # load image shape
    img_high = image.shape[0]
    img_width = image.shape[1]
    # iamge preprocess
    preprocess_image = img_preprocess(image)
    # load tensorrt model
    engine = load_engine(engine_path)
    # inference
    outputs = tensorrt_inference(engine, preprocess_image)

    # postprocess for bbox information
    boxes, labels, scores = postprocess(outputs, score_threshold)
    box_set = BoxSet()
    box_set.update(boxes, labels, scores, img_high, img_width)
    #print(box_set.boxes)
    
    #output json
    json_img = {f"{img_id}": []}
    for box_info in box_set.boxes:
        # print(box_info['box']) # list
        # print(box_info['score']) # int
        box_info['box'][2] = box_info['box'][2] - box_info['box'][0]
        box_info['box'][3] = box_info['box'][3] - box_info['box'][1]
        json_box = {
            "bbox": box_info['box'],
            "confidence": float(box_info['score']),
            "image_feature": None,
            "detection_id": 0
        }
        json_img[f"{img_id}"].append(json_box)
    return json_img
#%%
# inference images from folder to json
def infer_img_folder_2_json(img_folder,img_json_path,engine_path=TRT_MODEL_PATH,ouput_path=OUT_PATH+"tensorrt_output32.json",score_threshold = SCORE_THRESHOLD):
    with open(img_json_path,'r') as f:
        val_label_json = json.load(f)
    json_imgs = {}
    for image in val_label_json['images']:
        _img_id = image['id']
        _img_path = img_folder + image['file_name']
        print(image)
        json_img = infer_img_2_json(_img_path,engine_path,_img_id,score_threshold)
        #print(json_img)
        json_imgs.update(json_img)
    with open(ouput_path, 'w') as f:
        json.dump(json_imgs, f, indent=2)
#%%
# inference on image
draw_box_in_image()
#%%
# inference on image
draw_box_in_video(video_path1, output1)
# draw_box_in_video(video_path2, output2)
#%%
# inference images from folder to json
#infer_img_folder_2_json(img_folder, img_json_path)
# %%

#%%[markdown]
## show images
# %%
# import libs
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
# %%
img = np.asarray(Image.open(OUT_PATH+"output.jpg"))
plt.imshow(img)
# %%
