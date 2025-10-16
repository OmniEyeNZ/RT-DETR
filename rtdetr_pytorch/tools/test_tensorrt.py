#%%
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
 
import cv2
import math
import random
import json

SCORE_THRESHOLD = 0.5

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()

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


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def img_preprocess(image):
    image_h, image_w = image.shape[:2]
    ratio_h = 640 / image_h
    ratio_w = 640 / image_w

    img = cv2.resize(image, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=2)
    img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img[np.newaxis], dtype=np.int32)
    return img

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

def group_list(lst, group_size=4):
    return [lst[i:i + group_size].tolist() for i in range(0, len(lst), group_size)]


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

img_path = 'nz.fol_id.va-3971-0001.a.a.VarneyFamilyFarm-2025_08-02_05_00_00_1352.jpg'
# load image
image = cv2.imread(img_path)
# load image shape
img_high = image.shape[0]
img_width = image.shape[1]
# iamge preprocess
preprocess_image = img_preprocess(image)
# load tensorrt model
engine = load_engine('models/2025-08-11.varney.r18vd.64.trt.engine')
# inference
outputs = tensorrt_inference(engine, preprocess_image)
