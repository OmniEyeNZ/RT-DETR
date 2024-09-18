#%%
import json
import torch
import numpy as np
from ultralytics.utils.metrics import ConfusionMatrix, compute_ap, plot_pr_curve

#%%
# Define consts
GROUD_TRUTH_JSON_PATH =  "/yolo/yolo/dataset/label-test-subset.json"
PREDICT_JSON_DIR = "/yolo/yolo/out/"
#%%
# Define file paths to be processed
truth_json_path = GROUD_TRUTH_JSON_PATH
r18vd_predict_json_path = PREDICT_JSON_DIR + "r18vd-predict-output.json"
r50vd_predict_json_path = PREDICT_JSON_DIR + "r50vd-predict-output.json"
#%%
# load json file of predict and groundtruth 
def load_json(json_path:str):
    with open(json_path, 'r')as f:
        json_data = json.load(f)
    return json_data

#%%
# Generate a list of prediction boxes and do coordinate transformations
def generate_pre_list(predict_img_value, cls:int):
    ret_pre_list = []
    for piv in predict_img_value:
        piv['bbox'][2] = piv['bbox'][0] + piv['bbox'][2]
        piv['bbox'][3] = piv['bbox'][1] + piv['bbox'][3]
        ret_pre_list.append([*piv['bbox'], piv['confidence'], cls])
    return ret_pre_list
#%%
# Generate a list of groundtruth boxes and do coordinate transformations
def generate_gt_list(gt_list):
    ret_gt_list = []
    for gt_bbox in gt_list:
        gt_bbox[2] = gt_bbox[0] + gt_bbox[2]
        gt_bbox[3] = gt_bbox[1] + gt_bbox[3]
        ret_gt_list.append(gt_bbox)
    return ret_gt_list
#%%
# evaluate
def evaluate(predict_json_path, 
             ground_truth_json_path = GROUD_TRUTH_JSON_PATH,
             nc = 1,
             conf = 0.5,
             iou_thres = 0.5
             ):
    #load json
    predict_json = load_json(predict_json_path)
    ground_truth_json = load_json(ground_truth_json_path)
    ground_truth_json = ground_truth_json['annotations']
    # Create a class and specify nc(classes number), thresholds for conf and iou
    cm = ConfusionMatrix(nc, conf, iou_thres)
    #Traversal inference generated JSON
    FN_count = 0 # FN_count: The number of images that were not detected
    for img_id, img_value in predict_json.items():
        # Find all the box information for a picture from ground_truth_json
        gt_list = []
        for gtj in ground_truth_json:
            if gtj['image_id'] == int(img_id):
                gt_list.append(gtj['bbox'])
        # Generate box lists and transform coordinates
        pre_list = generate_pre_list(img_value, 0)
        gt_list = generate_gt_list(gt_list)
        # Generate a category for each attribute in gt_list(groundtruth list), corresponding to the index.
        gt_cls = [0 for _ in range(len(gt_list))]
        if len(pre_list) == 0:
            FN_count = FN_count + len(gt_list)
            continue
        pre_list = torch.tensor(pre_list)
        gt_list = torch.tensor(gt_list)
        gt_cls = torch.tensor(gt_cls)
        # evaluate
        cm.process_batch(pre_list, gt_list, gt_cls)

    # output Confusion Matrix
    #print(cm.matrix)
    # calculate
    TP = cm.matrix[0][0]
    FP = cm.matrix[0][1]
    FN = cm.matrix[1][0] + FN_count
    TN = cm.matrix[1][1]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print("FN_count =",FN_count)
    # print("precision = ", precision)
    # print("recall = ", recall)
    #cm.plot() # draw Confusion Matrix

    return precision, recall
# %%
# calculate ap
def calculate_ap(predict_json, truth_json, draw_prec_recall = False):
    # calculate ap[0.75]
    prec_50, recall_50 = evaluate(predict_json,truth_json,iou_thres=0.5)
    ap_50, _, _ = compute_ap([recall_50], [prec_50])
    # calculate ap[0.50]
    prec_75, recall_75 = evaluate(predict_json,truth_json,iou_thres=0.75)
    ap_75, _, _ = compute_ap([recall_75], [prec_75])
    
    # calculate ap[0.50:0.95]
    prec_list = []
    recall_list = []
    iou = 0.5
    while iou < 1:
        prec, recall = evaluate(predict_json,truth_json,iou_thres=iou)
        prec_list.append(prec)
        recall_list.append(recall)
        iou = round(iou + 0.05, 2)
    ap, _, _ = compute_ap(recall_list, prec_list)    

    # draw precision-recall curve
    if draw_prec_recall:
        px = np.array(recall_list)  # recall values
        py = np.array([prec_list])  # precision values
        ap = np.array([[ap]])  # average precision value
        names = {0: "dairy_cow"}
        plot_pr_curve(px, py, ap, names=names)
    
    print(f"AP={ap}; AP50={ap_50}; AP75={ap_75};")
    return [ap, ap_50, ap_75]

#%%
r18vd_ap = calculate_ap(r18vd_predict_json_path, truth_json_path)
r50vd_ap = calculate_ap(r50vd_predict_json_path, truth_json_path)

# %%
print(f"r18vd:\n\tap = {r18vd_ap[0]};\n\tap50 = {r18vd_ap[1]};\n\tap75 = {r18vd_ap[2]}")
print(f"r50vd:\n\tap = {r50vd_ap[0]};\n\tap50 = {r50vd_ap[1]};\n\tap75 = {r50vd_ap[2]}")