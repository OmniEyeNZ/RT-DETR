#%%
import cv2
from ultralytics import RTDETR

#%%
# Consts
OUTPUT_TMP_FOLDER = "/tmp/rt-detr/out/"

#%%
model = RTDETR("models/rtdetr-l.pt")

#%%
video_path = "/home/binwang/FarmLive_models/yolo/data/raw-videos/nz.fol_id.as-7776-0152.a.a.spring_river-2024_05-18_07_00_05.mp4"
results = model(video_path, show=False, stream=True, save=True, save_dir=OUTPUT_TMP_FOLDER)

#%%
# write the results into a video
MAX_NUM_FRAMES = 1000
video_writer = cv2.VideoWriter(
    OUTPUT_TMP_FOLDER + 'result.mp4', 
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    25, 
    (1920, 1080)
)
frame_num = 0
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # result.save(filename=OUTPUT_TMP_FOLDER + "result.jpg")  # save image to disk
    annotated_img = result.plot()
    frame_num += 1
    if frame_num < MAX_NUM_FRAMES: 
        video_writer.write(annotated_img)
    else:
        break
    
video_writer.release()

#%%
print('-------------- finished -------------')
# %%
