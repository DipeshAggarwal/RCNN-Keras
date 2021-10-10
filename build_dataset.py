from core.iou import compute_iou
from core import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os

for dir_path in (config.POSITIVE_PATH, config.NEGATIVE_PATH):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

image_paths = list(paths.list_images(config.ORIG_IMAGES))

total_positive = 0
total_negative = 0

for i, image_path in enumerate(image_paths):
    print("[INFO] Processing image: {}/{}...".format(i + 1, len(image_paths)))
    
    filename = image_path.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    annot_path = os.path.sep.join([config.ORIG_ANNOTS, "{}.xml".format(filename)])
    
    contents = open(annot_path).read()
    soup = BeautifulSoup(contents, "html.parser")
    gt_boxes = []
    
    w = int(soup.find("width").string)
    h = int(soup.find("height").string)
    
    for o in soup.find_all("object"):
        label = o.find("name").string
        x_min = int(o.find("xmin").string)
        y_min = int(o.find("ymin").string)
        x_max = int(o.find("xmax").string)
        y_max = int(o.find("ymax").string)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = max(h, y_max)
        
        gt_boxes.append((x_min, y_min, x_max, y_max))
        
    image = cv2.imread(image_path)
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposed_rects = []
    
    for x, y, w, h in rects:
        proposed_rects.append((x, y, x + w, y + h))
        
    positive_rois = 0
    negative_rois = 0
    
    for proposed_rect in proposed_rects[:config.MAX_PROPOSALS]:
        prop_start_x, prop_start_y, prop_end_x, prop_end_y = proposed_rect
        
        for gt_box in gt_boxes:
            iou = compute_iou(gt_box, proposed_rect)
            gt_start_x, gt_start_y, gt_end_x, gt_end_y = gt_box
            
            roi = None
            output_path = None
            
            if iou > 0.7 and positive_rois <= config.MAX_POSITIVE:
                roi = image[prop_start_y: prop_end_y, prop_start_x: prop_end_x]
                filename = "{}.png".format(total_positive)
                output_path = os.path.sep.join([config.POSITIVE_PATH, filename])
                
                positive_rois += 1
                total_positive += 1
                
            full_overlap = prop_start_x >= gt_start_x
            full_overlap = full_overlap and prop_start_y >= gt_start_y
            full_overlap = full_overlap and prop_end_y >= gt_end_y
            full_overlap = full_overlap and prop_end_x >= gt_end_x
            
            if not full_overlap and iou < 0.05 and negative_rois <= config.MAX_NEGATIVE:
                roi = image[prop_start_y: prop_end_y, prop_start_x: prop_end_x]
                filename = "{}.png".format(total_negative)
                output_path = os.path.sep.join([config.NEGATIVE_PATH, filename])
                
                negative_rois += 1
                total_negative += 1
                
            if roi is not None and output_path is not None:
                roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(output_path, roi)
