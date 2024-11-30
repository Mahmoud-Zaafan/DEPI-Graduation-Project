from ultralytics import YOLO
import numpy as np
import scipy
import scipy.optimize
import torch
import cv2

model = YOLO("yolov8m.pt")
yolo_classes = model.names


def yolo_predict(imgs, YOLO_model):
    # Load YOLO model
    model = YOLO_model

    det = []
    lbls = []
    mask = []
    plotting = []

    for img in imgs:
        # Predict using the model
        results = model.predict(img)
        result = results[0]
        plot = result.plot()

        # Get bounding boxes, class labels, confidences, and masks
        detections = [box.xyxy[0].tolist() for box in result.boxes]
        labels = [result.names[box.cls[0].item()] for box in result.boxes]

        det.append(detections)
        lbls.append(labels)
        plotting.append(plot)

    return det, lbls, plotting


# det is the bounding boxes, lbls is the class labels for each detection and plotting is the left and right images ready to be shown
# get centr, top left and bottom right of boxes


def tlbr_to_center1(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (tlx + brx) / 2
        cy = (tly + bry) / 2
        points.append([cx, cy])
    return points


def tlbr_to_corner(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (tlx + tlx) / 2
        cy = (tly + tly) / 2
        points.append((cx, cy))
    return points


def tlbr_to_corner_br(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (brx + brx) / 2
        cy = (bry + bry) / 2
        points.append((cx, cy))
    return points


def tlbr_to_area(boxes):
    areas = []
    for tlx, tly, brx, bry in boxes:
        cx = brx - tlx
        cy = bry - tly
        areas.append(abs(cx * cy))
    return areas


# get all distances from every object box to every other object box
# left image is boxes[0]
# right image is boxes[1]

# do broad casting.
# in python, col vector - row vector gives matrix:
# [a] - [c,d] = [a-c, a-d]
# [b]           [b-c, b-d]


def get_horiz_dist_centre(boxes):
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_horiz_dist_corner_tl(boxes):
    pnts1 = np.array(tlbr_to_corner(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_corner(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_horiz_dist_corner_br(boxes):
    pnts1 = np.array(tlbr_to_corner_br(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_corner_br(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_vertic_dist_centre(boxes):
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:, 1]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:, 1]
    return pnts1[:, None] - pnts2[None]


def get_area_diffs(boxes):
    pnts1 = np.array(tlbr_to_area(boxes[0]))
    pnts2 = np.array(tlbr_to_area(boxes[1]))
    return abs(pnts1[:, None] - pnts2[None])


def get_dist_to_centre_tl(box, img):
    sz1 = img.shape[1]
    center = sz1 / 2
    pnts = np.array(tlbr_to_corner(box))[:, 0]
    return abs(pnts - center)


def get_dist_to_centre_br(box, img):
    sz1 = img.shape[1]
    center = sz1 / 2
    pnts = np.array(tlbr_to_corner_br(box))[:, 0]
    return abs(pnts - center)


# create the tracking cost function.
# consists of theree parts.
#  1. The vertical move up and down of object centre of mass. Scale this up because we do not expect this to be very much.
#  2. The move left or right by the object. We only expect it to move right (from the left eye image). So penalise if it moves left.
#  3. The difference in area of pixels. Area of image is width x height, so divide by height, there for this will have max value of width


def get_cost(boxes, img, lbls=None):

    sz1 = img.shape[1]

    alpha = sz1
    beta = 10
    gamma = 5

    # vertical_dist, scale by gamma since can't move up or down
    vert_dist = gamma * abs(get_vertic_dist_centre(boxes))

    # horizonatl distance.
    horiz_dist = get_horiz_dist_centre(boxes)

    # increase cost if object has moved from right to left.
    horiz_dist[horiz_dist < 0] = beta * abs(horiz_dist[horiz_dist < 0])

    # area of box
    area_diffs = get_area_diffs(boxes) / alpha

    cost = np.array([vert_dist, horiz_dist, area_diffs])

    cost = cost.sum(axis=0)

    # add penalty term for different object classes
    if lbls is not None:
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                if lbls[0][i] != lbls[1][j]:
                    cost[i, j] += 150
    return cost


def get_horiz_dist(masks, prob_thresh=0.7):
    # gets the horizontal distance between the centre of mass for each object
    # left masks
    mask_bool = masks[0] > prob_thresh
    mask_bool = mask_bool.squeeze(1)
    # right masks
    mask_bool2 = masks[1] > prob_thresh
    mask_bool2 = mask_bool2.squeeze(1)

    # left params
    # com1 is center of mass of height
    # com2 is center of mass of width
    mask_size = (mask_bool).sum(dim=[1, 2])
    mask_com_matrix_1 = torch.tensor(range(mask_bool.shape[1]))
    com1 = ((mask_com_matrix_1.unsqueeze(1)) * mask_bool).sum(dim=[1, 2]) / mask_size
    mask_com_matrix_2 = torch.tensor(range(mask_bool.shape[2]))
    com2 = ((mask_com_matrix_2.unsqueeze(0)) * mask_bool).sum(dim=[1, 2]) / mask_size

    left_params = torch.stack((com1, com2, mask_size)).transpose(1, 0)

    # get right params
    mask_size2 = (mask_bool2).sum(dim=[1, 2])
    mask_com_matrix_12 = torch.tensor(range(mask_bool2.shape[1]))
    com12 = ((mask_com_matrix_12.unsqueeze(1)) * mask_bool2).sum(
        dim=[1, 2]
    ) / mask_size2
    mask_com_matrix_22 = torch.tensor(range(mask_bool2.shape[2]))
    com22 = ((mask_com_matrix_22.unsqueeze(0)) * mask_bool2).sum(
        dim=[1, 2]
    ) / mask_size2

    right_params = torch.stack((com12, com22, mask_size2)).transpose(1, 0)

    # calculate cost function
    cost = left_params[:, None] - right_params[None]
    return cost[:, :, 1]


def get_tracks(cost):
    return scipy.optimize.linear_sum_assignment(cost)


def get_object_dist(object_name, final_dists_list, tantheta, fl, sz1):
    distance = None
    try:
        for dist, label in final_dists_list:
            if label == object_name:
                distance = dist
                break
        else:
            raise ValueError(f"No object found with label: {object_name}")

        x = (7.05 / 2) * sz1 * (1 / tantheta) / distance + fl
        return round(x, ndigits=1)
    except ValueError as e:
        print(e)
        return None


def recognise_distance(left, right, object_name):
    Left_img = cv2.imread(left)
    Right_img = cv2.imread(right)
    imgs = [Left_img, Right_img]
    det, lbls, plotting = yolo_predict(imgs, model)
    sz1 = Right_img.shape[1]
    sz2 = Right_img.shape[0]
    centre = sz1 / 2
    tmp1 = get_dist_to_centre_br(det[0], Right_img)
    tmp2 = get_dist_to_centre_br(det[1], Right_img)
    cost = get_cost(det, Right_img, lbls=lbls)
    tracks = get_tracks(cost)
    h_d = [[lbls[0][i], lbls[1][j]] for i, j in zip(*tracks)]
    dists_tl = get_horiz_dist_corner_tl(det)
    dists_br = get_horiz_dist_corner_br(det)

    final_dists = []
    dctl = get_dist_to_centre_tl(det[0], Left_img)
    dcbr = get_dist_to_centre_br(det[0], Left_img)

    for i, j in zip(*tracks):
        if dctl[i] < dcbr[i]:
            final_dists.append((dists_tl[i][j], lbls[0][i]))

        else:
            final_dists.append((dists_br[i][j], lbls[0][i]))
    fl = 30 - 37.9 * 50 / 68.2459
    tantheta = (1 / (50 - fl)) * (7.05 / 2) * sz1 / 37.9
    fd = [i for (i, j) in final_dists]
    distance = get_object_dist(
        final_dists_list=final_dists,
        object_name=object_name,
        fl=fl,
        tantheta=tantheta,
        sz1=sz1,
    )

    return distance


hh = recognise_distance(
    "/content/drive/MyDrive/distance/left_eye_50cm.jpg",
    "/content/drive/MyDrive/distance/right_eye_50cm.jpg",
    "bottle",
)


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download stopwords and punkt if not already downloaded
nltk.download("stopwords")
nltk.download("punkt")

similar_words = {
    "person": ["human", "individual", "man", "woman", "person", "kid", "child"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "car": ["car", "automobile", "vehicle", "auto", "carriage"],
    "motorcycle": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "plane"],
    "bus": ["bus", "coach"],
    "train": ["train", "railway", "railroad"],
    "truck": ["truck", "lorry"],
    "boat": ["boat", "ship", "vessel"],
    "traffic light": ["traffic light", "signal", "stoplight"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign", "stop"],
    "parking meter": ["parking meter", "meter"],
    "bench": ["bench", "seat"],
    "bird": ["bird", "avian"],
    "cat": ["cat", "feline"],
    "dog": ["dog", "canine"],
    "horse": ["horse", "equine"],
    "sheep": ["sheep", "ovine"],
    "cow": ["cow", "bovine"],
    "elephant": ["elephant"],
    "bear": ["bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "rucksack"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag", "purse"],
    "tie": ["tie"],
    "suitcase": ["suitcase", "luggage"],
    "frisbee": ["frisbee"],
    "skis": ["skis"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "ball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove"],
    "skateboard": ["skateboard", "skate"],
    "surfboard": ["surfboard", "surf"],
    "tennis racket": ["tennis racket", "racket"],
    "bottle": ["bottle", "flask"],
    "wine glass": ["wine glass", "glass"],
    "cup": ["cup", "mug"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "bowl": ["bowl", "dish"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich", "burger"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake"],
    "chair": ["chair", "seat"],
    "couch": ["couch", "sofa"],
    "potted plant": ["potted plant", "pot plant", "houseplant"],
    "bed": ["bed"],
    "dining table": ["dining table", "table"],
    "toilet": ["toilet"],
    "tv": ["tv", "television"],
    "laptop": ["laptop", "notebook"],
    "mouse": ["mouse"],
    "remote": ["remote", "controller"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "mobile", "smartphone"],
    "microwave": ["microwave"],
    "oven": ["oven"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase", "pot"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddy"],
    "hair drier": ["hair drier", "hairdryer"],
    "toothbrush": ["toothbrush", "brush"],
}


def preprocess_question(question):
    # Lowercase the text
    question = question.lower()

    # Tokenize the text
    tokens = word_tokenize(question)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    tokens = [
        word
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    return tokens


def find_coco_category_from_tokens(tokens, similar_words_dict):
    for category, similar_words_list in similar_words_dict.items():
        for word in tokens:
            if word in similar_words_list or any(
                word in phrase.split() for phrase in similar_words_list
            ):
                return category
    return None


def coco_category(question, coco_dict):
    processed_question = preprocess_question(question)
    category = find_coco_category_from_tokens(processed_question, coco_dict)
    return category


test_question = "Where is the cell phone positioned?"
result = coco_category(coco_dict=similar_words, question=test_question)
print(result)
