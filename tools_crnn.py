import numpy as np
import json
import cv2
import os
import torch
from torch.autograd import Variable
from torchvision import transforms as transforms
import utils
import cv2
import numpy as np
import models.crnn as crnn
import math
import time
import pytesseract
from interval import Interval
import locality_aware_nms as nms_locality
import lanms
from icdar import restore_rectangle

letter = {"A":10, "B":12, "C":13, "D":14, "E":15, "F":16, "G":17, "H":18, "I":19, "J":20, "K":21, "L":23, "M":24,
          "N":25, "O":26, "P":27, "Q":28, "R":29, "S":30, "T":31, "U":32, "V":34, "W":35, "X":36, "Y":37, "Z":38}


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


class CRNN_REC(object):
    def __init__(self, model_path):
        self.model = crnn.CRNN(32, 1, 37, 256)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.converter = utils.strLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz')
        self.model.eval()

    def preprocess(self, patch):
        patch = cv2.resize(patch, (100, 32))
        if len(patch.shape) < 3:
            patch = np.expand_dims(patch, axis=-1)
        if patch.shape[-1] != 1:
            patch = np.expand_dims(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), axis=-1)
        assert len(patch.shape) == 3
        patch = transforms.ToTensor()(patch)
        patch.sub_(0.5).div_(0.5)
        if torch.cuda.is_available():
            patch = patch.cuda()
        return patch

    def recognize(self, patch):
        patch = self.preprocess(patch)
        patch = patch.view(1, *patch.size())
        patch = Variable(patch)
        preds = self.model(patch)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        # raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        # sim_pred = raw_pred.replace('-', '')
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return sim_pred.upper()


def get_images(data_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    # print('Find {} images'.format(len(files)))
    files.sort(key=lambda x: int(x[-8:-4]))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def box_area(box):
    threshold = Interval(1400, 7000)
    area = abs((box[1, 0] - box[0, 0]) * (box[2, 1] - box[1, 1]))
    return area in threshold

def filter_boxes(im, boxes, crnn, **parameters):
    boxes = boxes.astype(np.int32)
    boxes_list = []
    _ = [boxes_list.append(box) for box in boxes]
    boxes_list.sort(key=lambda x: x[0, 1])
    filtered_boxes = []

    def get_patch(im, box):
        bias = [5, 4]  ##
        box = box.astype(np.int32)
        patch = im[max(min(box[:, 1])-bias[0], 0):max(box[:, 1])+bias[0], max(min(box[:, 0])-bias[1], 0):max(box[:, 0])+bias[1]]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return patch

    owner_box = []
    potential_boxes = []
    for box in boxes_list:
        temp = get_patch(im, box)
        english = crnn.recognize(temp)
        # number = pytesseract.image_to_string(temp, config=config[1])
        if english[-1] == 'U':
            if len(english) == 4:
                owner_box.append(box)
                break
            else:
                potential_boxes.append(box)

        if len(owner_box) == 0 and len(potential_boxes)>0:
            owner_box.append(potential_boxes[0])

    assert len(owner_box) == 1
    owner_box = owner_box[0]
    bias = (owner_box[2][1] - owner_box[1][1])/1.5
    reference_point = owner_box[1]

    # shape = im.shape
    [filtered_boxes.append(box) if (box[0][0] >= reference_point[0]) and (box[0][1] in Interval(reference_point[1]-bias, reference_point[1]+bias)) else None for box in boxes]
    filtered_boxes = sorted(filtered_boxes, key=(lambda x: x[0][0] - reference_point[0]))
    if len(filtered_boxes) == 2:
        num_box = np.concatenate([filtered_boxes[0][np.newaxis, ...], filtered_boxes[1][np.newaxis, ...]], axis=0)
    else:
        num_box = filtered_boxes[0][np.newaxis, ...]

    # filtered_boxes = sorted(filtered_boxes, key=(lambda x: x[0][0]))

    return np.concatenate([owner_box[np.newaxis, ...], num_box], axis=0)

def get_patchs(im, boxes, with_morph=False, **parameters):
    patches = []
    cross_threshold = parameters['parameters']['get_patchs']['cross_threshold']  ##
    bias = parameters['parameters']['get_patchs']['bias']  ##
    for box in boxes:
        box = box.astype(np.int32)
        patch = im[min(box[:, 1])-bias[0]:max(box[:, 1])+bias[3], min(box[:, 0])-bias[1]:max(box[:, 0])+bias[2]]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if with_morph:
            if patch.shape[0] > cross_threshold:
                patch = cv2.morphologyEx(patch, cv2.MORPH_CROSS, np.ones((3, 1), np.uint8))
            patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        patches.append(patch)
        # cv2.imshow('patch', patch)
        # cv2.waitKey(0)
    return patches

def patches_filter(patches):
    shape_threshold = np.zeros(2)
    for patch in patches:
        if patch is not None:
            shape_threshold = shape_threshold + np.array(patch.shape)
    shape_threshold /= (len(patches)*1)

    patches_filtered = []
    index = []
    for i in range(len(patches)):
        if patches[i] is not None:
            if (np.array(patches[i].shape) >= shape_threshold).all():
                patches_filtered.append(patches[i])
                index.append(i)

    return patches_filtered, index

def boxes_to_array(boxes):
    numbers = []
    temp = []
    for i in boxes:
        if i in [' ', '\n']:
            try:
                numbers.append(int(''.join(temp)))
            except:
                numbers.append(''.join(temp))
            temp = []
        else:
            temp.append(i)
    numbers.append(int(''.join(temp)))
    numbers = numbers[6:] if numbers[0] == '~' else numbers
    return np.array(numbers).reshape([-1, 6])

def expand_box_width(box):
    width = box[1, 0] - box[0, 0]
    box[1, 0] += np.round(width * 0.2)
    box[2, 0] += np.round(width * 0.2)
    return box

def get_check_patch(im, box, config, text_len=6, flag=False, **parameters):
    if not flag:
        origin = box[1].astype(np.int32)
        # width = 65
        bias = parameters['parameters']['get_check_patch']['bias']
        width_ratio = parameters['parameters']['get_check_patch']['width_ratio']
        origin[0] += bias  ### attention
        height = np.round((box[2, 1] - box[1, 1])).astype(np.int32)
        width = (height*width_ratio).astype(np.int32)  # width_ratio:1.5
        check_patch = im[origin[1]-(height*0.1).astype(np.int32):origin[1]+(height*1.25).astype(np.int32), origin[0]:origin[0]+width]
        point = (origin[0], origin[1]-(height*0.1).astype(np.int32))
    else:
        try:
            patch = get_patchs(im, box[np.newaxis, ...], with_morph=True, **parameters)[0]
            check_boxes = pytesseract.image_to_boxes(patch, config=config[1])
            if '~' in check_boxes:
                patch = get_patchs(im, box[np.newaxis, ...], with_morph=False)[0]
                check_boxes = pytesseract.image_to_boxes(patch, config=config[1])
                if '~' in check_boxes:
                    check_boxes = check_boxes.replace('~', '0')
            check_array = boxes_to_array(check_boxes)[:text_len, :].astype(np.int32)
        except:
            patch = get_patchs(im, box[np.newaxis, ...], with_morph=False, **parameters)[0]
            check_boxes = pytesseract.image_to_boxes(patch, config=config[1])
            check_array = boxes_to_array(check_boxes)[:text_len, :].astype(np.int32)

        origin = (check_array[-1, 3] + box[0][0], check_array[-1, 2] + box[0][1])
        # origin = (check_array[5, 3] + box[0][0], check_array[5, 2] + box[0][1])
        height = (box[2, 1] - origin[1]).astype(np.int32)
        width = (box[2, 0] - origin[0] + 10).astype(np.int32)
        check_patch = im[origin[1]-(height*0.1).astype(np.int32):origin[1]+(height*1.25).astype(np.int32), origin[0]:origin[0]+width]
        point = (origin[0], origin[1]-(height*0.1).astype(np.int32))
    return check_patch, point

def locate_check_num(check_patch, origin_point, **parameters):
    bias = parameters['parameters']['locate_check_num']['bias']
    gray = cv2.cvtColor(check_patch, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if np.mean(thresh) < parameters['parameters']['locate_check_num']['bitwise_thresh']:
        thresh = cv2.bitwise_not(thresh)

    thresh = cv2.bitwise_not(thresh)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = parameters['parameters']['locate_check_num']['min_size']

    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size and stats[i+1, 3]!=check_patch.shape[0]:
            img2[output == i + 1] = 255
    position = np.array(np.where(img2)).T

    # if position.shape[0] == 0:
    #     print('redetect position')
    #     img2 = np.zeros((output.shape))
    #     for i in range(0, nb_components):
    #         if sizes[i] >= min_size:
    #             img2[output == i + 1] = 255
    #     position = np.array(np.where(img2)).T

    if position.shape[0] == 0:
        center = (gray.shape[1]/2, gray.shape[0]/2)
        tl = (np.round(center[0]*0.3).astype(np.int16), np.round(center[1]*0.3).astype(np.int16))
        br = (np.round(center[0]*1.8).astype(np.int16), np.round(center[1]*1.8).astype(np.int16))
    else:
        tl = (min(position[:, 1])-bias if min(position[:, 1]) >= bias else 0, min(position[:, 0])-bias if min(position[:, 0]) >= bias else 0)
        br = (max(position[:, 1])+bias, max(position[:, 0])+bias)
        if br[0] - tl[0] < 10:
            br = (br[0]+5, br[1])
            tl = (max(tl[0]-5, 0), tl[1])


    patch_go = check_patch[tl[1]:br[1], tl[0]:br[0]]
    return (tl[0]+origin_point[0], tl[1]+origin_point[1]), (br[0]+origin_point[0], br[1]+origin_point[1]), patch_go


def detect_check_num(patch, model):
    patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
    # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # _, patch = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    # patch = patch[..., np.newaxis]
    patch = patch[np.newaxis, ...]/255.0
    assert len(patch.shape) == 4
    num = np.round(model.predict(patch).squeeze())
    try:
        check_num = int(np.where(num==1)[0])
    except:
        check_num = 0
    return check_num


def check_container_number(number, check_num):
    number = list(number)
    check_out = 0
    for i, num in enumerate(number):
        value = letter[num] if i < 4 else int(num)
        check_out += value * 2**i

    check_out = check_out%11
    check_out = 0 if check_out==10 else check_out
    if check_out == int(check_num):
        return True
    else:
        return False

