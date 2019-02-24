import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pytesseract
import json
from keras.models import load_model
from interval import Interval

import locality_aware_nms as nms_locality
import lanms

tf.app.flags.DEFINE_string('test_data_path', './checks/backup', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './checks/maigao_2018_11_resnet/', '')
tf.app.flags.DEFINE_string('output_dir', './checks/output', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS

config = ['--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0',
          '--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789']
letter = {"A":10, "B":12, "C":13, "D":14, "E":15, "F":16, "G":17, "H":18, "I":19, "J":20, "K":21, "L":23, "M":24,
          "N":25, "O":26, "P":27, "Q":28, "R":29, "S":30, "T":31, "U":32, "V":34, "W":35, "X":36, "Y":37, "Z":38}

Debug_mode = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
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

def filter_boxes(im, boxes, config):
    boxes = boxes.astype(np.int32)
    boxes_list = []
    _ = [boxes_list.append(box) for box in boxes]
    boxes_list.sort(key=lambda x: x[0, 1])
    filtered_boxes = []
    def get_patch(im, box):
        bias = [5, 4]
        box = box.astype(np.int32)
        patch = im[min(box[:, 1])-bias[0]:max(box[:, 1])+bias[0], min(box[:, 0])-bias[1]:max(box[:, 0])+bias[1]]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return patch

    owner_box = []
    for box in boxes_list:
        temp = get_patch(im, box)
        # tic = time.time()
        english = pytesseract.image_to_string(temp, config=config[0])
        # print('LCH: the cost time is ', time.time() - tic , ' seconds')
        # number = pytesseract.image_to_string(temp, config=config[1])
        if len(english) == 4 and 'U' == english[-1]:
            owner_box.append(box)
            break
        # if len(number) == 6:
        #     temp_boxes.append(box)

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

def get_patchs(im, boxes, with_morph=False):
    patches = []
    cross_threshold = 36
    bias = [2, 3, 5, 3]
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
    shape_threshold /= (len(patches)*1.2)

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

def get_check_patch(im, box, text_len=6, flag=False):
    if not flag:
        origin = box[1].astype(np.int32)
        # width = 65
        bias = 10
        origin[0] += bias  ### attention
        height = np.round((box[2, 1] - box[1, 1])).astype(np.int32)
        width = (height*1.5).astype(np.int32)
        check_patch = im[origin[1]-(height*0.1).astype(np.int32):origin[1]+(height*1.25).astype(np.int32), origin[0]:origin[0]+width]
        point = (origin[0], origin[1]-(height*0.1).astype(np.int32))
    else:
        try:
            patch = get_patchs(im, box[np.newaxis, ...], with_morph=True)[0]
            check_boxes = pytesseract.image_to_boxes(patch, config=config[1])
            if '~' in check_boxes:
                patch = get_patchs(im, box[np.newaxis, ...], with_morph=False)[0]
                check_boxes = pytesseract.image_to_boxes(patch, config=config[1])
                if Debug_mode:
                    print('LCH: the ~ detected')
                if '~' in check_boxes:
                    check_boxes = check_boxes.replace('~', '0')
            check_array = boxes_to_array(check_boxes)[:text_len, :].astype(np.int32)
        except:
            patch = get_patchs(im, box[np.newaxis, ...], with_morph=False)[0]
            check_boxes = pytesseract.image_to_boxes(patch, config=config[1])
            check_array = boxes_to_array(check_boxes)[:text_len, :].astype(np.int32)

        origin = (check_array[-1, 3] + box[0][0], check_array[-1, 2] + box[0][1])
        # origin = (check_array[5, 3] + box[0][0], check_array[5, 2] + box[0][1])
        height = (box[2, 1] - origin[1]).astype(np.int32)
        width = (box[2, 0] - origin[0] + 10).astype(np.int32)
        check_patch = im[origin[1]-(height*0.1).astype(np.int32):origin[1]+(height*1.25).astype(np.int32), origin[0]:origin[0]+width]
        point = (origin[0], origin[1]-(height*0.1).astype(np.int32))
    return check_patch, point

def locate_check_num(check_patch, origin_point):
    bias = 1
    gray = cv2.cvtColor(check_patch, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if np.mean(thresh) < 125:
        thresh = cv2.bitwise_not(thresh)

    thresh = cv2.bitwise_not(thresh)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 40

    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size and stats[i+1, 3]!=check_patch.shape[0]:
            img2[output == i + 1] = 255
    position = np.array(np.where(img2)).T
    opening = img2
    '''
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,np.ones((3,3),np.uint8), iterations=2)
    opening = cv2.morphologyEx(opening,cv2.MORPH_OPEN,np.ones((5,5),np.uint8), iterations=2)

    opening = cv2.bitwise_not(opening)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 80

    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size and stats[i+1, 3]!=check_patch.shape[0]:
            img2[output == i + 1] = 255

    opening = img2
    position = np.array(np.where(opening)).T
    '''
    if Debug_mode:
        plt.subplot(141)
        plt.imshow(thresh)
        plt.subplot(142)
        plt.imshow(cv2.bitwise_not(thresh))
        plt.subplot(143)
        plt.imshow(opening)
        plt.subplot(144)
        plt.imshow(cv2.cvtColor(check_patch, cv2.COLOR_BGR2RGB))
        plt.show()

    tl = (min(position[:, 1])-bias if min(position[:, 1]) >= bias else 0, min(position[:, 0])-bias if min(position[:, 0]) >= bias else 0)
    br = (max(position[:, 1])+bias, max(position[:, 0])+bias)
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
    check_num = int(np.where(num==1)[0])
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


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    patch_in_box = False


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            # print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            check_model = load_model('/home/xddz/PyCharmProjects/python36/MaGo-container_num_rec/checks/check_model_2018_12_31.h5')
            for im_fn in im_fn_list:
                data = {"name": os.path.basename(im_fn)}
                # im = cv2.imread(im_fn)[:, :, ::-1]
                print('LCH: the im_fn is ', im_fn)
                start_time = time.time()
                number = ''
                im = cv2.imread(im_fn)
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start

                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                # print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                #     im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                patches = get_patchs(im, boxes)
                patches, index = patches_filter(patches)
                boxes = boxes[index]

                boxes = filter_boxes(im, boxes, config)
                patches = get_patchs(im, boxes, with_morph=False)
                assert len(patches) == 2 or 3
                if len(patches) == 2:
                    for i, (patch, conf, size) in enumerate(zip(patches, config, [4, 6])):
                        if Debug_mode:
                            plt.imshow(patch)
                            plt.show()

                        if i == 0:
                            temp = pytesseract.image_to_string(patch, config=conf)
                            if temp is None:
                                patch = cv2.morphologyEx(patch, cv2.MORPH_ERODE, np.ones((3, 3), np.int8), iterations=2)
                                temp = pytesseract.image_to_string(patch, config=conf)
                        if i == 1:
                            temp = pytesseract.image_to_string(patch, config=conf)
                            text_box = boxes_to_array(pytesseract.image_to_boxes(patch, config=conf))
                            assert text_box.shape[0] >= 6
                            if int(text_box[5, 3]) not in Interval(patch.shape[1]*0.9, patch.shape[1]) or len(temp) != size:
                                if Debug_mode:
                                    print("LCH: the check_num is included! temp size is ", len(temp))
                            # if not len(temp) == size:
                                temp = temp[:6]
                                patch_in_box = True
                        number += temp
                    check_patch, origin_point = get_check_patch(im, boxes[-1], flag=patch_in_box)
                else:
                    for i, (patch, conf, size) in enumerate(zip(patches, [config[0], config[1], config[1]], [4, 3, 3])):
                        if Debug_mode:
                            plt.imshow(patch)
                            plt.show()

                        if i == 0:
                            temp = pytesseract.image_to_string(patch, config=conf)
                            if temp is None:
                                patch = cv2.morphologyEx(patch, cv2.MORPH_ERODE, np.ones((3, 3), np.int8), iterations=2)
                                temp = pytesseract.image_to_string(patch, config=conf)
                        else:
                            temp = pytesseract.image_to_string(patch, config=conf)
                            text_box = boxes_to_array(pytesseract.image_to_boxes(patch, config=conf))
                            if int(text_box[2, 3]) not in Interval(patch.shape[1]*0.85, patch.shape[1]) or len(temp) != size:
                                if Debug_mode:
                                    print("LCH: the check_num is included! temp size is ", len(temp))
                            # if not len(temp) == size:
                                temp = temp[:3]
                                patch_in_box = True
                        number += temp

                    check_patch, origin_point = get_check_patch(im, boxes[-1], text_len=3, flag=patch_in_box)

                number = number[-10:]
                assert len(number) == 10

                # cv2.imwrite('check_patch.jpg', check_patch)
                if Debug_mode:
                    plt.imshow(check_patch)
                    plt.show()
                check_tl, check_br, check_patch = locate_check_num(check_patch, origin_point)
                if Debug_mode:
                    plt.imshow(check_patch)
                    plt.show()
                check_num = detect_check_num(check_patch, check_model)
                result = check_container_number(number, check_num)
                message = number + ' [' + str(check_num) + ']  ' + str(result)

                print('LCH: the', os.path.basename(im_fn), ' detected number is ', number, 'check_num is ', check_num)
                for box in boxes:
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32)) ### box coord declare here
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
                cv2.putText(im, message, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness=3)
                cv2.rectangle(im, check_tl, check_br, (255, 0, 0), 2)
                patch_in_box = False

                data = {"name": os.path.basename(im_fn), "boxes_num": boxes.shape[0]+1, "check": result, "container_num":number+str(check_num)}
                json_message = json.dumps(data, cls=MyEncoder)
                print(json_message)

                print('LCH: the cost time is ', time.time() - start_time, ' seconds. ')
                if Debug_mode:
                    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    plt.show()
                else:
                    cv2.imshow('output', im)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        saved_name = 'detected_'+os.path.basename(im_fn)
                        cv2.imwrite(saved_name, im)
                        print(saved_name, ' saved ...')

if __name__ == '__main__':
    tf.app.run()
