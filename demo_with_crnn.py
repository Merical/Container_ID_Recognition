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
from tools_crnn import *

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


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    patch_in_box = False
    crnn = CRNN_REC('./checks/mago_crnn_2019_1_25.pth')

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

            im_fn_list = get_images('./checks/backup')
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
                            temp = crnn.recognize(patch)
                            # if temp is None:
                            #     patch = cv2.morphologyEx(patch, cv2.MORPH_ERODE, np.ones((3, 3), np.int8), iterations=2)
                            #     temp = crnn.recognize(patch)
                        if i == 1:
                            temp = crnn.recognize(patch)
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
                            temp = crnn.recognize(patch)
                            # if temp is None:
                            #     patch = cv2.morphologyEx(patch, cv2.MORPH_ERODE, np.ones((3, 3), np.int8), iterations=2)
                            #     temp = pytesseract.image_to_string(patch, config=conf)
                        else:
                            temp = crnn.recognize(patch)
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
