import tensorflow as tf
import matplotlib.pyplot as plt
import yaml
from keras.models import load_model
from tools_crnn import *
import model
import socket

config = ['--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0',
          '--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789']

Debug_mode = False
local_mode = False

def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    patch_in_box = False
    f = open("./config/config.yaml", encoding='utf-8')
    parameters = yaml.load(f)

    addr = (parameters['socket']['ip'], parameters['socket']['port'])
    # client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

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
            ckpt_state = tf.train.get_checkpoint_state(parameters['model']['checkpoint_path'])
            model_path = os.path.join(parameters['model']['checkpoint_path'], os.path.basename(ckpt_state.model_checkpoint_path))
            check_model = load_model(parameters['model']['check_num_model'])
            crnn = CRNN_REC(parameters['model']['crnn_model'])
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            while True:
                f = open("./config/config.yaml", encoding='utf-8')
                parameters = yaml.load(f)
                local_time = time.localtime()

                TempDir = os.path.join(parameters['image_paths']['TempDir'], "{0:04d}{1:02d}{2:02d}".format(local_time[0], local_time[1], local_time[2]))
                DetectDir = os.path.join(parameters['image_paths']['DetectDir'], "{0:04d}{1:02d}{2:02d}".format(local_time[0], local_time[1], local_time[2]))
                FailDir = os.path.join(parameters['image_paths']['FailDir'], "{0:04d}{1:02d}{2:02d}".format(local_time[0], local_time[1], local_time[2]))
                MissDetectDir = os.path.join(parameters['image_paths']['MissDetectDir'], "{0:04d}{1:02d}{2:02d}".format(local_time[0], local_time[1], local_time[2]))

                for i in [TempDir, DetectDir, FailDir, MissDetectDir]:
                    if not os.path.exists(i):
                        os.makedirs(i)

                im_fn_list = get_images(TempDir)
                if len(im_fn_list) == 0:
                    print('LCH: Waiting images to come ...')
                    time.sleep(1)
                    continue

                for im_fn in im_fn_list:
                    print('LCH: the im_fn is ', im_fn)
                    # client_sock.connect(addr)
                    start_time = time.time()
                    number = ''
                    im = cv2.resize(cv2.imread(im_fn), (1280, 720))
                    im_resized, (ratio_h, ratio_w) = resize_image(im)

                    try:
                    # if True:
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

                        # patches = get_patchs(im, boxes, **parameters)
                        # patches, index = patches_filter(patches)
                        # boxes = boxes[index]

                        boxes = filter_boxes(im, boxes, crnn, **parameters)
                        patches = get_patchs(im, boxes, with_morph=False, **parameters)
                        assert len(patches) == 2 or 3
                        if len(patches) == 2:
                            for i, (patch, conf, size) in enumerate(zip(patches, config, [4, 6])):
                                if Debug_mode:
                                    plt.imshow(patch)
                                    plt.show()

                                if i == 0:
                                    temp = crnn.recognize(patch)
                                    if len(temp) < size:
                                        temp = pytesseract.image_to_string(patch, config=conf)
                                if i == 1:
                                    # temp = crnn.recognize(patch)
                                    # if len(temp) < size:
                                    #     temp = pytesseract.image_to_string(patch, config=conf)
                                    temp = pytesseract.image_to_string(patch, config=conf)
                                    if len(temp) < size:
                                        for _ in range(size-len(temp)):
                                            boxes[-1] = expand_box_width(boxes[-1])
                                            expanded_patch = get_patchs(im, [boxes[-1]], with_morph=False, **parameters)[0]
                                        if Debug_mode:
                                            plt.imshow(expanded_patch)
                                            plt.show()
                                        temp = crnn.recognize(expanded_patch)
                                        # temp = pytesseract.image_to_string(patch, config=conf).replace(' ', '')
                                        if len(temp) < size:
                                            temp = pytesseract.image_to_string(expanded_patch, config=conf)

                                    text_box = boxes_to_array(pytesseract.image_to_boxes(patch, config=conf))
                                    if text_box.shape[0] >= size:
                                        if int(text_box[5, 3]) not in Interval(patch.shape[1]*0.76, patch.shape[1]):
                                            if Debug_mode:
                                                print("LCH: the check_num is included! temp size is ", len(temp))
                                            temp = temp[:6]
                                            patch_in_box = True

                                number += temp
                            check_patch, origin_point = get_check_patch(im, boxes[-1], config, flag=patch_in_box, **parameters)
                        else:
                            for i, (patch, conf, size) in enumerate(zip(patches, [config[0], config[1], config[1]], [4, 3, 3])):
                                if Debug_mode:
                                    plt.imshow(patch)
                                    plt.show()

                                if i == 0:
                                    temp = crnn.recognize(patch)
                                else:
                                    # temp = crnn.recognize(patch)
                                    temp = pytesseract.image_to_string(patch, config=conf).replace(' ', '')
                                    if temp == '':
                                        temp = crnn.recognize(patch)
                                    if len(temp) >= 6:
                                        number += temp[:6]
                                        boxes = np.delete(boxes, -1, 0)
                                        break
                                    # if len(temp) < size and i == 1:
                                    #     for _ in range(size-len(temp)):
                                    #         boxes[-1] = expand_box_width(boxes[-1])
                                    #         expanded_patch = get_patchs(im, [boxes[-1]], with_morph=False, **parameters)[0]
                                    #     temp = pytesseract.image_to_string(patch, config=conf).replace(' ', '')
                                    #     patch = expanded_patch
                                if i == 2:
                                    text_box = boxes_to_array(pytesseract.image_to_boxes(patch, config=conf))
                                    if int(text_box[2, 3]) not in Interval(patch.shape[1]*0.83, patch.shape[1]) and len(temp) != size:
                                        if Debug_mode:
                                            print("LCH: the check_num is included! temp size is ", len(temp))
                                        temp = temp[:3]
                                        patch_in_box = True

                                number += temp

                            check_patch, origin_point = get_check_patch(im, boxes[-1], config, text_len=3, flag=patch_in_box, **parameters)

                        number = number[-10:]
                        assert len(number) == 10

                        if Debug_mode:
                            print('LCH: check_patch detect')
                            plt.imshow(check_patch)
                            plt.show()
                        check_tl, check_br, check_patch = locate_check_num(check_patch, origin_point, **parameters)
                        if Debug_mode:
                            plt.imshow(check_patch)
                            plt.show()
                        check_num = detect_check_num(check_patch, check_model)
                        result = check_container_number(number, check_num)
                        message = number + ' [' + str(check_num) + ']  ' + str(result)

                        print('LCH: the', os.path.basename(im_fn), ' detected number is ', number, 'check_num is ', check_num)
                        for box in boxes:
                            box = sort_poly(box.astype(np.int32)) ### box coord declare here
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
                        cv2.putText(im, message, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness=3)
                        cv2.rectangle(im, check_tl, check_br, (255, 0, 0), 2)
                        patch_in_box = False

                        data = {"name": os.path.basename(im_fn), "boxes_num": boxes.shape[0]+1, "check": result, "container_num":number+str(check_num)}
                        json_message = json.dumps(data, cls=MyEncoder)
                        json_message += "\r\n\r\n"
                        print(json_message)

                        print('LCH: the cost time is ', time.time() - start_time, ' seconds. ')
                        if local_mode:
                            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                            plt.show()
                        else:
                            if result:
                                client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                client_sock.connect(addr)
                                # client_sock.send(bytes(json_message, encoding='utf-8'))
                                client_sock.send(json_message.encode('utf-8'))
                                print('LCH: json message sent to ', addr)
                                cv2.imwrite(os.path.join(DetectDir, os.path.basename(im_fn)), im)
                                client_sock.close()
                            else:
                                client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                client_sock.connect(addr)
                                # client_sock.send(bytes(json_message, encoding='utf-8'))
                                client_sock.send(json_message.encode('utf-8'))
                                print('LCH: json message sent to ', addr)
                                cv2.imwrite(os.path.join(MissDetectDir, os.path.basename(im_fn)), im)
                                client_sock.close()
                        os.remove(im_fn)

                    except:
                        if local_mode:
                            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                            plt.show()
                        else:
                            cv2.imwrite(os.path.join(FailDir, os.path.basename(im_fn)), im)
                            os.remove(im_fn)


if __name__ == '__main__':
    main()
