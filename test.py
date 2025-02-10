import os, time, threading, queue, cv2
from collections import defaultdict
from chatgpt import get_result_chatgpt
from gemini import get_result_gemini
from ultralytics import YOLO
from yolo import driver_processing, roi_processing
from baseiou import getIOU
import numpy as np

q = queue.Queue()
flag = True

def yolo20_test():
    path = 'C:/Users/koo/workspace/dataset/20yolo'
    prompt = "Is this car or truck? Answer just car or truck in one word."
    res = []
    for i in range(20):
        img = path + '/images/' + str(i) + '.png'
        f = open(path + '/labels/' + str(i) + '.txt', 'r')
        ans = f.readline()
        f.close()
        gpt = get_result_chatgpt(img, prompt).lower().split('\n')[0].split('.')[0]
        print(gpt)
        if not (gpt == 'car' or gpt == 'truck'):
            gpt = 'null'
        res.append([i, ans, gpt])
    print('0~9: wrong answer, 10~19: correct answer')
    print('[i, answer, gpt-4o-mini]')
    for r in res:
        print(r)

def yolo20_test_matrix():
    path = 'C:/Users/koo/Desktop/3.png'
    prompt = 'There are 4 images of vehicles in the 2x2 area, please answer in one word whether the vehicle in each area is a car or a truck. The order of answers is from the top left to the right, and from top to bottom.'
    start = time.time()
    result = get_result_chatgpt(path, prompt)
    end = time.time()
    print (str(end - start) + ' secs')
    print(result)

def yolo_test():
    path = 'C:/Users/PC/Desktop/field/dataset/2025-01-23_데이터수집_동구청/20250123_145208.mp4'
    model = YOLO('best.pt')
    model.predict(path, save = True)
    # model.predict(path, save = True, show_labels = False, show_conf = False)

def yolo_conf_test():
    path = 'C:/Users/PC/Desktop/field/dataset/best'
    lis = os.listdir(path + '/images')
    model = YOLO('yolo11x.pt')
    count = 0
    wrong_conf = []
    for l in lis:
        pre = []
        ref = []
        results = model(path + '/images/' + l, classes = [0, 1, 2, 3, 5, 7])
        for r in results:
            for b in r.boxes.xywhn:
                pre.append(b.tolist())
                count += 1
            idx = 0
            for c in r.boxes.cls:
                cls = int(c)
                if (cls == 0):
                    cls = 4
                elif (cls == 1):
                    cls = 0
                elif (cls == 2):
                    cls = 2
                elif (cls == 3):
                    cls = 3
                elif (cls == 5):
                    cls = 1
                elif (cls == 7):
                    cls = 5
                pre[idx].insert(0, cls)
                idx += 1
            idx = 0
            for con in r.boxes.conf:
                conf = float(con)
                pre[idx].append(conf)
                idx += 1
        pre = driver_processing(pre)
        ref = roi_processing(ref, 0.5, 0.75, 1.0, 0.5)
        pre = roi_processing(pre, 0.5, 0.75, 1.0, 0.5)
        f = open(path + '/labels/' + l[:l.rfind('.')] + '.txt', 'r')
        tmp = f.readlines()
        f.close()
        for t in tmp:
            tt = t.split(' ')
            ref.append([int(tt[0]), float(tt[1]), float(tt[2]), float(tt[3]), float(tt[4])])
        
        if (len(pre) > 0):
            for p in pre:
                maxIOU = 0.0
                ref_type = -1
                pre_type = -1
                for r in ref:
                    iou = getIOU(r, p)
                    maxIOU = max(maxIOU, iou)
                    if (maxIOU == iou):
                        ref_type = r[0]
                        pre_type = p[0]
                if (maxIOU > 0.5):
                    if (ref_type != pre_type):
                        wrong_conf.append([l, ref_type, pre_type, p[5]])
                else:
                    wrong_conf.append([l, -1, p[0], p[5]])
    for w in wrong_conf:
        print(w)
    print('total objs: ' + str(count) + ', wrong objs: ' + str(len(wrong_conf)))

def isInArea(area, pre):
    result = False
    x1 = area[1] - area[3] / 2
    x2 = area[1] + area[3] / 2
    y1 = area[2] - area[4] / 2
    y2 = area[2] + area[4] / 2
    if ((pre[1] > x1 and pre[1] < x2) and (pre[2] > y1 and pre[2] < y2)):
        result = True
    return result

def prompt_thread():
    global flag
    while (True):
        prom = input()
        if (prom == ''):
            flag = False

def verify_thread():
    global flag
    while (flag):
        if (q.qsize() > 0):
            pop = q.get()

def yolo_grid_test():
    prompt = threading.Thread(target = prompt_thread, args = [])
    prompt.start()
    verify = threading.Thread(target = verify_thread, args = [])
    verify.start()
    col = 10
    row = 10
    g_w = round(1 / col, 2)
    g_h = round(1 / row, 2)
    grids = []
    for r in range(row):
        row_grids = []
        for c in range(col):
            dic = {
                'car': 0.0,
                'bus': 0.0,
                'truck': 0.0
            }
            xywh = [dic, round(g_w / 2 + c * g_w, 2), round(g_h / 2 + r * g_h, 2), g_w, g_h]
            row_grids.append(xywh)
        grids.append(row_grids)

    path = 'C:/Users/koo/workspace/dataset/labeling/2510_new'
    lis = os.listdir(path + '/images')
    model = YOLO('yolo11x.pt')
    count = 0
    total = 0
    for l in lis:
        pre = []
        results = model(path + '/images/' + l, classes = [0, 1, 2, 3, 5, 7])
        for r in results:
            for b in r.boxes.xywhn:
                pre.append(b.tolist())
            idx = 0
            for c in r.boxes.cls:
                cls = int(c)
                pre[idx].insert(0, cls)
                idx += 1
        pre = driver_processing(pre)
        pre = roi_processing(pre, [0.675, 0.8, 0.75, 0.4])
        if (len(pre) > 0):
            for p in pre:
                total += 1
                for row in grids:
                    for c in row:
                        if (isInArea(c, p)):
                            size = p[3] * p[4]
                            if (p[0] == 7):
                                if (c[0]['truck'] and c[0]['car']):
                                    if (min(size, c[0]['truck']) / max(size, c[0]['truck']) < c[0]['car'] / c[0]['truck']):
                                        count += 1
                                    c[0]['truck'] = (size + c[0]['truck']) / 2
                                else:
                                    if (c[0]['truck']):
                                        c[0]['truck'] = (size + c[0]['truck']) / 2
                                    else:
                                        c[0]['truck'] = size
                            elif (p[0] == 2):
                                if (c[0]['car'] and c[0]['truck']):
                                    if (min(size, c[0]['car']) / max(size, c[0]['car']) < c[0]['car'] / c[0]['truck']):
                                        count += 1
                                    c[0]['car'] = (size + c[0]['car']) / 2
                                else:
                                    if (c[0]['car']):
                                        c[0]['car'] = (size + c[0]['car']) / 2
                                    else:
                                        c[0]['car'] = size
                            elif (p[0] == 5):
                                if (c[0]['bus'] and c[0]['truck']):
                                    if (min(size, c[0]['bus']) / max(size, c[0]['bus']) < c[0]['truck'] / c[0]['bus']):
                                        count += 1
                                    c[0]['bus'] = (size + c[0]['bus']) / 2
                                else:
                                    if (c[0]['bus']):
                                        c[0]['bus'] = (size + c[0]['bus']) / 2
                                    else:
                                        c[0]['bus'] = size
    print(grids)
    print('[candidate objs count / total objs count]')
    print([count, total])

def tracking():
    model = YOLO('best.pt')
    path = 'C:/Users/koo/Desktop/수집 데이터/20250206_142913.mp4'
    cap = cv2.VideoCapture(path)
    track_history = defaultdict(lambda: [])
    count = [0, 0, 0, 0, 0]
    start_roi = [{}, 960, 250, 840, 50]
    end_roi = [100, 700, 1700, 50]
    while cap.isOpened():
        res, frame = cap.read()
        if res == False:
            break
        results = model.track(frame, persist = True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()
        annotated_frame = results[0].plot()

        for box, track_id, class_id in zip(boxes, track_ids, clss):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y + h / 2)))
            if len(track) > 50:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed = False, color = (10, 10, 230), thickness = 3)
            if ((float(x) > start_roi[1] and float(x) < start_roi[1] + start_roi[3]) and
                (float(y + h / 2) > start_roi[2] and float(y + h / 2) < start_roi[2] + start_roi[4])):
                    if not (track_id in start_roi[0]):
                        start_roi[0][track_id] = class_id
            if ((float(x) > end_roi[0] and float(x) < end_roi[0] + end_roi[2]) and
                (float(y + h / 2) > end_roi[1] and float(y + h / 2) < end_roi[1] + end_roi[3])):
                    if (track_id in start_roi[0]):
                        if (start_roi[0][track_id] != class_id):
                            count[4] += 1
                        elif (class_id == 2):
                            count[0] += 1
                        elif (class_id == 5):
                            count[1] += 1
                        elif (class_id == 1):
                            count[2] += 1
                        elif (class_id == 3):
                            count[3] += 1
                        start_roi[0].pop(track_id, None)
        cv2.putText(annotated_frame, 'car:' + str(count[0]), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'truck:' + str(count[1]), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'bus:' + str(count[2]), (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'motorcycle:' + str(count[3]), (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'wrong:' + str(count[4]), (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.rectangle(annotated_frame, (start_roi[1], start_roi[2]), (start_roi[1] + start_roi[3], start_roi[2] + start_roi[4]), (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (end_roi[0], end_roi[1]), (end_roi[0] + end_roi[2], end_roi[1] + end_roi[3]), (0, 255, 0), 2)
        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tracking()

#
# prompt = ['There are 16 images of vehicles in the 4x4 area, 
#           please answer in one word whether the vehicle in 
#           each area is a car or a truck.
#           The order of answers is from the top left to the right, 
#           and from top to bottom.']
#