import os, time, threading, queue, cv2, torch, queue
from collections import defaultdict
from chatgpt import get_result_chatgpt
from gemini import get_result_gemini
from ultralytics import YOLO
from yolo import driver_processing, roi_processing
from baseiou import getIOU
import numpy as np
from threading import Lock

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

def tracking_1():
    model = YOLO('best.pt')
    path = 'C:/Users/koo/Desktop/수집 데이터/20250206_142913.mp4'
    cap = cv2.VideoCapture(path)
    track_history = defaultdict(lambda: [])
    count = [0, 0, 0, 0, 0]
    start_roi = [{}, 960, 250, 840, 50]
    end_roi = [100, 700, 1700, 50]
    fps = [0, 0, 0]
    def fps_thread(f):
        while (cap.isOpened()):
            f[1] = f[2]
            time.sleep(1)
            fps[0] = fps[2] - fps[1]
    f_thread = threading.Thread(target = fps_thread, args = [fps])
    f_thread.start()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('runs/run.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (int(width), int(height)))
    while cap.isOpened():
        res, frame = cap.read()
        fps[2] += 1
        if res == False:
            break
        results = model.track(frame, persist = True)
        if (results[0].boxes.id == None):
            break
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
        cv2.putText(annotated_frame, 'FPS:' + str(fps[0]), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'car:' + str(count[0]), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'truck:' + str(count[1]), (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'bus:' + str(count[2]), (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'motorcycle:' + str(count[3]), (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'wrong:' + str(count[4]), (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.rectangle(annotated_frame, (start_roi[1], start_roi[2]), (start_roi[1] + start_roi[3], start_roi[2] + start_roi[4]), (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (end_roi[0], end_roi[1]), (end_roi[0] + end_roi[2], end_roi[1] + end_roi[3]), (0, 255, 0), 2)
        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(annotated_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def tracking_2():
    model = YOLO('best.pt')
    path = 'C:/Users/koo/Desktop/수집 데이터/20250206_143447.mp4'
    cap = cv2.VideoCapture(path)
    track_history = defaultdict(lambda: [])
    count = [0, 0, 0, 0, 0]
    start_roi = [{}, 1260, 250, 330, 50]
    end_roi = [220, 700, 1260, 50]
    fps = [0, 0, 0]
    def fps_thread(f):
        while (cap.isOpened()):
            f[1] = f[2]
            time.sleep(1)
            fps[0] = fps[2] - fps[1]
    f_thread = threading.Thread(target = fps_thread, args = [fps])
    f_thread.start()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('runs/run2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (int(width), int(height)))
    while cap.isOpened():
        res, frame = cap.read()
        fps[2] += 1
        if res == False:
            break
        results = model.track(frame, persist = True)
        if (results[0].boxes.id == None):
            break
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
        cv2.putText(annotated_frame, 'FPS:' + str(fps[0]), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'car:' + str(count[0]), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'truck:' + str(count[1]), (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'bus:' + str(count[2]), (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'motorcycle:' + str(count[3]), (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, 'wrong:' + str(count[4]), (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        cv2.rectangle(annotated_frame, (start_roi[1], start_roi[2]), (start_roi[1] + start_roi[3], start_roi[2] + start_roi[4]), (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (end_roi[0], end_roi[1]), (end_roi[0] + end_roi[2], end_roi[1] + end_roi[3]), (0, 255, 0), 2)
        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(annotated_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def tracking_3():
    model = YOLO('yolo11x.pt')
    names = model.names
    path = 'C:/Users/koo/Desktop/수집 데이터/20250123_145208.mp4'
    cap = cv2.VideoCapture(path)
    track_history = defaultdict(lambda: [])
    count = {
        'person': 0,
        'car': 0,
        'truck': 0,
        'bus': 0,
        'motorcycle': 0,
        'bicycle': 0,
        'wrong': 0
    }
    veh = {}
    comp = []
    start_roi = [1260, 250, 600, 50]
    middle_roi = [600, 220, 50, 280]
    end_roi = [300, 700, 1560, 50]
    fps = [0, 0, 0]
    def fps_thread(f):
        while (cap.isOpened()):
            f[1] = f[2]
            time.sleep(1)
            fps[0] = fps[2] - fps[1]
    f_thread = threading.Thread(target = fps_thread, args = [fps])
    f_thread.start()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('runs/run4.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (int(width), int(height)))
    while cap.isOpened():
        res, frame = cap.read()
        fps[2] += 1
        if res == False:
            break
        results = model.track(frame, persist = True, classes = [0, 1, 2, 3, 5, 7])
        if (results[0].boxes.id == None):
            break
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
            if ((float(x) > start_roi[0] and float(x) < start_roi[0] + start_roi[2]) and
                (float(y + h / 2) > start_roi[1] and float(y + h / 2) < start_roi[1] + start_roi[3])):
                    if not (track_id in veh):
                        veh[track_id] = class_id
            if ((float(x) > middle_roi[0] and float(x) < middle_roi[0] + middle_roi[2]) and
                (float(y + h / 2) > middle_roi[1] and float(y + h / 2) < middle_roi[1] + middle_roi[3])):
                    if (track_id in veh):
                        if (veh[track_id] != class_id):
                            count['wrong'] += 1
                        else:
                            count[names[class_id]] += 1
                        veh.pop(track_id, None)
                        comp.append(track_id)
                    else:
                        if not(track_id in comp):
                            veh[track_id] = class_id
            if ((float(x) > end_roi[0] and float(x) < end_roi[0] + end_roi[2]) and
                (float(y + h / 2) > end_roi[1] and float(y + h / 2) < end_roi[1] + end_roi[3])):
                    if (track_id in veh):
                        if (veh[track_id] != class_id):
                            count['wrong'] += 1
                        else:
                            count[names[class_id]] += 1
                        veh.pop(track_id, None)
                        comp.append(track_id)
                    else:
                        if not(track_id in comp):
                            veh[track_id] = class_id
        cv2.putText(annotated_frame, 'FPS:' + str(fps[0]), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
        idx = 2
        for k in count.keys():
            cv2.putText(annotated_frame, k + ":" + str(count[k]), (50, 70 * idx), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
            idx += 1
        cv2.rectangle(annotated_frame, (start_roi[0], start_roi[1]), (start_roi[0] + start_roi[2], start_roi[1] + start_roi[3]), (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (middle_roi[0], middle_roi[1]), (middle_roi[0] + middle_roi[2], middle_roi[1] + middle_roi[3]), (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (end_roi[0], end_roi[1]), (end_roi[0] + end_roi[2], end_roi[1] + end_roi[3]), (0, 255, 0), 2)
        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(annotated_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
        
def tracking_4():
    MODEL = 'yolo11x.pt'
    model = YOLO(MODEL)
    fps = [0, 0, 0, True, 0, 0]
    q = queue.Queue()
    def fps_thread(f):
        while f[3]:
            f[1] = f[2]
            time.sleep(1)
            f[0] = f[2] - f[1]
    f_thread = threading.Thread(target = fps_thread, args = [fps])
    f_thread.start()
    def is_moving(track):
        return len(track) > 4 and (track[len(track) - 1][0] - track[len(track) - 5][0]) ** 2 + (track[len(track) - 1][1] - track[len(track) - 5][1]) ** 2 > 16
    def rtsp_thread(q, f):
        path = 'rtsp://admin:0p9o8i7u@@@1.233.65.68:7778/0/profile2/media.smp'
        # path = 'rtsp://admin:saloris4321@192.168.0.60:554/Streaming/Channels/101'
        cap = cv2.VideoCapture(path)
        f[4] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        f[5] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while f[3]:
            _, frame = cap.read()
            if (q.qsize() < 100):
                q.put(frame)
        cap.release()
    r_thread = threading.Thread(target = rtsp_thread, args = [q, fps])
    r_thread.start()
    def show_thread(q, f):
        names = model.names
        track_history = defaultdict(lambda: [])
        count = {
            'person': 0,
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0,
            'wrong': 0
        }
        veh = {}
        comp = []
        start_roi = [0, 1, 2, 3]
        middle_roi = [10, 300, 1910, 50]
        end_roi = [500, 5, 50, 1005]
        # start_roi = [500, 700, 1410, 50]
        # end_roi = [10, 900, 1900, 50]
        out = cv2.VideoWriter('runs/run.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (f[4], f[5]))
        while f[3]:
            if (q.qsize() > 0):
                frame = q.get()
                if MODEL == 'yolo11x.pt' or MODEL == 'yolo11n.pt':
                    results = model.track(frame, persist = True, classes = [0, 1, 2, 3, 5, 7])
                else:
                    results = model.track(frame, persist = True)
                fps[2] += 1
                flag = False
                boxes = results[0].boxes.xywh.cpu()
                track_ids = torch.Tensor().int().cpu().tolist()
                clss = torch.Tensor().int().cpu().tolist()
                if not (results[0].boxes.id == None):
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    clss = results[0].boxes.cls.int().cpu().tolist()
                annotated_frame = results[0].plot()
                for box, track_id, class_id in zip(boxes, track_ids, clss):
                    x, y, w, h = box
                    point = [float(x), float(y)]
                    track = track_history[track_id]
                    track.append((point[0], point[1]))
                    if len(track) > 50:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed = False, color = (10, 10, 230), thickness = 3)
                    if ((point[0] > start_roi[0] and point[0] < start_roi[0] + start_roi[2]) and
                        (point[1] > start_roi[1] and point[1] < start_roi[1] + start_roi[3])):
                            if not (track_id in veh):
                                veh[track_id] = class_id
                    if ((point[0] > middle_roi[0] and point[0] < middle_roi[0] + middle_roi[2]) and
                        (point[1] > middle_roi[1] and point[1] < middle_roi[1] + middle_roi[3])):
                            if (track_id in veh):
                                if (veh[track_id] != class_id):
                                    if (is_moving(track)):
                                        count['wrong'] += 1
                                else:
                                    if (is_moving(track)):
                                        count[names[class_id]] += 1
                                veh.pop(track_id, None)
                                comp.append(track_id)
                            else:
                                if not(track_id in comp):
                                    veh[track_id] = class_id
                    if ((point[0] > end_roi[0] and point[0] < end_roi[0] + end_roi[2]) and
                        (point[1] > end_roi[1] and point[1] < end_roi[1] + end_roi[3])):
                            if (track_id in veh):
                                if (veh[track_id] != class_id):
                                    if (is_moving(track)):
                                        count['wrong'] += 1
                                else:
                                    if (is_moving(track)):
                                        count[names[class_id]] += 1
                                veh.pop(track_id, None)
                                comp.append(track_id)
                            else:
                                if not(track_id in comp):
                                    veh[track_id] = class_id
                    if (is_moving(track)):
                        flag = True
                cv2.putText(annotated_frame, str(flag), (1400, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, 'FPS:' + str(f[0]), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
                idx = 2
                for k in count.keys():
                    if k == 'wrong' or k == 'bus':
                        continue
                    cv2.putText(annotated_frame, k + ":" + str(count[k]), (50, 70 * idx), cv2.FONT_HERSHEY_SIMPLEX, 3, (212, 255, 83), 3, cv2.LINE_AA)
                    idx += 1
                cv2.rectangle(annotated_frame, (start_roi[0], start_roi[1]), (start_roi[0] + start_roi[2], start_roi[1] + start_roi[3]), (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (middle_roi[0], middle_roi[1]), (middle_roi[0] + middle_roi[2], middle_roi[1] + middle_roi[3]), (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (end_roi[0], end_roi[1]), (end_roi[0] + end_roi[2], end_roi[1] + end_roi[3]), (0, 255, 0), 2)
                cv2.imshow("Tracking", annotated_frame)
                if (flag):
                    out.write(annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    f[3] = False
        out.release()
    s_thread = threading.Thread(target = show_thread, args = [q, fps])
    s_thread.start()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tracking_4()

#
# prompt = ['There are 16 images of vehicles in the 4x4 area, 
#           please answer in one word whether the vehicle in 
#           each area is a car or a truck.
#           The order of answers is from the top left to the right, 
#           and from top to bottom.']
#