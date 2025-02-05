from ultralytics import YOLO
from baseiou import getIOU_spec

def get_result_yolo(path, model, custom):
    pre = []
    result = None
    if (custom):
        result = model(path)
    else:
        result = model(path, classes = [0, 1, 2, 3, 5, 7])
    for r in result:
        for b in r.boxes.xywhn:
            pre.append(b.tolist())
        if (custom):
            idx = 0
            for c in r.boxes.cls:
                cls = int(c)
                pre[idx].insert(0, cls)
                idx += 1
        else:
            idx = 0
            for c in r.boxes.cls: # cls id 보정
                cls = int(c)
                if (cls == 0): # Person 보정
                    cls = 4
                elif (cls == 1): # bicycle 보정
                    cls = 0
                elif (cls == 2): # car 보정
                    cls = 2 
                elif (cls == 3): # motorcycle 보정
                    cls = 3
                elif (cls == 5): # bus 보정
                    cls = 1
                elif (cls == 7): # truck 보정
                    cls = 5
                pre[idx].insert(0, cls)
                idx += 1
    return pre

def roi_processing(res, x, y, w, h):
    ref = [-1, x, y, w, h]
    result = []
    for r in res:
        iou = getIOU_spec(ref, r)
        if (iou > 0.3):
            result.append(r)
    return result

def driver_processing(res):
    result = []
    for r in res:
        if (r[0] == 4):
            maxiou = 0.0
            for r2 in res:
                if (r2[0] != 4):
                    maxiou = max(maxiou, getIOU_spec(r2, r))
            if (maxiou < 0.3):
                result.append(r)
        else:
            result.append(r)
    return result

# path = 'C:/Users/PC/Desktop/field/dataset/best/images/'
# model = YOLO('best.pt')
# results = model(path + '20250123_144754_mp4-0051_jpg.rf.858502c8be34efc710cf41c26564838a.jpg')
# for r in results:
#      print(r.boxes)