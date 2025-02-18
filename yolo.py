from ultralytics import YOLO
from baseiou import getIOU_spec

def getClasses(names): # get classes list from model.names
    res = []
    for n in names: # get classes only for vehicles on the road
        if (names[n] == 'person' or
            names[n] == 'bicycle' or
            names[n] == 'car' or
            names[n] == 'motorcycle' or
            names[n] == 'bus' or
            names[n] == 'truck'):
            res.append(n)
    return res

def get_result_yolo(path, model, names):
    pre = []
    result = None
    cls_ref = {}
    for i in range(len(names)):
        cls_ref[names[i]] = i
    result = model(path, classes = getClasses(model.names))
    for r in result:
        for b in r.boxes.xywhn:
            pre.append(b.tolist())
        idx = 0
        for c in r.boxes.cls:
            cls = -1
            if (int(c) in model.names):
                if (model.names[int(c)] in cls_ref):
                    cls = cls_ref[model.names[int(c)]]
            pre[idx].insert(0, cls)
            idx += 1
    return pre

def roi_processing(res, roi):
    ref = [-1, roi[0], roi[1], roi[2], roi[3]]
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