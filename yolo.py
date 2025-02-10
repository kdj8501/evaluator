from ultralytics import YOLO
from baseiou import getIOU_spec

def get_result_yolo(path, model, custom, names):
    pre = []
    result = None
    cls_ref = {}
    for i in range(len(names)):
        cls_ref[names[i]] = i
    if (custom):
        result = model(path)
    else:
        result = model(path, classes = [0, 1, 2, 3, 5, 7])
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