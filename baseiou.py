import numpy as np

def getIOU_np(ref, pre):
    intersection = np.logical_and(ref, pre)
    union = np.logical_or(ref, pre)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def getDiceCoefficient(ref, pre):
    intersection = np.logical_and(ref, pre)
    a = np.logical_and(ref, ref)
    b = np.logical_and(pre, pre)
    dice = (2. * np.sum(intersection)) / (np.sum(a) + np.sum(b))
    return dice

def getIOU(list1, list2):
    x = list1[1]
    y = list1[2]
    w = list1[3]
    h = list1[4]
    x1 = x - (w / 2)
    x2 = x + (w / 2)
    y1 = y - (h / 2)
    y2 = y + (h / 2)
    sq1 = w * h
    x = list2[1]
    y = list2[2]
    w = list2[3]
    h = list2[4]
    x3 = x - (w / 2)
    x4 = x + (w / 2)
    y3 = y - (h / 2)
    y4 = y + (h / 2)
    sq2 = w * h
    x5 = max(x1, x3)
    x6 = min(x2, x4)
    y5 = max(y1, y3)
    y6 = min(y2, y4)
    if (x6 < x5 or y6 < y5):
        return 0.0
    sq3 = (x6 - x5) * (y6 - y5)
    iou = sq3 / sq1 + sq2 - sq3
    return iou

def getIOU_spec(list1, list2):
    x = list1[1]
    y = list1[2]
    w = list1[3]
    h = list1[4]
    x1 = x - (w / 2)
    x2 = x + (w / 2)
    y1 = y - (h / 2)
    y2 = y + (h / 2)
    sq1 = w * h
    x = list2[1]
    y = list2[2]
    w = list2[3]
    h = list2[4]
    x3 = x - (w / 2)
    x4 = x + (w / 2)
    y3 = y - (h / 2)
    y4 = y + (h / 2)
    sq2 = w * h
    x5 = max(x1, x3)
    x6 = min(x2, x4)
    y5 = max(y1, y3)
    y6 = min(y2, y4)
    if (x6 < x5 or y6 < y5):
        return 0.0
    sq3 = (x6 - x5) * (y6 - y5)
    iou = sq3 / sq2

    return iou

def getAccuracy(ref, pre, thres):
    if (len(ref) == 0):
        if (len(pre) == 0):
            return 1.0
        else:
            return 0.0
    else:
        if (len(pre) == 0):
            return 0.0
    tp = 0
    fp = 0
    fn = 0
    for r in ref:
        maxIOU = 0.0
        for p in pre:
            iou = getIOU(r, p)
            maxIOU = max(maxIOU, iou)
        if (maxIOU == 0.0):
            fn += 1
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
        if (maxIOU > thres):
            if (ref_type == pre_type):
                tp += 1
            else:
                fp += 1
        elif (maxIOU > 0 and maxIOU < thres):
            fp += 1
    if (tp + fp + fn == 0.0):
        return 0.0
    return tp / (tp + fp + fn)

def getPrecision(ref, pre, thres):
    if (len(ref) == 0):
        if (len(pre) == 0):
            return 1.0
        else:
            return 0.0
    if (len(pre) == 0):
        return 0.0
    tp = 0
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
        if (maxIOU > thres):
            if (ref_type == pre_type):
                tp += 1
    return tp / len(pre)

def getRecall(ref, pre, thres):
    if (len(ref) == 0):
        if (len(pre) == 0):
            return 1.0
        else:
            return 0.0
    tp = 0
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
        if (maxIOU > thres):
            if (ref_type == pre_type):
                tp += 1
    return tp / len(ref)

def getF1(ref, pre, thres):
    if ((getPrecision(ref, pre, thres) + getRecall(ref, pre, thres)) == 0.0):
        return 0.0
    return 2 * (getPrecision(ref, pre, thres) * getRecall(ref, pre, thres)) / (getPrecision(ref, pre, thres) + getRecall(ref, pre, thres))

def getAP(ref, pre, thres):
    length = len(ref)
    if (length == 0):
        if (len(pre) == 0):
            return 1.0
        else:
            return 0.0
    elif (len(pre) == 0):
        return 0.0
    tmp = []
    tp = 0
    fp = 0
    fn = 0
    if (len == 0):
        return 0.0
    for r in ref:
        maxIOU = 0.0
        for p in pre:
            iou = getIOU(r, p)
            maxIOU = max(maxIOU, iou)
        if (maxIOU == 0.0):
            fn += 1
    for p in pre:
        maxIOU = 0.0
        ref_type = -1
        pre_type = -1
        for r in ref:
            iou = getIOU(p, r)
            maxIOU = max(maxIOU, iou)
            if (maxIOU == iou):
                ref_type = r[0]
                pre_type = p[0]
        if (maxIOU > thres):
            if (ref_type == pre_type):
                tp += 1
            else:
                fp += 1

        if (tp + fp == 0.0):
            tmp.append([0.0, round(tp / length, 1)])
        else:
            tmp.append([round(tp / (tp + fp), 3), round(tp / length, 1)])

    # print("[precision, recall] by sequential")
    # print(tmp)
    ########## Draw PR Graph with 11-point interpolation ##########
    poiv = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    poin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in tmp:
        if (i[1] == 0.0):
            poiv[0] = i[0]
            poin[0] += 1
        if (i[1] == 0.1):
            poiv[1] += i[0]
            poin[1] += 1
        if (i[1] == 0.2):
            poiv[2] += i[0]
            poin[2] += 1
        if (i[1] == 0.3):
            poiv[3] += i[0]
            poin[3] += 1
        if (i[1] == 0.4):
            poiv[4] += i[0]
            poin[4] += 1
        if (i[1] == 0.5):
            poiv[5] += i[0]
            poin[5] += 1
        if (i[1] == 0.6):
            poiv[6] += i[0]
            poin[6] += 1
        if (i[1] == 0.7):
            poiv[7] += i[0]
            poin[7] += 1
        if (i[1] == 0.8):
            poiv[8] += i[0]
            poin[8] += 1
        if (i[1] == 0.9):
            poiv[9] += i[0]
            poin[9] += 1
        if (i[1] == 1.0):
            poiv[10] += i[0]
            poin[10] += 1
    apsum = 0.0
    for x in range(11):
        avrPre = 0.0
        postAvrPre = 0.0
        if (poin[x] != 0):
            avrPre = poiv[x] / poin[x]
        for y in range(11 - x - 1):
            if (poin[y + x + 1] != 0):
                postAvrPre = max(poiv[y + x + 1] / poin[y + x + 1], postAvrPre)
        apsum += max(avrPre, postAvrPre)
    ap = (apsum) / 11
    return ap

def getmAP(ref, pre, thres):
    refcls = {}
    precls = {}
    if (len(ref) == 0):
        if (len(pre) == 0):
            return 1.0
        else:
            return 0.0
    elif (len(pre) == 0):
        return 0.0

    for r in ref:
        if (str(r[0]) in refcls):
            refcls[str(r[0])].append(r)
        else:
            refcls[str(r[0])] = []
            refcls[str(r[0])].append(r)

    for p in pre:
        if (str(p[0]) in precls):
            precls[str(p[0])].append(p)
        else:
            precls[str(p[0])] = []
            precls[str(p[0])].append(p)

    ap = []
    for p in precls:
        if (p in refcls):
            ap.append(getAP(refcls[p], precls[p], thres))
        else:
            ap.append(0)
    sumAP = 0
    for a in ap:
        sumAP += a
    if (len(ap) == 0):
        return -1
    mAP = sumAP / len(ap)
    return mAP