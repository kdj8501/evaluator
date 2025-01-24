import numpy as np

def getIOU_np(ref, pre):
    intersection = np.logical_and(ref, pre)
    union = np.logical_or(ref, pre)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def getDiceCoefficient(ref, pre):
    intersection = np.logical_and(ref, pre)
    smooth = 1e-6
    dice = (2. * np.sum(intersection) + smooth) / (np.sum(pre) + np.sum(ref) + smooth)
    return dice

def getIOU(list1, list2):
    x1 = list1[1]
    y1 = list1[2]
    w1 = list1[3]
    h1 = list1[4]
    x2 = x1 + w1
    y2 = y1 + h1

    x3 = list2[1]
    y3 = list2[2]
    w2 = list2[3]
    h2 = list2[4]
    x4 = x3 + w2
    y4 = y3 + h2

    sq1 = (w1 * h1)
    sq2 = (w2 * h2)
    maxx = max(x1, x2, x3, x4)
    minx = min(x1, x2, x3, x4)
    maxy = max(y1, y2, y3, y4)
    miny = min(y1, y2, y3, y4)
    if (maxx - minx > w1 + w2 or maxy - miny > h1 + h2):
        return 0.0
    
    x5 = max(x1, x3)
    x6 = min(x2, x4)
    y5 = max(y1, y3)
    y6 = min(y2, y4)
    sq3 = (x6 - x5) * (y6 - y5)
    iou = sq3 / (sq1 + sq2 - sq3)
    # if (iou < 0.5 and iou > 0.0):
    #     print([iou, list1, list2])
    return iou

def getAccuracy(ref, pre, thres):
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
    if (tp + fp + fn == 0.0):
        return 0.0
    return tp / (tp + fp + fn)

def getPrecision(ref, pre, thres):
    if (len(pre) == 0.0):
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
    if (len(ref) == 0.0):
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

def getAP(ref, pre, len, thres):
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
            tmp.append([0.0, 0.0])
        else:
            tmp.append([round(tp / (tp + fp), 3), round(tp / len, 1)])

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
            ap.append(getAP(ref, precls[p], len(refcls[p]), thres))
        else:
            ap.append(0)
    sumAP = 0
    for a in ap:
        sumAP += a
    mAP = sumAP / len(ap)
    return mAP

def run():
    ref = []
    ref.append([0, 0.740625, 0.7140625, 0.03203125, 0.10625])
    ref.append([2, 0.3890625, 0.540625, 0.05625, 0.08359375])
    ref.append([2, 0.51875, 0.74921875, 0.0765625, 0.14140625])
    ref.append([2, 0.17890625, 0.415625, 0.0328125, 0.04140625])
    ref.append([2, 0.17109375, 0.36640625, 0.0234375, 0.0296875])
    ref.append([2, 0.225, 0.31171875, 0.01171875, 0.015625])
    ref.append([2, 0.09609375, 0.53203125, 0.071875, 0.08671875])
    ref.append([2, 0.28203125, 0.271875, 0.0078125, 0.01640625])
    ref.append([7, 0.1078125, 0.43828125, 0.04609375, 0.065625])
    ref.append([2, 0.21015625, 0.31953125, 0.0125, 0.01953125])
    ref.append([2, 0.21015625, 0.31953125, 0.0125, 0.01953125])
    ref.append([2, 0.29921875, 0.27265625, 0.00625, 0.0078125])
    ref.append([2, 0.1828125, 0.27421875, 0.00625, 0.00859375])
    ref.append([2, 0.11640625, 0.284375, 0.01953125, 0.015625])
    pre = []
    pre.append([2, 0.5197, 0.7490, 0.0814, 0.1436])
    pre.append([2, 0.3904, 0.5411, 0.0571, 0.0842])
    pre.append([2, 0.1792, 0.4162, 0.0350, 0.0418])
    pre.append([7, 0.1076, 0.4394, 0.0485, 0.0671])
    pre.append([2, 0.1711, 0.3660, 0.0263, 0.0306])
    pre.append([0, 0.7412, 0.7089, 0.0331, 0.0933])
    pre.append([2, 0.2101, 0.3214, 0.0155, 0.0195])
    pre.append([7, 0.0961, 0.5324, 0.0750, 0.0874])
    pre.append([2, 0.2257, 0.3165, 0.0157, 0.0117])
    pre.append([2, 0.0961, 0.5322, 0.0755, 0.0875])

    print("accuracy: " + str(getAccuracy(ref, pre, 0.5)))
    print("precision: " + str(getPrecision(ref, pre, 0.5)))
    print("recall: " + str(getRecall(ref, pre, 0.5)))
    print("f1: " + str(getF1(ref, pre, 0.5)))
    print("mAP: " + str(getmAP(ref, pre, 0.5)))
    
if __name__ == '__main__':
    run()