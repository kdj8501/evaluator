from ultralytics import YOLO

def get_result_yolo(path, model):
    pre = []
    result = model(path)
    for r in result:
        for b in r.boxes.xywhn:
            pre.append(b.tolist())
        idx = 0
        for c in r.boxes.cls: # cls id 보정
            cls = int(c)
            if (cls == 4 or cls == 6 or cls > 7): # 비행기, 기차, 보트, 이 외 사물 제외
                cls = -1
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
        idx = 0
        for i in range(len(pre)):
            if (pre[idx][0] == -1):
                del(pre[idx])
                continue
            idx += 1
    return pre

# path = 'C:/Users/koo/workspace/dataset/labeling/test2/images/'
# file = 'bandicam-2025-01-23-09-55-34-532_jpg.rf.949978e7322e14e093a5ce33546b6ca8.jpg'
# model = YOLO('yolo11x-seg.pt')
# model.predict(path + file, save = True, show_boxes = False, show_conf = False, show_labels = False)
# model.predict(path + "bandicam-2025-01-23-09-55-34-532_jpg.rf.949978e7322e14e093a5ce33546b6ca8.jpg", save = True)
# model.predict(path + "bandicam-2025-01-23-09-56-10-160_jpg.rf.156accc366cb93b72c21aa5b75fa9464.jpg", save = True)
# model.predict(path + "bandicam-2025-01-23-09-56-31-810_jpg.rf.1964939a01112632b6e201a727512e6a.jpg", save = True)
# model.predict()