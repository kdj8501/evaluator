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
            if (cls == 0 or cls == 4 or cls == 6 or cls > 7): # 사람, 비행기, 기차, 보트, 이 외 사물 제외
                del(pre[idx])
                continue
            elif (cls == 1): # bicycle 보정
                cls = 2
            elif (cls == 2): # car 보정
                cls = 1
            elif (cls == 3): # motorcycle 보정
                cls = 2
            elif (cls == 5): # bus 보정
                cls = 0
            elif (cls == 7): # truck 보정
                cls = 4
            pre[idx].insert(0, cls)
            idx += 1
    return pre

path = "C:/Users/koo/Desktop/labeling/test2/images/"
model = YOLO('yolo11x.pt')
model.predict(path + "bandicam-2025-01-23-09-55-34-532_jpg.rf.17296b1523c0b0af069a1cbf8dcbfbd9.jpg", save = True)
model.predict(path + "bandicam-2025-01-23-09-56-10-160_jpg.rf.2bc30872a419e3f5b92742167abcc793.jpg", save = True)
model.predict(path + "bandicam-2025-01-23-09-56-31-810_jpg.rf.5ee80c2a46fc09140a2e53fb262603ba.jpg", save = True)