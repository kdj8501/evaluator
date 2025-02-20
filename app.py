import sys, os, yaml, cv2, torch, evaluate, xlsxwriter
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL import Image
import numpy as np
import paligemma, resnet, yolo, segformer, baseiou
from resnet import Vocabulary
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO
import time

################################## Thread for testing ##################################
class WorkerForResnet(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(str, list, list, int)

    def __init__(self, lines, model, path, vocab, device):
        super().__init__()
        self.model = model
        self.lines = lines
        self.path = path
        self.vocab = vocab
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.flag = False
        self.device = device

    def thread_stop(self):
        self.flag = True

    def run(self):
        report = []
        idx = 0
        total = len(self.lines)
        sum_bleu_score = 0.0
        sum_rouge_score = 0.0
        sum_meteor_score = 0.0
        date = time.strftime('%Y_%m_%d_%H_%M_%S')
        
        for l in self.lines:
            if self.flag == True:
                return
            img = Image.open(self.path + "/Images/" + l.split("\n")[0].split(",")[0])
            answer = l.split("\n")[0].split(",")[1]
            generate = resnet.generate_caption_for_image(img, self.model, self.vocab, self.device)
            idx += 1
            ref = []
            pre = []
            ref.append(answer)
            pre.append(generate)
            bleu = self.bleu.compute(references = ref, predictions = pre)['precisions'][0]
            rouge = self.rouge.compute(references = ref, predictions = pre)['rouge1'].item()
            meteor = self.meteor.compute(references = ref, predictions = pre)['meteor'].item()
            sum_bleu_score += bleu
            sum_rouge_score += rouge
            sum_meteor_score += meteor

            report.append([l.split("\n")[0].split(",")[0], answer, generate, str(bleu), str(rouge), str(meteor)])
            print("[" + str(idx) + "/" + str(total) + "] " + answer + ", " + generate + " | scores b: " + str(bleu) + " r: " + str(rouge) + " m: " + str(meteor))
            self.sig.emit(int(idx / total * 100))

        total_avr = [sum_bleu_score / total, sum_rouge_score / total, sum_meteor_score / total]
        self.rep.emit(date, report, total_avr, 0)
        print("Total Average Score = b: " + str(total_avr[0]) + " r: " + str(total_avr[1]) + " m: " + str(total_avr[2]))

class WorkerForPali(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(str, list, list, int)

    def __init__(self, lines, model, processor, path, device):
        super().__init__()
        self.model = model
        self.processor = processor
        self.lines = lines
        self.path = path
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.flag = False
        self.device = device

    def thread_stop(self):
        self.flag = True

    def run(self):
        report = []
        idx = 0
        total = len(self.lines)
        sum_bleu_score = 0.0
        sum_rouge_score = 0.0
        sum_meteor_score = 0.0
        date = time.strftime('%Y_%m_%d_%H_%M_%S')
        
        for l in self.lines:
            if self.flag == True:
                return
            img = self.path + "/Images/" + l.split("\n")[0].split(",")[0]
            answer = l.split("\n")[0].split(",")[1]
            generate = paligemma.get_result_pali(img, "describe en", self.model, self.processor, self.device)
            idx += 1
            ref = []
            pre = []
            ref.append(answer)
            pre.append(generate)
            bleu = self.bleu.compute(references = ref, predictions = pre)['precisions'][0]
            rouge = self.rouge.compute(references = ref, predictions = pre)['rouge1'].item()
            meteor = self.meteor.compute(references = ref, predictions = pre)['meteor'].item()
            sum_bleu_score += bleu
            sum_rouge_score += rouge
            sum_meteor_score += meteor

            report.append([l.split("\n")[0].split(",")[0], answer, generate, str(bleu), str(rouge), str(meteor)])
            print("[" + str(idx) + "/" + str(total) + "] " + answer + ", " + generate + " | scores b: " + str(bleu) + " r: " + str(rouge) + " m: " + str(meteor))
            self.sig.emit(int(idx / total * 100))

        total_avr = [sum_bleu_score / total, sum_rouge_score / total, sum_meteor_score / total]
        self.rep.emit(date, report, total_avr, 0)
        print("Total Average Score = b: " + str(total_avr[0]) + " r: " + str(total_avr[1]) + " m: " + str(total_avr[2]))

class WorkerForYolo(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(str, list, list, int)
    disp = pyqtSignal(np.ndarray)

    def __init__(self, model, path, thres, roi):
        super().__init__()
        self.model = model
        self.path = path
        self.flag = False
        self.thres = thres
        self.roi = roi

    def thread_stop(self):
        self.flag = True

    def run(self):
        date = time.strftime('%Y_%m_%d_%H_%M_%S')
        report = []
        idx = 0
        sum_accuracy = 0.0
        sum_precision = 0.0
        sum_recall = 0.0
        sum_mAP = 0.0
        data = os.listdir(self.path + "/images")
        names = []
        with open(self.path + '/data.yaml') as f:
            film = yaml.load(f, Loader = yaml.FullLoader)
            names = film['names']
        for i in range(len(data)):
            data[i] = data[i][:data[i].rfind(".")]
        total = len(data)
        total_cls_sum = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        total_cls = [0, 0, 0, 0, 0, 0]
        total_cls_avr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for d in data:
            if (self.flag):
                return
            idx += 1
            f = open(self.path + "/labels/" + d + ".txt", 'r')
            tmp = f.readlines()
            f.close()
            ref = []
            for t in tmp:
                tt = t.split(" ")
                ref.append([int(tt[0]), float(tt[1]), float(tt[2]), float(tt[3]), float(tt[4])])
            pre = yolo.get_result_yolo(self.path + "/images/" + d + ".jpg", self.model, names)
            #####
            pre = yolo.driver_processing(pre)
            ref = yolo.roi_processing(ref, self.roi)
            pre = yolo.roi_processing(pre, self.roi)
            #####
            ############################# DISPLAY #############################
            if not os.path.exists('report/' + date):
                os.makedirs('report/' + date)
            img = cv2.imread(self.path + '/images/' + d + '.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            for r in ref:
                c_x, c_y = (int(r[1] * w), int(r[2] * h))
                c_w, c_h = (int(r[3] * w), int(r[4] * h))
                c_x1, c_x2 = (int(c_x - c_w / 2), int(c_x + c_w / 2))
                c_y1, c_y2 = (int(c_y - c_h / 2), int(c_y + c_h / 2))
                cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
                cv2.putText(img, str(r[0]), (c_x1 + 10, c_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            for p in pre:
                c_x, c_y = (int(p[1] * w), int(p[2] * h))
                c_w, c_h = (int(p[3] * w), int(p[4] * h))
                c_x1, c_x2 = (int(c_x - c_w / 2), int(c_x + c_w / 2))
                c_y1, c_y2 = (int(c_y - c_h / 2), int(c_y + c_h / 2))
                cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), (0, 0, 255), 2)
                cv2.putText(img, str(p[0]), (c_x2 - 30, c_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, str(p[5]), (c_x2 - 100, c_y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            self.disp.emit(img)
            save_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            save_img = Image.fromarray(img, 'RGB')
            save_img.save('report/' + date + '/' + d + '.jpg', 'JPEG')
            ####################################################################
            ref_cls = [[], [], [], [], [], []]
            pre_cls = [[], [], [], [], [], []]
            ap_cls = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
            for i in range(6):
                for p in pre:
                    if (p[0] == i):
                        pre_cls[i].append(p)
                for r in ref:
                    if (r[0] == i):
                        ref_cls[i].append(r)
                if not (len(pre_cls[i]) == 0 and len(ref_cls[i]) == 0):
                    ap_cls[i] = baseiou.getAP(ref_cls[i], pre_cls[i], self.thres)
                    total_cls_sum[i] += ap_cls[i]
                    total_cls[i] += 1

            precision = baseiou.getPrecision(ref, pre, self.thres)
            recall = baseiou.getRecall(ref, pre, self.thres)
            acc = baseiou.getAccuracy(ref, pre, self.thres)
            mAP = baseiou.getmAP(ref, pre, self.thres)
            sum_accuracy += acc
            sum_precision += precision
            sum_recall += recall
            sum_mAP += mAP
            # print([pre, ref])
            report.append([d, precision, recall, acc, mAP, ap_cls[0], ap_cls[1], ap_cls[2], ap_cls[3], ap_cls[4], ap_cls[5]])
            print("[" + str(idx) + "/" + str(total) + "] img_name: " + d + ", [P, R, A, mAP]: " + str([precision, recall, acc, mAP]))
            self.sig.emit(int(idx / total * 100))

        total_avr_acc = sum_accuracy / total
        total_avr_prc = sum_precision / total
        total_avr_rec = sum_recall / total
        total_avr_map = sum_mAP / total
        for i in range(6):
            if (total_cls[i] == 0):
                total_cls_avr[i] = -1.0
                continue
            total_cls_avr[i] = total_cls_sum[i] / total_cls[i]

        self.rep.emit(date, report, [total_avr_prc, total_avr_rec, total_avr_acc, total_avr_map,
                               total_cls_avr[0], total_cls_avr[1], total_cls_avr[2],
                               total_cls_avr[3], total_cls_avr[4], total_cls_avr[5]], 1)
        print("Total Average Score = " + str([total_avr_prc, total_avr_rec, total_avr_acc, total_avr_map]))
        f = open('report/' + date + '/' + 'class_id.txt', 'w')
        lidx = 0
        for n in names:
            f.write(n + ": " + str(lidx) + '\n')
            lidx += 1
        f.close()

class WorkerForSeg(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(str, list, list, int)
    disp = pyqtSignal(np.ndarray)

    def __init__(self, path, model, processor, device):
        super().__init__()
        self.model = model
        self.path = path
        self.flag = False
        self.device = device
        self.processor = processor

    def thread_stop(self):
        self.flag = True

    def run(self):
        date = time.strftime('%Y_%m_%d_%H_%M_%S')
        orig_data = os.listdir(self.path + "/images")
        label_data = os.listdir(self.path + '/labels')
        total = len(orig_data)
        report = []
        sum_iou = 0.0
        sum_dice = 0.0
        idx = 0
        for d in label_data:
            if (self.flag):
                return
            image = Image.open(self.path + '/labels/' + d)
            image = np.array(image)
            if image.shape[2] == 4:
                image = image[:, :, :3]
            result = segformer.get_result_seg(self.path + '/images/' + orig_data[idx], self.model, self.processor, self.device)
            if result.shape[2] == 4:
                result = result[:, :, :3]
            iou = baseiou.getIOU_np(image, result)
            dice = baseiou.getDiceCoefficient(image, result)
            sum_iou += iou
            sum_dice += dice
            report.append([orig_data[idx], iou, dice])
            print("[" + str(idx + 1) + "/" + str(total) + "] img_name: " + d + ", [iou, DiceCoefficient]: " + str([iou, dice]))
            self.sig.emit(int((idx + 1) / total * 100))
            self.disp.emit(result)
            idx += 1
        total_avr_iou = sum_iou / total
        total_avr_dice = sum_dice / total
        self.rep.emit(date, report, [total_avr_iou, total_avr_dice], 2)
        print("Total Average Score = " + str([total_avr_iou, total_avr_dice]))     
        
################################## GUI ##################################
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.disp_size = [1360, 850]
        self.roi = [0.5, 0.5, 1.0, 1.0]
        self.model = None
        self.excel_path = ""
        self.data_path = ""
        self.model_path = ""
        self.thres = 0.5
        self.indexes = [0, 0, [], []]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = np.zeros((1080, 1920, 3), np.uint8)
        self.curImage = img.copy()

        self.setWindowTitle("Model Evaluator v2.3")
        self.setGeometry(50, 50, 1790, 910)
        self.setFixedSize(1790, 910)

        self.table = QTableWidget(self)
        self.table.setRowCount(self.indexes[1])
        self.table.setColumnCount(1)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.tableAction)
        self.table.move(10, 10)
        self.table.setFixedSize(400, 300)

        self.calcReport = QLabel('', self)
        self.calcReport.move(700, 20)
        self.calcReport.setFixedSize(400, 15)

        self.debug_display = QLabel(self)
        self.debug_display.move(420, 50)
        self.debug_display.setFixedSize(self.disp_size[0], self.disp_size[1])

        self.b_left = QPushButton('<', self)
        self.b_left.move(420, 10)
        self.b_left.clicked.connect(self.left)

        self.l_count = QLabel('0/0', self)
        self.l_count.setFixedSize(160, 20)
        self.l_count.setFont(QFont('굴림', 20))
        self.l_count.move(1140, 15)

        self.b_right = QPushButton('>', self)
        self.b_right.move(1680, 10)
        self.b_right.clicked.connect(self.right)

        self.roiBtn = QPushButton('관심영역 지정', self)
        self.roiBtn.move(310, 320)
        self.roiBtn.clicked.connect(self.setROI)

        self.roiResetBtn = QPushButton('관심영역 초기화', self)
        self.roiResetBtn.move(210, 320)
        self.roiResetBtn.clicked.connect(self.resetROI)

        self.display = QLabel(self)
        self.display.move(10, 365)
        self.display.setFixedSize(400, 250)

        self.pBar = QProgressBar(self)
        self.pBar.move(10, 640)
        self.pBar.resize(300, 20)
        self.pBar.setValue(0)

        self.testBtn = QPushButton("테스트 시작", self)
        self.testBtn.move(310, 635)
        self.testBtn.clicked.connect(self.test)

        self.l_thres = QLabel("IOU Threshold(%)", self)
        self.l_thres.move(55, 675)
        self.t_thres = QLineEdit("50", self)
        self.t_thres.setFixedSize(30, 20)
        self.t_thres.move(160, 680)

        self.thresBtn = QPushButton("적용", self)
        self.thresBtn.move(220, 675)
        self.thresBtn.clicked.connect(self.setThres)

        self.l_captioning = QLabel('Image Captioning', self)
        self.l_captioning.move(20, 710)
        self.radio1 = QRadioButton("ResNet + LSTM", self)
        self.radio1.move(160, 715)
        self.radio1.setFixedSize(150, 20)
        self.radio2 = QRadioButton("PaliGemma", self)
        self.radio2.move(280, 715)
        self.radio2.setFixedSize(150, 20)
        self.l_segmentation = QLabel('Segmentation', self)
        self.l_segmentation.move(20, 730)
        self.radio3 = QRadioButton("SegFormer", self)
        self.radio3.move(160, 735)
        self.radio3.setFixedSize(150, 20)
        self.l_segmentation = QLabel('Object Detection', self)
        self.l_segmentation.move(20, 750)
        self.radio4 = QRadioButton("YOLO", self)
        self.radio4.move(160, 755)
        self.radio4.setFixedSize(150, 20)
        self.radio4.setChecked(True)

        self.excelBtn = QPushButton("결과 출력 위치", self)
        self.excelBtn.move(10, 780)
        self.excelBtn.clicked.connect(self.excel)
        self.folderBtn = QPushButton("데이터 폴더", self)
        self.folderBtn.move(150, 780)
        self.folderBtn.clicked.connect(self.data)
        self.selBtn = QPushButton("모델 위치", self)
        self.selBtn.move(290, 780)
        self.selBtn.clicked.connect(self.sel_model)

        self.l_model = QLabel('Model: ' + self.model_path, self)
        self.l_model.setFixedSize(600, 20)
        self.l_model.move(10, 820)
        self.l_data = QLabel('Data Folder: ' + self.data_path, self)
        self.l_data.setFixedSize(600, 20)
        self.l_data.move(10, 840)
        self.l_report = QLabel('Export Folder: ' + self.excel_path, self)
        self.l_report.setFixedSize(600, 20)
        self.l_report.move(10, 860)

        self.display_img(img)
        self.display_debug_img(img)

    def resetROI(self):
        self.roi[0] = 0.5
        self.roi[1] = 0.5
        self.roi[2] = 1.0
        self.roi[3] = 1.0
        self.display_debug_img(self.curImage)

    def setROI(self):
        self.showMessage('드래그로 관심 영역을 지정 후 Enter키나 Space키를 \n누르고 창을 닫으면 영역이 저장됩니다.')
        img = self.curImage.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w, c = img.shape
        n_w, n_h = int(w / 2), int(h / 2)
        newImg = cv2.resize(img, dsize = (n_w, n_h), interpolation = cv2.INTER_LINEAR)
        sx, sy, sw, sh	= cv2.selectROI('set ROI', newImg, False)
        cx, cy, cw, ch = sx / n_w, sy / n_h, sw / n_w, sh / n_h
        self.roi[0] = cx + cw / 2
        self.roi[1] = cy + ch / 2
        self.roi[2] = cw
        self.roi[3] = ch
        self.display_debug_img(self.curImage)

    def setThres(self):
        self.thres = int(self.t_thres.text()) / 100

    def test(self):
        if (self.data_path == ''):
            self.showMessage('데이터 폴더를 선택해주세요.')
            return
        if (self.excel_path == ''):
            self.showMessage('결과 출력 폴더를 선택해주세요.')
            return
        if (self.model_path == ''):
            self.showMessage('모델을 선택해주세요.')
            return
        if (self.testBtn.text() == "테스트 중지"):
            self.worker.thread_stop()
            self.pBar.setValue(0)
            self.testBtn.setText("테스트 시작")
        elif (self.testBtn.text() == "테스트 시작"):
            self.testBtn.setText("테스트 중지")
            if (self.radio1.isChecked()): # ResNet + LSTM
                f = open(self.data_path + "/captions.txt", 'r')
                f.readline()
                lines = f.readlines()
                f.close()
                test = lines[:-1]
                self.worker = WorkerForResnet(test, self.model, self.data_path, self.model_path + "/vocab.pkl", self.device)
                self.worker.start()
                self.worker.sig.connect(self.process)
                self.worker.rep.connect(self.report)
            elif (self.radio2.isChecked()): # PaliGemma2
                f = open(self.data_path + "/captions.txt", 'r')
                f.readline()
                lines = f.readlines()
                f.close()
                test = lines[:-1]
                self.worker = WorkerForPali(test, self.model, self.processor, self.data_path, self.device)
                self.worker.start()
                self.worker.sig.connect(self.process)
                self.worker.rep.connect(self.report)
            elif (self.radio3.isChecked()): # SegFormer
                self.worker = WorkerForSeg(self.data_path, self.model, self.processor, self.device)
                self.worker.start()
                self.worker.sig.connect(self.process)
                self.worker.rep.connect(self.report)
                self.worker.disp.connect(self.display_img)
            elif (self.radio4.isChecked()): # YOLO
                self.worker = WorkerForYolo(self.model, self.data_path, self.thres, self.roi)
                self.worker.start()
                self.worker.sig.connect(self.process)
                self.worker.rep.connect(self.report)
                self.worker.disp.connect(self.display_img)

    def report(self, name, rep, avr, div):
        if (div == 0): # Image Captioning Reporting
            workbook = xlsxwriter.Workbook(self.excel_path + '/' + name + '_report.xlsx')
            worksheet = workbook.add_worksheet()
            worksheet.write(0, 0, "image_name")
            worksheet.write(0, 1, "reference")
            worksheet.write(0, 2, "prediction")
            worksheet.write(0, 3, "B")
            worksheet.write(0, 4, "R")
            worksheet.write(0, 5, "M")
            row = 1
            for r in rep:
                for i in range(len(r)):
                    worksheet.write(row, i, r[i])
                row += 1
            worksheet.write(row + 1, 0, "total average score")
            worksheet.write(row + 2, 0, "B")
            worksheet.write(row + 2, 1, "R")
            worksheet.write(row + 2, 2, "M")
            worksheet.write(row + 3, 0, str(avr[0]))
            worksheet.write(row + 3, 1, str(avr[1]))
            worksheet.write(row + 3, 2, str(avr[2]))
            workbook.close()
        elif (div == 1): # Object Detection Reporting
            workbook = xlsxwriter.Workbook(self.excel_path + '/' + name + '_report.xlsx')
            worksheet = workbook.add_worksheet()
            worksheet.write(0, 0, "image_name")
            worksheet.write(0, 1, "precision")
            worksheet.write(0, 2, "recall")
            worksheet.write(0, 3, "accuracy")
            worksheet.write(0, 4, "mAP." + str(int(round(self.thres, 2) * 100)))
            worksheet.write(0, 5, "bicycle AP")
            worksheet.write(0, 6, "bus AP")
            worksheet.write(0, 7, "car AP")
            worksheet.write(0, 8, "motorcycle AP")
            worksheet.write(0, 9, "person AP")
            worksheet.write(0, 10, "truck AP")
            row = 1
            for r in rep:
                for i in range(len(r)):
                    worksheet.write(row, i, r[i])
                row += 1
            worksheet.write(row + 1, 0, "total average score")
            worksheet.write(row + 1, 1, str(avr[0]))
            worksheet.write(row + 1, 2, str(avr[1]))
            worksheet.write(row + 1, 3, str(avr[2]))
            worksheet.write(row + 1, 4, str(avr[3]))
            worksheet.write(row + 1, 5, str(avr[4]))
            worksheet.write(row + 1, 6, str(avr[5]))
            worksheet.write(row + 1, 7, str(avr[6]))
            worksheet.write(row + 1, 8, str(avr[7]))
            worksheet.write(row + 1, 9, str(avr[8]))
            worksheet.write(row + 1, 10, str(avr[9]))
            workbook.close()
        elif (div == 2): # Segmentation Reporting
            workbook = xlsxwriter.Workbook(self.excel_path + '/' + name + '_report.xlsx')
            worksheet = workbook.add_worksheet()
            worksheet.write(0, 0, "image_name")
            worksheet.write(0, 1, "iou")
            worksheet.write(0, 2, "DiceCoefficient")
            row = 1
            for r in rep:
                for i in range(len(r)):
                    worksheet.write(row, i, r[i])
                row += 1
            worksheet.write(row + 1, 0, "total average score")
            worksheet.write(row + 1, 1, str(avr[0]))
            worksheet.write(row + 1, 2, str(avr[1]))
            workbook.close()

    def process(self, num):
        self.pBar.setValue(num)
        if (num == 100):
            self.testBtn.setText("테스트 시작")
            self.showMessage('테스트가 완료되어 결과 출력 폴더에 결과를 저장했습니다.')

    def display_img(self, image):
        img = image.copy()
        h, w, c = img.shape
        if (self.radio4.isChecked()):
            c_x, c_y = (int(self.roi[0] * w), int(self.roi[1] * h))
            c_w, c_h = (int(self.roi[2] * w), int(self.roi[3] * h))
            c_x1, c_x2 = (int(c_x - c_w / 2), int(c_x + c_w / 2))
            c_y1, c_y2 = (int(c_y - c_h / 2), int(c_y + c_h / 2))
            cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), (255, 0, 0), 3)
        qImg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        pixmap = pixmap.scaled(400, 250)
        self.display.setPixmap(pixmap)

    def display_debug_img(self, image):
        img = image.copy()
        h, w, c = img.shape
        self.calcReport.setText("")
        if not (self.model_path == ''):
            if (self.radio1.isChecked()):
                ''
            elif (self.radio2.isChecked()):
                ''
            elif (self.radio3.isChecked()):
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                result = segformer.get_result_seg(img, self.model, self.processor, self.device)
                img = cv2.resize(img, (256, 256))
                if result.shape[2] == 4:
                    result = result[:, :, :3]
                r_iou = baseiou.getIOU_np(img, result)
                r_dice = baseiou.getDiceCoefficient(img, result)
                self.calcReport.setText('IOU: ' + str(r_iou) + ', DiceCoefficient: ' + str(r_dice))
                img = result
            elif (self.radio4.isChecked()):
                names = []
                with open(self.data_path + '/data.yaml') as f:
                    film = yaml.load(f, Loader = yaml.FullLoader)
                    names = film['names']
                f = open(self.data_path + "/labels/" +
                         self.indexes[2][self.indexes[0] - 1][:self.indexes[2][self.indexes[0] - 1].rfind(".")] + ".txt", 'r')
                tmp = f.readlines()
                f.close()
                ref = []
                for t in tmp:
                    tt = t.split(" ")
                    ref.append([int(tt[0]), float(tt[1]), float(tt[2]), float(tt[3]), float(tt[4])])
                pre = yolo.get_result_yolo(img, self.model, names)
                #####
                pre = yolo.driver_processing(pre)
                ref = yolo.roi_processing(ref, self.roi)
                pre = yolo.roi_processing(pre, self.roi)
                #####
                ############################# DISPLAY #############################
                for r in ref:
                    c_x, c_y = (int(r[1] * w), int(r[2] * h))
                    c_w, c_h = (int(r[3] * w), int(r[4] * h))
                    c_x1, c_x2 = (int(c_x - c_w / 2), int(c_x + c_w / 2))
                    c_y1, c_y2 = (int(c_y - c_h / 2), int(c_y + c_h / 2))
                    cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
                    cv2.putText(img, str(r[0]), (c_x1 + 10, c_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                for p in pre:
                    c_x, c_y = (int(p[1] * w), int(p[2] * h))
                    c_w, c_h = (int(p[3] * w), int(p[4] * h))
                    c_x1, c_x2 = (int(c_x - c_w / 2), int(c_x + c_w / 2))
                    c_y1, c_y2 = (int(c_y - c_h / 2), int(c_y + c_h / 2))
                    cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), (0, 0, 255), 2)
                    cv2.putText(img, str(p[0]), (c_x2 - 30, c_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, str(p[5]), (c_x2 - 100, c_y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                ####################################################################
                r_p = round(baseiou.getPrecision(ref, pre, self.thres), 2)
                r_r = round(baseiou.getRecall(ref, pre, self.thres), 2)
                r_acc = round(baseiou.getAccuracy(ref, pre, self.thres), 2)
                r_map = round(baseiou.getmAP(ref, pre, self.thres), 2)
                self.calcReport.setText('P: ' + str(r_p) + ', R: ' + str(r_r) + ', Acc: ' + str(r_acc) + ', mAP: ' + str(r_map))
                c_x, c_y = (int(self.roi[0] * w), int(self.roi[1] * h))
                c_w, c_h = (int(self.roi[2] * w), int(self.roi[3] * h))
                c_x1, c_x2 = (int(c_x - c_w / 2), int(c_x + c_w / 2))
                c_y1, c_y2 = (int(c_y - c_h / 2), int(c_y + c_h / 2))
                cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), (255, 0, 0), 3)
        h, w, c = img.shape
        qImg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        pixmap = pixmap.scaled(self.disp_size[0], self.disp_size[1])
        self.debug_display.setPixmap(pixmap)

    def excel(self):
        fname = QFileDialog.getExistingDirectory(self, '결과 출력 폴더 선택', '')
        self.excel_path = fname
        self.l_report.setText('Export Folder: ' + fname)

    def data(self):
        fname = QFileDialog.getExistingDirectory(self, '데이터 폴더 선택', '')
        self.data_path = fname
        self.l_data.setText('Data Folder: ' + fname)
        if os.path.exists(fname + '/images'):
            data = os.listdir(fname + '/images')
            self.indexes[2] = data
            if (len(data) > 0):
                self.indexes[0] = 1
                self.indexes[1] = len(data)
                img = cv2.imread(fname + '/images/' + data[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.curImage = img.copy()
                self.display_debug_img(img)
                self.updateIndex()
            self.updateList()
        else:
            self.showMessage("이미지 폴더가 발견되지 않았습니다.")

    def sel_model(self):
        if not (self.model == None):
            torch.cuda.empty_cache()
            del (self.model)
        if (self.radio1.isChecked() or self.radio2.isChecked() or self.radio3.isChecked()):
            fname = QFileDialog.getExistingDirectory(self, '모델 위치 선택', '')
        else:
            fname = QFileDialog.getOpenFileName(self, '', '', 'All File(*);; PyTorch(*.pt)')[0]

        if (fname):
            if (fname == ''):
                self.showMessage('올바른 모델을 선택해주세요.')
                return
            self.model_path = fname
            self.l_model.setText('Model: ' + fname)
            if (self.radio1.isChecked()):
                if (os.path.isdir(self.model_path + '/final_model.pth')):
                    self.showMessage('final_model.pth 파일이 감지되지 않았습니다.')
                    self.model_path = ''
                    return
                if (os.path.isdir(self.model_path + '/vocab.pkl')):
                    self.showMessage('vocab.pkl 파일이 감지되지 않았습니다.')
                    self.model_path = ''
                    return
                self.model = resnet.load_trained_model(self.model_path + "/final_model.pth", self.model_path + "/vocab.pkl")
            elif (self.radio2.isChecked()):
                self.processor = PaliGemmaProcessor.from_pretrained(self.model_path, local_files_only=True)
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.bfloat16,
                        local_files_only=True
                    ).to(self.device)
            elif (self.radio3.isChecked()):
                self.processor = SegformerImageProcessor.from_pretrained(self.model_path)
                self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_path, trust_remote_code = True)
                self.model.to(self.device)
            elif (self.radio4.isChecked()):
                self.model = YOLO(fname)

    def updateIndex(self):
        self.l_count.setText(str(self.indexes[0]) + "/" + str(self.indexes[1]))
        self.updateList()

    def updateList(self):
        self.table.setRowCount(self.indexes[1])
        self.indexes[3].clear()
        for i in range(self.indexes[1]):
            self.indexes[3].append(QTableWidgetItem(self.indexes[2][i].format()))
            self.table.setItem(i, 0, self.indexes[3][i])
        if (len(self.indexes[3]) > 0):
            self.table.setCurrentItem(self.indexes[3][self.indexes[0] - 1])

    def tableAction(self):
        idx = self.table.currentItem().row()
        self.indexes[0] = idx + 1
        img = cv2.imread(self.data_path + '/images/' +
                            self.indexes[2][self.indexes[0] - 1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.curImage = img.copy()
        self.display_debug_img(img)
        self.updateIndex()

    def left(self):
        if (self.indexes[1] > 0):
            if (self.indexes[0] > 0):
                if (self.indexes[0] == 1):
                    self.indexes[0] = self.indexes[1]
                else:
                    self.indexes[0] -= 1
                img = cv2.imread(self.data_path + '/images/' +
                                 self.indexes[2][self.indexes[0] - 1])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.curImage = img.copy()
                self.display_debug_img(img)
                self.updateIndex()
    
    def right(self):
        if (self.indexes[1] > 0):
            if (self.indexes[0] < self.indexes[1] + 1):
                if (self.indexes[0] == self.indexes[1]):
                    self.indexes[0] = 1
                else:
                    self.indexes[0] += 1
                img = cv2.imread(self.data_path + '/images/' +
                                 self.indexes[2][self.indexes[0] - 1])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.curImage = img.copy()
                self.display_debug_img(img)
                self.updateIndex()

    def showMessage(self, string):
        QMessageBox.information(self, 'Message', string)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()