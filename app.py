import sys, os
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PIL import Image
import evaluate
import xlsxwriter
import paligemma
import resnet
import yolo, baseiou
from resnet import Vocabulary
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from ultralytics import YOLO
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################## Thread for testing ##################################
class WorkerForResnet(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(list, list, int)

    def __init__(self, lines, model, path, vocab):
        super().__init__()
        self.model = model
        self.lines = lines
        self.path = path
        self.vocab = vocab
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.flag = False

    def thread_stop(self):
        self.flag = True

    def run(self):
        report = []
        idx = 0
        total = len(self.lines)
        sum_bleu_score = 0.0
        sum_rouge_score = 0.0
        sum_meteor_score = 0.0
        
        for l in self.lines:
            if self.flag == True:
                return
            img = Image.open(self.path + "/Images/" + l.split("\n")[0].split(",")[0])
            answer = l.split("\n")[0].split(",")[1]
            generate = resnet.generate_caption_for_image(img, self.model, self.vocab)
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
        self.rep.emit(report, total_avr, 0)
        print("Total Average Score = b: " + str(total_avr[0]) + " r: " + str(total_avr[1]) + " m: " + str(total_avr[2]))
        del self.model
        torch.cuda.empty_cache()

class WorkerForPali(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(list, list, int)

    def __init__(self, lines, model, processor, path):
        super().__init__()
        self.model = model
        self.processor = processor
        self.lines = lines
        self.path = path
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.flag = False

    def thread_stop(self):
        self.flag = True

    def run(self):
        report = []
        idx = 0
        total = len(self.lines)
        sum_bleu_score = 0.0
        sum_rouge_score = 0.0
        sum_meteor_score = 0.0
        
        for l in self.lines:
            if self.flag == True:
                return
            img = self.path + "/Images/" + l.split("\n")[0].split(",")[0]
            answer = l.split("\n")[0].split(",")[1]
            generate = paligemma.get_result_pali(img, "describe en", self.model, self.processor)
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
        self.rep.emit(report, total_avr, 0)
        print("Total Average Score = b: " + str(total_avr[0]) + " r: " + str(total_avr[1]) + " m: " + str(total_avr[2]))
        del self.model
        torch.cuda.empty_cache()

class WorkerForYolo(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(list, list, int)

    def __init__(self, model, path, thres):
        super().__init__()
        self.model = model
        self.path = path
        self.flag = False
        self.thres = thres

    def thread_stop(self):
        self.flag = True

    def run(self):
        report = []
        idx = 0
        sum_accuracy = 0.0
        sum_precision = 0.0
        sum_recall = 0.0
        sum_mAP = 0.0
        data = os.listdir(self.path + "/images")
        for i in range(len(data)):
            data[i] = data[i][:data[i].rfind(".")]
        total = len(data)

        for d in data:
            if (self.flag == True):
                return
            idx += 1
            f = open(self.path + "/labels/" + d + ".txt", 'r')
            tmp = f.readlines()
            f.close()
            ref = []
            for t in tmp:
                tt = t.split(" ")
                ref.append([int(tt[0]), float(tt[1]), float(tt[2]), float(tt[3]), float(tt[4])])
            pre = yolo.get_result_yolo(self.path + "/images/" + d + ".jpg", self.model)
            precision = baseiou.getPrecision(ref, pre, self.thres)
            recall = baseiou.getRecall(ref, pre, self.thres)
            acc = baseiou.getAccuracy(ref, pre, self.thres)
            mAP = baseiou.getmAP(ref, pre, self.thres)
            sum_accuracy += acc
            sum_precision += precision
            sum_recall += recall
            sum_mAP += mAP
            # print([pre, ref])
            report.append([d, precision, recall, acc, mAP])
            print("[" + str(idx) + "/" + str(total) + "] img_name: " + d + ", [P, R, A, mAP]: " + str([precision, recall, acc, mAP]))
            self.sig.emit(int(idx / total * 100))

        total_avr_acc = sum_accuracy / total
        total_avr_prc = sum_precision / total
        total_avr_rec = sum_recall / total
        total_avr_map = sum_mAP / total

        self.rep.emit(report, [total_avr_prc, total_avr_rec, total_avr_acc, total_avr_map], 1)
        print("Total Average Score = " + str([total_avr_prc, total_avr_rec, total_avr_acc, total_avr_map]))
        del self.model
        torch.cuda.empty_cache()

################################## GUI ##################################
class MainWindow(QMainWindow):
    excel_path = ""
    data_path = ""
    model_path = ""
    thres = 0.0
    sel = 0 # 0: ResNet & LSTM, 1: PaliGemma2, 2: YOLO v11x

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Model Evaluator v1.1")
        self.setGeometry(300, 300, 400, 160)
        self.setFixedSize(400, 160)

        self.pBar = QProgressBar(self)
        self.pBar.move(20, 20)
        self.pBar.resize(370, 20)
        self.pBar.setValue(0)

        self.label = QLabel("Threshold(%)", self)
        self.label.move(80, 55)
        self.text = QLineEdit("50", self)
        self.text.setFixedSize(30, 20)
        self.text.move(160, 60)

        self.testBtn = QPushButton("테스트 시작", self)
        self.testBtn.move(220, 55)
        self.testBtn.clicked.connect(self.test)

        self.radio1 = QRadioButton("ResNet + LSTM", self)
        self.radio1.move(20, 95)
        self.radio1.setFixedSize(150, 20)
        self.radio2 = QRadioButton("PaliGemma", self)
        self.radio2.move(160, 90)
        self.radio3 = QRadioButton("YOLO v11x", self)
        self.radio3.move(280, 90)
        self.radio3.setChecked(True)

        self.excelBtn = QPushButton("결과 출력 위치", self)
        self.excelBtn.move(10, 120)
        self.excelBtn.clicked.connect(self.excel)
        self.folderBtn = QPushButton("데이터 폴더", self)
        self.folderBtn.move(150, 120)
        self.folderBtn.clicked.connect(self.data)
        self.selBtn = QPushButton("모델 위치", self)
        self.selBtn.move(290, 120)
        self.selBtn.clicked.connect(self.model)

    def test(self):
        self.__class__.thres = int(self.text.text()) / 100
        if (self.__class__.data_path == "" or self.__class__.model_path == "" or self.__class__.excel_path == ""):
            return
        if (self.testBtn.text() == "테스트 중지"):
            self.worker.thread_stop()
            self.pBar.setValue(0)
            self.testBtn.setText("테스트 시작")
        elif (self.testBtn.text() == "테스트 시작"):
            if (self.radio1.isChecked()):
                self.__class__.sel = 0
            elif (self.radio2.isChecked()):
                self.__class__.sel = 1
            elif (self.radio3.isChecked()):
                self.__class__.sel = 2

            self.testBtn.setText("테스트 중지")

            if (self.sel == 0):
                f = open(self.__class__.data_path + "/captions.txt", 'r')
                f.readline()
                lines = f.readlines()
                f.close()
                test = lines[:100]
                self.model = resnet.load_trained_model(self.__class__.model_path + "/final_model.pth", self.__class__.model_path + "/vocab.pkl")
                self.worker = WorkerForResnet(test, self.model, self.__class__.data_path, self.__class__.model_path + "/vocab.pkl")
                self.worker.start()
                self.worker.sig.connect(self.process)
                self.worker.rep.connect(self.report)
                del self.model
                torch.cuda.empty_cache()
            elif (self.sel == 1):
                f = open(self.__class__.data_path + "/captions.txt", 'r')
                f.readline()
                lines = f.readlines()
                f.close()
                test = lines[:100]
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.__class__.model_path,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True
                ).to(DEVICE)
                self.processor = PaliGemmaProcessor.from_pretrained(self.__class__.model_path, local_files_only=True)
                self.worker = WorkerForPali(test, self.model, self.processor, self.__class__.data_path)
                self.worker.start()
                self.worker.sig.connect(self.process)
                self.worker.rep.connect(self.report)
                del self.model
                torch.cuda.empty_cache()
            elif (self.sel == 2):
                self.model = YOLO('yolo11x.pt')
                self.worker = WorkerForYolo(self.model, self.__class__.data_path, self.__class__.thres)
                self.worker.start()
                self.worker.sig.connect(self.process)
                self.worker.rep.connect(self.report)
                del self.model
                torch.cuda.empty_cache()

    def report(self, rep, avr, div):
        if (div == 0):
            date = time.strftime('%Y_%m_%d_%H_%M_%S')
            workbook = xlsxwriter.Workbook(self.__class__.excel_path + '/' + date + '_report.xlsx')
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
        elif (div == 1):
            date = time.strftime('%Y_%m_%d_%H_%M_%S')
            workbook = xlsxwriter.Workbook(self.__class__.excel_path + '/' + date + '_report.xlsx')
            worksheet = workbook.add_worksheet()
            worksheet.write(0, 0, "image_name")
            worksheet.write(0, 1, "precision")
            worksheet.write(0, 2, "recall")
            worksheet.write(0, 3, "accuracy")
            worksheet.write(0, 4, "mAP." + str(int(round(self.__class__.thres, 2) * 100)))
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
            workbook.close()

    def process(self, num):
        self.pBar.setValue(num)
        if (num == 100):
            self.testBtn.setText("테스트 시작")

    def excel(self):
        fname = QFileDialog.getExistingDirectory(self, '결과 출력 폴더 선택', '')
        self.__class__.excel_path = fname

    def data(self):
        fname = QFileDialog.getExistingDirectory(self, '데이터 폴더 선택', '')
        self.__class__.data_path = fname

    def model(self):
        fname = QFileDialog.getExistingDirectory(self, '모델 위치 선택', '')
        self.__class__.model_path = fname

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()