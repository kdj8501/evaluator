import sys
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PIL import Image
import evaluate
import xlsxwriter
import paligemma
import resnet
from resnet import Vocabulary
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################## Thread for testing ##################################
class WorkerForResnet(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(list, list)

    def __init__(self, lines, model, path, vocab):
        super().__init__()
        self.model = model
        self.lines = lines
        self.path = path
        self.vocab = vocab
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")

    def run(self):
        report = []
        idx = 0
        total = len(self.lines)
        sum_bleu_score = 0.0
        sum_rouge_score = 0.0
        sum_meteor_score = 0.0
        
        for l in self.lines:
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

            report.append(l.split("\n")[0].split(",")[0] + "," + answer + "," + generate + "," + str(bleu) + "," + str(rouge) + "," + str(meteor))
            print("[" + str(idx) + "/" + str(total) + "] " + answer + ", " + generate + " | scores b: " + str(bleu) + " r: " + str(rouge) + " m: " + str(meteor))
            self.sig.emit(int(idx / total * 100))

        total_avr = [sum_bleu_score / total, sum_rouge_score / total, sum_meteor_score / total]
        self.rep.emit(report, total_avr)
        print("Total Average Score = b: " + str(total_avr[0]) + " r: " + str(total_avr[1]) + " m: " + str(total_avr[2]))
        del self.model
        torch.cuda.empty_cache()

class WorkerForPali(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(list, list)

    def __init__(self, lines, model, processor, path):
        super().__init__()
        self.model = model
        self.processor = processor
        self.lines = lines
        self.path = path
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")

    def run(self):
        report = []
        idx = 0
        total = len(self.lines)
        sum_bleu_score = 0.0
        sum_rouge_score = 0.0
        sum_meteor_score = 0.0
        
        for l in self.lines:
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

            report.append(l.split("\n")[0].split(",")[0] + "," + answer + "," + generate + "," + str(bleu) + "," + str(rouge) + "," + str(meteor))
            print("[" + str(idx) + "/" + str(total) + "] " + answer + ", " + generate + " | scores b: " + str(bleu) + " r: " + str(rouge) + " m: " + str(meteor))
            self.sig.emit(int(idx / total * 100))

        total_avr = [sum_bleu_score / total, sum_rouge_score / total, sum_meteor_score / total]
        self.rep.emit(report, total_avr)
        print("Total Average Score = b: " + str(total_avr[0]) + " r: " + str(total_avr[1]) + " m: " + str(total_avr[2]))
        del self.model
        torch.cuda.empty_cache()

################################## GUI ##################################
class MainWindow(QMainWindow):
    excel_path = ""
    data_path = ""
    model_path = ""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Model Evaluator v1.0")
        self.setGeometry(300, 300, 400, 180)
        self.setFixedSize(400, 180)

        self.pBar = QProgressBar(self)
        self.pBar.move(20, 20)
        self.pBar.resize(370, 20)
        self.pBar.setValue(0)
        self.testBtn = QPushButton("테스트 시작", self)
        self.testBtn.move(150, 55)
        self.testBtn.clicked.connect(self.test)

        self.label = QLabel("Threshold(%)", self)
        self.label.move(150, 90)
        self.text = QLineEdit("60", self)
        self.text.setFixedSize(30, 20)
        self.text.move(230, 95)

        self.excelBtn = QPushButton("결과 출력 위치", self)
        self.excelBtn.move(10, 140)
        self.excelBtn.clicked.connect(self.excel)
        self.folderBtn = QPushButton("데이터 폴더", self)
        self.folderBtn.move(150, 140)
        self.folderBtn.clicked.connect(self.data)
        self.selBtn = QPushButton("모델 선택", self)
        self.selBtn.move(290, 140)
        self.selBtn.clicked.connect(self.model)

    def test(self):
        if (self.__class__.data_path == "" or self.__class__.model_path == "" or self.__class__.excel_path == ""):
            return
        if (self.testBtn.text() == "테스트 시작"):
            self.testBtn.setText("테스트 중지")
            f = open(self.__class__.data_path + "/captions.txt", 'r')
            f.readline()
            lines = f.readlines()
            f.close()
            test = lines[:100]

            sel = 0 # 0: ResNet & LSTM, 1: PaliGemma2
            if (sel == 0):
                self.model = resnet.load_trained_model(self.__class__.model_path + "/final_model.pth", self.__class__.model_path + "/vocab.pkl")
                self.worker = WorkerForResnet(test, self.model, self.__class__.data_path, self.__class__.model_path + "/vocab.pkl")
                self.worker.start()
                self.worker.sig.connect(self.process)
                self.worker.rep.connect(self.report)
                del self.model
                torch.cuda.empty_cache()
            elif (sel == 1):
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.__class__.model_path,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True
                ).to(DEVICE)
                self.processor = PaliGemmaProcessor.from_pretrained(self.__class__.model_path, local_files_only=True)
                self.worker2 = WorkerForPali(test, self.model, self.processor, self.__class__.data_path)
                self.worker2.start()
                self.worker2.sig.connect(self.process)
                self.worker2.rep.connect(self.report)
                del self.model
                torch.cuda.empty_cache()

    def report(self, rep, avr):
        workbook = xlsxwriter.Workbook(self.__class__.excel_path + '/report.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, "image_name")
        worksheet.write(0, 1, "reference")
        worksheet.write(0, 2, "prediction")
        worksheet.write(0, 3, "B")
        worksheet.write(0, 4, "R")
        worksheet.write(0, 5, "M")
        row = 1
        for r in rep:
            tmp = r.split(",")
            worksheet.write(row, 0, tmp[0])
            worksheet.write(row, 1, tmp[1])
            worksheet.write(row, 2, tmp[2])
            worksheet.write(row, 3, tmp[3])
            worksheet.write(row, 4, tmp[4])
            worksheet.write(row, 5, tmp[5])
            row += 1
        worksheet.write(row + 1, 0, "total average score")
        worksheet.write(row + 2, 0, "B")
        worksheet.write(row + 2, 1, "R")
        worksheet.write(row + 2, 2, "M")
        worksheet.write(row + 3, 0, str(avr[0]))
        worksheet.write(row + 3, 1, str(avr[1]))
        worksheet.write(row + 3, 2, str(avr[2]))
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
        fname = QFileDialog.getExistingDirectory(self, '모델 폴더 선택', '')
        self.__class__.model_path = fname

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()