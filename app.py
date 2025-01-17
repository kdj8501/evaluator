import sys, re
from collections import Counter
import torch
import torch.nn as nn
import pickle
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PIL import Image
import evaluate
import xlsxwriter

##################### predefine for ResNet & LSTM #####################
EMBED_DIM = 256
HIDDEN_DIM = 512
MAX_SEQ_LENGTH = 25
print("CUDA Available: " + str(torch.cuda.is_available()))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################### class for ResNet & LSTM #######################
class Vocabulary:
    def __init__(self, freq_threshold = 5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "pad", 1: "startofseq", 2: "endofseq", 3: "unk"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.index = 4

    def __len__(self):
        return len(self.itos)
    
    def tokenizer(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.index
                self.itos[self.index] = word
                self.index += 1

    def numericalize(self, text):
        tokens = self.tokenizer(text)
        numericalized = []
        for token in tokens:
            if token in self.stoi:
                numericalized.append(self.stoi[token])
            else:
                numericalized.append(self.stoi["<unk>"])
        return numericalized
    
########## Load Vocabulary ##########
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
# print(vocab)
vocab_size = len(vocab)
# print(vocab_size)
    
class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet50(weights = ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = True
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim, momentum = 0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.batch_norm(features)
        return features
    
class DecoderLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions, states):
        captions_in = captions
        emb = self.embedding(captions_in)
        features = features.unsqueeze(1)
        lstm_input = torch.cat((features, emb), dim = 1)
        outputs, returned_states = self.lstm(lstm_input, states)
        logits = self.fc(outputs)
        return logits, returned_states
    
    def generate(self, features, max_len = 20):
        batch_size = features.size(0)
        states = None
        generated_captions = []

        start_idx = 1
        end_idx = 2
        current_tokens = [start_idx]

        for _ in range(max_len):
            input_tokens = torch.LongTensor(current_tokens).to(features.device).unsqueeze(0)
            logits, states = self.forward(features, input_tokens, states)
            logits = logits.contiguous().view(-1, vocab_size)
            predicted = logits.argmax(dim = 1)[-1].item()

            generated_captions.append(predicted)
            current_tokens.append(predicted)

        return generated_captions
    
class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def generate(self, images, max_len = MAX_SEQ_LENGTH):
        features = self.encoder(images)
        return self.decoder.generate(features, max_len = max_len)

###################### function for load trained model ######################
def load_trained_model(path):
    encoder = ResNetEncoder(embed_dim = EMBED_DIM)
    decoder = DecoderLSTM(EMBED_DIM, HIDDEN_DIM, vocab_size)
    model = ImageCaptioningModel(encoder, decoder).to(DEVICE)

    state_dict = torch.load(path, map_location = DEVICE, weights_only = True)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    return model

####################### function for generate result #######################
transform_inference = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]),
    ]
)

def generate_caption_for_image(img, model):
    pil_img = img.convert("RGB")
    img_tensor = transform_inference(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_indices = model.generate(img_tensor, max_len = MAX_SEQ_LENGTH)

    result_words = []
    end_token_idx = vocab.stoi["endofseq"]
    for idx in output_indices:
        if idx == end_token_idx:
            break
        word = vocab.itos.get(idx, "unk")
        if word not in ["startofseq", "pad", "endofseq"]:
            result_words.append(word)
    return " ".join(result_words)

################################## Thread for testing ##################################
class Worker(QThread):
    sig = pyqtSignal(int)
    rep = pyqtSignal(list, list)

    def __init__(self, lines, model, path):
        super().__init__()
        self.model = model
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
            img = Image.open(self.path + "/Images/" + l.split("\n")[0].split(",")[0])
            answer = l.split("\n")[0].split(",")[1]
            generate = generate_caption_for_image(img, self.model)
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
            model = load_trained_model(self.__class__.model_path)
            path = self.__class__.data_path

            test = lines[:100]

            self.worker = Worker(test, model, path)
            self.worker.start()
            self.worker.sig.connect(self.process)
            self.worker.rep.connect(self.report)

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
        fname, check = QFileDialog.getOpenFileName(self, '모델 선택', '')
        self.__class__.model_path = fname

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()