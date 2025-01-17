import pickle
import torch
import torch.nn as nn
import re
from collections import Counter
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms

EMBED_DIM = 256
HIDDEN_DIM = 512
MAX_SEQ_LENGTH = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, embed_dim, hidden_dim, vocab, num_layers = 1):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, len(vocab))

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
            logits = logits.contiguous().view(-1, len(self.vocab))
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
def load_trained_model(path, vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    encoder = ResNetEncoder(embed_dim = EMBED_DIM)
    decoder = DecoderLSTM(EMBED_DIM, HIDDEN_DIM, vocab)
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

def generate_caption_for_image(img, model, vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
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