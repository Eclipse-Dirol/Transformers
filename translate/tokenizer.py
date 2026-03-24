from transformers import MarianTokenizer as mt
import config

class Tokenizer():
    def __init__(self):
        self.tokenizer = mt.from_pretrained('Helsinki-NLP/opus-mt-en-ru')
        self.src_lang = config.src_lang
        self.tgt_lang = config.tgt_lang
        self.max_len = config.max_len
        self.padding = config.padding

    def encode(self, text: str):
        return self.tokenizer(
            text,
            return_tensor='pt',
            padding=self.padding,
            max_length=self.max_len
            )

    def decode(self, data):
        return self.tokenizer.decode(data)