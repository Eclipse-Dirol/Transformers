from tokenizer import Tokenizer
from transformer import Transformer

def main():
    token = Tokenizer()
    text = input('Write your text for translate Russia language\n')
    token_text = token.encode(text = text)
    model = Transformer(x = token_text)
    model.eval()
    output = model()
    output = output.argmax(dim=-1).squeeze(0).tolist()
    translate_text = token.decode(data = output)
    return translate_text

if __name__ == '__main__':
    answer = main()
    print(answer)