import argparse
import io
parser = argparse.ArgumentParser(description='classifier')
parser.add_argument('--test', type=str, default=False, help='test')

args = parser.parse_args()

if __name__ == '__main__':
    text_file_list = ['./Data/austen_utf8.txt',
                      './Data/dickens_utf8.txt',
                      './Data/tolstoy_utf8.txt',
                      './Data/wilde_utf8.txt']
    all_sentences = []
    for file in text_file_list:
        file_sentences = []
        f = io.open(file, encoding='utf-8')
        for line in f:
            file_sentences.append(line)
        all_sentences.append(file_sentences)