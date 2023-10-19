
import re
import json

INSTANCE_POSTFIX = '_instances.json'
TEXT_POSTFIX = '_text.txt'


def is_dir(x):
    return '.' not in x


def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fin:
        json_obj = json.load(fin)
    return json_obj


def load_texts(text_file):
    with open(text_file, 'r', encoding='utf-8') as fin:
        text_lines = [re.sub(r'\s', '', line) for line in fin.readlines()]
    return text_lines
