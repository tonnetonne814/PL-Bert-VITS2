from PL_BERT_ja.text import cmudict, pinyin
import pyopenjtalk
import PL_BERT_ja.phonemize

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]
# _japanese = ['ky','sp', 'sh', 'ch', 'ts','ty', 'ry', 'ny', 'by', 'hy', 'gy', 'kw', 'gw', 'kj', 'gj', 'my', 'py','dy']
japanese = ['$', '%', '&', '「', '」', '=', '~', '^', '|', '[', ']', '{', '}', '*', '+', '#', '<', '>']
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
    + japanese
)

symbol_to_id = {s: i for i, s in enumerate(symbols)}

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = symbol_to_id
    def __call__(self, text):
        indexes = []
        japanese = False
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except:
                if char == "。" or char == "、":
                    indexes.append(0) # padとして扱う

        return indexes


if __name__ == '__main__':
    print(pyopenjtalk.g2p("こんにちは。"))
    print(symbols)
    cleaner = TextCleaner()
    