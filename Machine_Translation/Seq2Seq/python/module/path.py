import os

train_path = "../data/train"
train_en_sgm = os.path.join(train_path, "train.en-zh.en.sgm")
train_zh_sgm = os.path.join(train_path, "train.en-zh.zh.sgm")
train_en = os.path.join(train_path, "train.en")
train_zh = os.path.join(train_path, "train.zh")

valid_path = "../data/valid"
valid_en_sgm = os.path.join(valid_path, "valid.en-zh.en.sgm")
valid_zh_sgm = os.path.join(valid_path, "valid.en-zh.zh.sgm")
valid_en = os.path.join(valid_path, "valid.en")
valid_zh = os.path.join(valid_path, "valid.zh")

test_path = "../data/test"
test_en_sgm = os.path.join(test_path, "test.en-zh.en.sgm")
test_en = os.path.join(test_path, "test.en")

dict_path = "../data/dict"
en_dict = os.path.join(dict_path, "english.dict")
zh_dict = os.path.join(dict_path, "chinese.dict")
