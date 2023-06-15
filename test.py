import pickle

EVAL_DATA_PATH = '../pickle/tinyimagenet/resnet34/eval.p'

with open(EVAL_DATA_PATH, "rb") as f:
    eval_dataset = pickle.load(f)

for i in range(len(eval_dataset)):
    if eval_dataset[i][2] == 1:
        print(eval_dataset[i][1])
    else: 
        pass
