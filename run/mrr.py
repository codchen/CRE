def get_mrr(filename, topk=3):
    preds = []
    truths = []
    with open('result_{}_top{}_0_0.csv'.format(filename, topk), 'r') as file:
        lines = [line for line in file]
        t1 = []
        t2 = []
        for line in lines:
            f, s, _ = line.split(',', 2)
            t1.append(float(f))
            t2.append(float(s))
        for i in range(0, len(lines), topk):
            preds.append(t1[i:i+topk])
            truths.append(t2[i:i+topk])
    total = 0
    for i in range(len(preds)):
        for j in range(topk):
            if truths[i][j] > 0:
                total += 1.0 / (j + 1)
                break
    return total / len(truths)

print(get_mrr('lstm_complex'))
for name in [
    "weston",
    "han",
    "cnn_transe",
    "cnn_complex",
    "lstm_transe",
    "lstm_complex",
    "transformer_transe",
    "transformer_complex"
]:
    print("{}: {}, {}".format(name, get_mrr(name), get_mrr(name, 5)))