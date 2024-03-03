label_set = set()
f = open("train.txt", mode="r", encoding="utf-8")
for line in f:
    label_set.add(line.split(":")[0])
f = open("test.txt", mode="r", encoding="utf-8")
for line in f:
    label_set.add(line.split(":")[0])

f = open("label.data", mode="w", encoding="utf-8")
for label in sorted(label_set):
    f.write(label + "\n")
