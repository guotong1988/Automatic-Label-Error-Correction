import random

for tmp in ["test", "train"]:
    predict_labels = []
    for one in open("predict_" + tmp + "data.txt", encoding="utf-8", mode="r"):
        predict_labels.append(one.strip())

    origin_labels = []
    input_list = []
    for one in open("data/" + tmp + ".txt", encoding="utf-8", mode="r"):
        origin_labels.append(one.strip().split(":")[0])
        input_list.append(one.strip().split(":")[1])

    seed_count = 100000
    for index in range(seed_count):
        f_rand = open("data/explore_data/" + tmp + str(index) + ".txt", encoding="utf-8", mode="w")
        tmp_count = 0
        for i in range(len(predict_labels)):
            if predict_labels[i] != origin_labels[i]:
                if random.randint(1, 2) == 1:
                    tmp_count += 1
                    f_rand.write(predict_labels[i] + ":" + input_list[i] + "\n")
                else:
                    f_rand.write(origin_labels[i] + ":" + input_list[i] + "\n")
            else:
                f_rand.write(origin_labels[i] + ":" + input_list[i] + "\n")
        print(tmp_count)
    print()

