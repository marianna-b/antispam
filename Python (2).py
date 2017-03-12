from typing import *


class Message:
    def __init__(self):
        self.body = []
        self.header = []
        self.spam = False

    def read_from_file(self, file: str):
        with open(file) as f:
            header = f.readline()
            theme = header.split(":")[-1]
            self.header = list(map(int, theme.split()))
            f.readline()
            file = " ".join(f.readlines())
            self.body = [int(currentInt) for currentInt in file.split()]

    def __str__(self, *args, **kwargs):
        return str(self.spam) + "\n" + str(self.header) + "\n" + str(self.body)


def load_data(data_root: str, folder_count: int) -> Tuple[List[Message], List[List[Message]]]:
    import os
    message_list = [[] for i in range(folder_count)]
    messages = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            message = Message()
            messages.append(message)
            message.spam = "legit" in file
            message.read_from_file(root + "/" + file)
            cnt = int(root[-1])  # Never do such things
            message_list[cnt].append(message)
    return messages, message_list


msgs, msgsList = load_data("Bayes/pu1", 10)


class Classifier:
    def __init__(self, p_h: Dict[str, float], p_s: Dict[str, float], header_weight=2):
        self.p_h = p_h
        self.p_s = p_s
        self.h = 0  # type: float
        self.header_weight = header_weight

    def classify(self, msg: Message) -> bool:
        import math
        spam_n = 1
        ham_n = 1
        some_sum = 0

        for word in msg.body + msg.header * 2:
            if word not in self.p_s or word not in self.p_h:
                continue
            some_sum += math.log(self.p_s[word] / self.p_h[word])

        return some_sum <= self.h


def train_classifier(messages: List[Message], header_weight: int = 2) -> Classifier:
    header_weight = 1
    spam = []
    ham = []
    for msg in messages:
        if msg.spam:
            spam.append(msg)
        else:
            ham.append(msg)

    def calc_word_property(msgs):
        amount = 0
        in_set = {}
        for msg in msgs:
            amount += len(msg.body) + header_weight * len(msg.header)
            for word in msg.body:
                in_set[word] = in_set.get(word, 0) + 1
            for word in msg.header:
                in_set[word] = in_set.get(word, 0) + header_weight
        return in_set, amount

    in_ham, ham_amount = calc_word_property(ham)
    in_spam, spam_amount = calc_word_property(spam)

    p_h = {}
    for k, v in in_ham.items():
        p_h[k] = v / ham_amount

    p_s = {}
    for k, v in in_spam.items():
        p_s[k] = v / spam_amount

    return Classifier(p_s, p_h, header_weight=header_weight)


def cross_validation(array_of_message_lists: List[List[Message]]):
    for idx in range(len(array_of_message_lists)):
        messages = []
        for idx2 in range(len(array_of_message_lists)):
            if idx != idx2:
                messages.extend(array_of_message_lists[idx2])
        classifier = train_classifier(messages)
        f_p, f_n, t_n, t_p = [0] * 4

        for msg in array_of_message_lists[idx]:
            ans = classifier.classify(msg)
            if ans:
                if msg.spam:
                    t_p += 1
                else:
                    f_p += 1
            else:
                if msg.spam:
                    f_n += 1
                else:
                    t_n += 1

        score = 2 * t_p / (2 * t_p + f_n + f_p)
        print(t_p, f_p, t_n, f_n)
        print(score, f_p)


cross_validation(msgsList)
