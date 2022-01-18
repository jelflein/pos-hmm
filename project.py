import math
import sys
import re
from collections import Counter, defaultdict
from functools import lru_cache
from difflib import unified_diff
from difflib import Differ

from hmm import HMM

start_tag = "START"

# Erkennt Zahlen mit Komma auch solche wie 200.000.000
number = re.compile("(-)?\d+((,|.)\d)?")

ovv3_start = re.compile("^\^.{1,3}")
ovv3_low = re.compile("^([a-z]|[ßäöü]){1,3}")
ovv3_capital = re.compile("^([A-Z]|[ÄÖÜ]){1}([a-z]|[ßäöü]){1,2}")

def validate_file_and_open(file_name: str, mode: str):
    try:
        fd = open(file_name, mode)
    except OSError:
        print("Can not read file")
        sys.exit(-1)

    return fd


def compute_trans_and_emission(train_fd):
    word_and_tags = {}
    bi_grams = {}
    uni_gram_tags = {}

    with train_fd:
        tag_before = start_tag

        for line in train_fd:
            if line != "\n":
                start = tag_before == start_tag
                if start:
                    uni_gram_tags[start_tag] += 1

                line = line.strip()
                split = line.split("\t")

                word = split[0]
                tag = split[1]
                uni_gram_tags[tag] += 1

                bi_grams[(tag_before, tag)] += 1
                tag_before = tag

                word_and_tags[(word, tag)] += 1
            else:
                tag_before = start_tag

    for word_and_tag in list(word_and_tags.keys()):
        if word_and_tags[word_and_tag] == 1:
            del word_and_tags[word_and_tag]
            word_and_tags[("OVV", word_and_tag[1])] += 1

    return compute_trans(bi_grams, uni_gram_tags), compute_emission(word_and_tags)


def classify_word(word, total):
    if number.fullmatch(word) is not None:
        return "NUMBER"

    if total > 1:
        return word

    if ovv3_capital.fullmatch(word) is not None:
        return "OVV3-CAPITAL"
    elif ovv3_low.fullmatch(word) is not None:
        return "OVV3-LOW"
    elif ovv3_start.fullmatch(word) is not None:
        return "OVV3-START"


def compute_emission(word_and_tags: dict) -> defaultdict:
    emission = defaultdict(dict)

    total = word_and_tags.total()

    for word_and_tag in word_and_tags:
        word = word_and_tag[0]
        tag = word_and_tag[1]

        emission[tag][word] = math.log(word_and_tags[word_and_tag] / total)

    return emission


def compute_trans(bi_tuples: dict, uni_gram_tags: dict) -> defaultdict:
    trans = defaultdict(dict)

    for bi_tuple in bi_tuples:
        w1 = bi_tuple[0]
        w2 = bi_tuple[1]

        trans[w1][w2] = math.log(bi_tuples[bi_tuple] / uni_gram_tags[w1])

    return trans


def save_nested_dict(input_dict: dict, output: str):
    try:
        fd = open(output, 'w')
    except OSError:
        print("Can not wirte to file")
        sys.exit(-1)

    with fd:
        for key_out in input_dict:
            for key_nested in input_dict[key_out]:
                prob = input_dict[key_out][key_nested]
                line_to_wirte = str(prob) + "\t" + key_out + "\t" + key_nested + "\n"

                fd.write(line_to_wirte)


def read_nested_dict(input_file_name: str):
    try:
        fd = open(input_file_name, 'r')
    except OSError:
        print("Can not read file")
        sys.exit(-1)

    outer_dict = defaultdict(dict)

    with fd:
        for line in fd:
            line = line.strip()
            split = line.split("\t")

            prob_uncasted = split[0]
            outer_key = split[1]
            nested_key = split[2]

            outer_dict[outer_key][nested_key] = float(prob_uncasted)

    return outer_dict


def tag_file(input_descriptor, output: str, trans: dict, emissions: dict):
    with input_descriptor:
        sentence = []
        sentences = []

        for line in input_descriptor:
            if line == "\n":
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(line.strip())

    hmm = HMM(trans, emissions)

    output_descriptor = validate_file_and_open(output, 'w')

    with output_descriptor:
        for sentence in sentences:
            tagged_sentence = hmm.decode(sentence)
            print(tagged_sentence)

            for word_tag in tagged_sentence[1]:
                output_descriptor.write(word_tag[0] + "\t" + word_tag[1] + "\n")

            output_descriptor.write("\n")


def eval_files(compare_descriptor, eval_descriptor, diff_results_descriptor):
    with compare_descriptor:
        compare = compare_descriptor.readlines()

    with eval_descriptor:
        eval = eval_descriptor.readlines()

    diffs = unified_diff(eval, compare, tofile=compare_descriptor.name, fromfile=eval_descriptor.name, lineterm="")
    with diff_results_descriptor:
        for line in diffs:
            diff_results_descriptor.write(line + "\n")

    correct = 0
    for i in range(0, len(eval)):
        if i >= len(compare):
            print("Die Vergleichsdatei hat weniger Zeilen als die Eval-Datei")
            break

        if eval[i] == compare[i]:
            correct += 1

    total_correct_line = len(eval)

    print("Akkuratheit: " + str((correct / total_correct_line) * 100) + "%")

    print("Total: " + str(total_correct_line))
    print("Korrekte Wörter: " + str(correct))


def print_help():
    print("Alle parameter sind Dateien!")
    print("\ttrain <training corpus> <bi gramm output> <emission output>")
    print("\ttag <bi gramm input> <emission input> <corpus> <tagged corpus out>")
    print("\teval <vergleichsdatei> <eval-datei> <diff output>")


if __name__ == "__main__":
    if (len(sys.argv) - 1) <= 1:
        print("Es wurden keine oder zu wenige Argumente übergeben!")
        print_help()
        sys.exit(-1)

    mode = sys.argv[1]

    if mode == "train":
        if (len(sys.argv) - 1) != 4:
            print("Die Anzahl der Argumente stimmt nicht!")
            sys.exit(-1)

        training_corpus_file_name = sys.argv[2]
        training_corpus_descriptor = validate_file_and_open(training_corpus_file_name, 'r')

        trans_and_emission = compute_trans_and_emission(training_corpus_descriptor)

        trans = trans_and_emission[0]
        emissions = trans_and_emission[1]

        bi_gram_out_file_name = sys.argv[3]
        emission_out_file_name = sys.argv[4]

        save_nested_dict(trans, bi_gram_out_file_name)
        save_nested_dict(emissions, emission_out_file_name)
    elif mode == "tag":
        if (len(sys.argv) - 1) != 5 and mode == "tag":
            print("Die Anzahl der Argumente stimmt nicht!")
            sys.exit(-1)

        bi_gramm_file_name = sys.argv[2]
        emission_file_name = sys.argv[3]

        bi_grams = read_nested_dict(bi_gramm_file_name)
        emissions = read_nested_dict(emission_file_name)

        open_input_descriptor = validate_file_and_open(sys.argv[4], 'r')
        tagged_out_file_name = sys.argv[5]

        tag_file(open_input_descriptor, tagged_out_file_name, bi_grams, emissions)
    elif mode == "eval":
        if (len(sys.argv) - 1) != 4:
            print("Die Anzahl der Argumente stimmt nicht!")
            sys.exit(-1)

        compare_descriptor = validate_file_and_open(sys.argv[2], 'r')
        eval_descriptor = validate_file_and_open(sys.argv[3], 'r')
        diff_results_descriptor = validate_file_and_open(sys.argv[4], 'w')

        eval_files(compare_descriptor, eval_descriptor, diff_results_descriptor)
    else:
        print("Kein bekannter Modus für das Programm")
        print_help()
        sys.exit(-1)
