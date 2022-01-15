import sys
from collections import Counter, defaultdict

from hmm import HMM

start_tag = "START"
#end_tag = "$END"

def main():
    if len(sys.argv) == 0:
        print("Keine Datei angegeben als Argument")
        return

    try:
        fd = open(sys.argv[1], 'r')
    except OSError:
        print("Can not read file")
        sys.exit()

    lines = []
    bi_tuples = []
    tags = []

    with fd:
        tag_before = start_tag

        for line in fd:
            if line != "\n":
                if tag_before == start_tag:
                    tags.append(start_tag)

                line = line.strip()
                split = line.split("\t")

                word = split[0]
                tag = split[1]

                tags.append(tag)

                bi_tuples.append((tag_before, tag))
                tag_before = tag

                lines.append(line.strip())
            else:
                #tags.append(end_tag)
                #bi_tuples.append((tag_before, end_tag))
                tag_before = start_tag

    cnt = Counter(lines)
    emission = defaultdict(dict)
    uni_gram_tags = Counter(tags)

    total = cnt.total()

    for word_and_tag in cnt:
        split = word_and_tag.split("\t")
        word = split[0]
        tag = split[1]

        emission[tag][word] = cnt[word_and_tag] / total

    all = 0
    for key in emission:
        all += len(emission[key].values())

    bi_tuples = Counter(bi_tuples)

    trans = defaultdict(dict)

    for bi_tuple in bi_tuples:
        w1 = bi_tuple[0]
        w2 = bi_tuple[1]

        trans[w1][w2] = bi_tuples[bi_tuple] / uni_gram_tags[w1]

    #print(emission)

    h = HMM(trans, emission)
    print(h.decode(["Der", "Baum", "ist", "groß", "."]))

def eval():
    eval_fd = open("tiger22/tiger22-eval.tsv", 'r')

    with eval_fd:
        eval_lines = eval_fd.readlines()

    compare_fd = open("tiger22/tiger22-eval_false.tsv", 'r')

    with compare_fd:
        compare_lines = compare_fd.readlines()

    eval_count = 0
    equals_count = 0

    for i in range(0, len(eval_lines)):
        if i >= len(compare_lines):
            eval_count += 1
        elif eval_lines[i] == compare_lines[i]:
            eval_count = eval_count + 1
            equals_count = equals_count + 1
        else:
            eval_count = eval_count + 1

    print((equals_count / eval_count) * 100)

if __name__ == "__main__":
    #eval()
    main()