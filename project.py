import math
import sys
from collections import Counter, defaultdict
from difflib import unified_diff

from classify import classify_word, classifyers, blocking_list
from hmm import HMM

start_tag = "START"


def validate_file_and_open(file_name: str, mode: str):
    """
    Diese Funktion öffnet einen File-Descriptor und fängt Fehler. Bei einem Fehler wird
    das Programm mit -1 beendet.

    :param file_name: Der Dateiname
    :param mode: der Modus, in welchem die Datei gelesen werden soll
    :return:
    """
    try:
        fd = open(file_name, mode)
    except OSError:
        print("Can not read file")
        sys.exit(-1)

    return fd


def compute_trans_and_emission(train_fd, denoising: bool = False) -> tuple:
    """
    Diese Funktion berechnet die Trans- und Emissionswahrscheinlichkeit für das HMM.

    Die Wörter werden verändert, wenn es an einem Satz Anfang steht wird ein '^' an den
    Anfang geheftet. Wenn es eine Zahl, Link oder ein geteiltes Wort ist, wird dies mit
    einem Hilfs-Tag ersetzt (NUMBER, LINK, TRUNC). Die entsprechenden regulären Ausdrücke
    findet man in classify.py

    Wenn ein Wort nur einmal im Korpus vorkommt, wird dies mithilfe von classify_word
    aus classify.py ein passendes Hilfswort ermittelt.

    Wenn denoising aktiviert ist, wird erzwungen, dass Zahlen, Links und geteilte Wörter,
    die Tags: CARD, XY und TRUNC zugewiesen bekommen.

    Die Zeilen werden mit .strip() bereinigt.

    Es wird erwartet dass das Datei-Formart eingehalten wird, dieses lautet:

    <Wahrscheinlichkeit>\t<Tag>\t<Wort>\n

    Sätze werden mit einer leeren Zeile getrennt.

    Der Rückgabe Wert ist ein Tupel aus nested dict (bi-gramme, emissions). Das Format ist
    jeweils sehr ähnlich:

    bi_gramme[wort 1][wort 2] = log Wahrscheinlichkeit
    emissions[tag][wort] = log Wahrscheinlichkeit

    siehe auch compute_emissoin und compute_trans für mehr Informationen.

    :param train_fd: ein File Descriptor des Trainingskorpus
    :param denoising: siehe 4 Absatz
    :return: ein Tupel aus nested dict (bi-gramme, emissions) siehe letzen Absatz
    """
    word_and_tags = defaultdict(int)
    bi_grams = defaultdict(int)
    uni_gram_tags = defaultdict(int)
    words = defaultdict(int)

    # einlesen des Korpus und ermittlung der Bi- und Uni-Gramme
    with train_fd:
        tag_before = start_tag

        for line in train_fd:
            if line != "\n":
                start = tag_before == start_tag

                line = line.strip()
                split = line.split("\t")

                word = split[0]
                tag = split[1]

                words[word] += 1

                uni_gram_tags[tag] += 1

                # Wenn Anfang ^
                if start:
                    word = "^" + word
                    uni_gram_tags[start_tag] += 1

                bi_grams[(tag_before, tag)] += 1
                tag_before = tag

                word_and_tags[(word, tag)] += 1
            else:
                tag_before = start_tag

    # Klassifizieren der Wörter mithilfe von RegEx (Zahlen, Links etc.)
    for word_and_tag in list(word_and_tags.keys()):
        word = word_and_tag[0]
        tag = word_and_tag[1]

        classified_word_and_tag = classify_word(word, words[word_and_tag[0]], tag, denoising)

        if classified_word_and_tag is not None:
            word = classified_word_and_tag[0]
            tag = classified_word_and_tag[1]

            word_and_tags[(word, tag)] += word_and_tags[word_and_tag] + 1
            del word_and_tags[word_and_tag]

    return compute_trans(bi_grams, uni_gram_tags), compute_emission(word_and_tags)


def compute_emission(word_and_tags: dict) -> defaultdict:
    """
    Diese Function ermittelt mithilfe von relativen Wahrscheinlichkeiten,
    die Emissionen für das HMM.

    :param word_and_tags: Ein dict welches als Key ein Tupel im Format (wort, tag) hat,
    der Value ist das Vorkommen im Korpus.

    :return: Ein nested dict im Format emission[tag][word] = log Wahrscheinlichkeit
    """
    emission = defaultdict(dict)

    # Anzahl aller Wörter im Korpus
    total = sum(word_and_tags.values())

    for word_and_tag in word_and_tags:
        word = word_and_tag[0]
        tag = word_and_tag[1]

        emission[tag][word] = math.log(word_and_tags[word_and_tag] / total)

    return emission


def compute_trans(bi_tuples: dict, uni_gram_tags: dict) -> defaultdict:
    """
    Diese Funktion berechnet die Wahrscheinlichkeit der schon vorher ermittelten Bi-Gramme (tag 1, tag 2).

    :param bi_tuples: ein dict welches als Key ein Bi-Gramm Tupel der Tags hat (tag 1, tag 2),
    der Value ist die Anzahl wie oft dieses Bi-Gramm im Korpus vorkommt.

    :param uni_gram_tags: Anzahl wie oft das Tag im Korpus vorkommt im Format  uni_gram_tags[tag] = Anzahl im Korpus

    :return: bi_gramme[tag1][tag2] = log Wahrscheinlichkeit
    """
    trans = defaultdict(dict)

    for bi_tuple in bi_tuples:
        tag1 = bi_tuple[0]
        tag2 = bi_tuple[1]

        trans[tag1][tag2] = math.log(bi_tuples[bi_tuple] / uni_gram_tags[tag1])

    return trans


def save_nested_dict(input_dict: dict, output: str):
    """
    Speichert ein zweifach nested dict im folgenden Format:

    Die Leerzeichen dienen nur der Lesbarkeit.

    input_dict[key_1][nested_key] \t key_1 \t nested_key \n

    Die Datei wird überschrieben.

    Das Programm wird beendet mit -1, falls die Datei nicht geöffnet werden kann.

    Diese funktion ist das Gegenstück zur: read_nested_dict

    :param input_dict: ein zweifach verschachteltes Dict.
    :param output: ein Dateiname
    """
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
    """
    Liest ein zweifach nested dict im folgenden Format:

    Die Leerzeichen dienen nur der Lesbarkeit.

    input_dict[key_1][nested_key] \t key_1 \t nested_key \n

    Das Programm wird beendet mit -1, falls die Datei nicht geöffnet werden kann.

    Diese funktion ist das Gegenstück zur: save_nested_dict

    :param input_file_name: ein Dateiname
    :return: ein zweifach verschachteltes Dict.
    """
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


def smooth_emissions(emissions: dict):
    """
    Diese Function hat den Hintergrund, falls in classify.py mit classify_word ein Wort klassifiziert wird
    und diese Klasse ist nicht in den Emissionen (Hidden States) vom HMM enthalten, das immer ein Wert für
    jede Klasse aus classify vorhanden ist.

    Dafür sucht man sich den nächsten Wert raus. Man kann sich dies als Array vorstellen:

    OVV-3 OVV-3-LOW OVV-3-CAPITAL OVV3-START … OVV-9-START …OVV-HUGE-START

    Falls alle Werte der Klasse OVV-3-* bekannt sind außer die Klasse OVV-3-LOW,
    werden alle Werte von OVV3-* Klassen ermitellt und der Durchschschnitt ermittelt,
    falls alle Werte von OVV-3-* leer sein sollten, wird zur nächst höhren Klasse gegangen usw.
    Falls alles leer sein sollte, sind die Trainingsdaten nicht optimal gewählt und es sollt ein andere gewählt werden.

    Es werden bestimmte Klasse ausgeschlossen, da es dort keine unbekannte Wörter gibt wie z.B. Artikel usw.
    Diese stehen in classify.py in blocking_list.

    :param emissions: emissions[tag][wort] = log Wahrscheinlichkeit
    :return: smoothed emissons
    """
    for key in list(emissions.keys()):
        if key in blocking_list:
            continue

        nested_dict = emissions[key]

        ovv = [0, 0, 0, 0]
        div = [0, 0, 0, 0]

        #Zusammen zählen der Werte
        for i in range(0, len(classifyers)):
            classifyer = classifyers[i]

            if classifyer in nested_dict:
                if i < 4:
                    ovv[0] += nested_dict[classifyer]
                    div[0] += 1
                elif i < 8:
                    ovv[1] += nested_dict[classifyer]
                    div[1] += 1
                elif i < 12:
                    ovv[2] += nested_dict[classifyer]
                    div[2] += 1
                elif i < 16:
                    ovv[3] += nested_dict[classifyer]
                    div[3] += 1

        #durchschnitt berechen
        for i in range(0, len(ovv)):
            if div[i] != 0:
                ovv[i] = ovv[i] / div[i]

        #Falls die Klasse OVV-N-* immer noch 0
        #wird die nächst nähre gesucht
        for i in range(0, len(ovv)):
            j = i
            k = i

            while ovv[i] == 0:
                j += 1
                k -= 1

                if k >= 0:
                    ovv[i] = ovv[k]

                if j < len(ovv) and ovv[i] == 0:
                    ovv[i] = ovv[j]

        for i in range(0, len(classifyers)):
            classifyer = classifyers[i]

            if classifyer not in nested_dict:
                if i < 4:
                    nested_dict[classifyer] = ovv[0]
                elif i < 8:
                    nested_dict[classifyer] = ovv[1]
                elif i < 12:
                    nested_dict[classifyer] = ovv[2]
                elif i < 16:
                    nested_dict[classifyer] = ovv[3]

    return emissions


def tag_file(input_descriptor, output: str, trans: dict, emissions: dict):
    """
    Die Methode liest die zu taggende Datei ein und ermittelt mithilfe des HMM die wahrscheinlichsten Tags.
    Das Format der zu taggende Datei ist:

    Wort\n
    Wort\n
    \n

    Sätze werden mit einer leeren Zeile getrennt. Die Datei endet mit \n\n

    Nachdem die Tags mithilfe des HMM ermittelt worden sind, für mehr Information siehe hmm.py,
    wird das Ergebnis im Format in die Datei geschrieben:

    Wort\tTag\n
    Wort\tTag\n
    \n

    Sätze werden mit einer leeren Zeile getrennt. Die Datei endet mit \n\n

    Auf Fehlerfälle des HMM wird nicht eingegangen, diese äußern sich darin dass die Wahrscheinlichkeit
    -inf und jedes Wort das Tag "Start" hat. Die Wahrscheinlichkeit die vom Viterbi-Alog. berechnet wird,
    wird auch fallen gelasssen.

    :param input_descriptor: zu taggende Datei
    :param output: Das Ergebnis des HMM

    :param trans: aka Bi-Gramme bi-gramme[wort1][wort2] = log Wahrscheinlichkeit
    siehe auch hmm.py

    :param emissions: emissions[tag][wort] = log Wahrscheinlichkeit
    siehe auch hmm.py
    """

    #einlesen der Datei
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
    """
    Ermittelt wie viele Zeilen in % übereinstimmten und gibt diese aus, sowie auch die Anzahl der Zeilen und Korrekter
    Zeilen.

    Es wird auch das Delta beider Dateien ermittelt, das Ergebnis wird in eine Datei geschrieben, es wird das diff-Format
    verwendet (siehe UNIX Standard oder Wikipedia).

    :param compare_descriptor: Vergleichsdatei 1
    :param eval_descriptor: Vergleichsdatei 2 oder Eval-Datei
    :param diff_results_descriptor: Datei wo das Ergebnis gepspeichert werden soll, wird überschrieben.
    """
    with compare_descriptor:
        compare = compare_descriptor.readlines()

    with eval_descriptor:
        eval = eval_descriptor.readlines()

    if len(compare) > len(eval):
        print("Die Vergleichsdatei hat mehr Zeilen als die Eval-Datei")

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
    """
    Gibt eine einfache Hilfe aus.
    """
    print("Alle parameter sind Dateien!")
    print("\ttrain <training corpus> <bi gramm output> <emission output>")
    print("\ttag <bi gramm input> <emission input> <corpus> <tagged corpus out>")
    print("\teval <vergleichsdatei> <eval-datei> <diff output>")
    print("\ttrain-tag-eval <training corpus> <bi gramm output> <emission output> <bi gramm input> <emission input> "
          "<corpus> <tagged corpus out> <vergleichsdatei> <eval-datei> <diff output>")


def train(training_corpus_file_name, bi_gram_out_file_name, emission_out_file_name, denoising):
    """
    Dies ist eine Hilfs-Funktion welche mit compute_trans_and_emission die Bi-Gramme und Emissions berechnet
    und diese dann in die jeweilige Dateischreibt. Die Dateien werden überschrieben (siehe save_nested_dict).

    :param training_corpus_file_name:
    :param bi_gram_out_file_name:
    :param emission_out_file_name:
    :param denoising:
    """
    training_corpus_descriptor = validate_file_and_open(training_corpus_file_name, 'r')
    trans_and_emission = compute_trans_and_emission(training_corpus_descriptor, denoising)

    trans = trans_and_emission[0]
    emissions = trans_and_emission[1]

    save_nested_dict(trans, bi_gram_out_file_name)
    save_nested_dict(emissions, emission_out_file_name)


def tag(bi_gramm_file_name, emission_file_name, untagged_text, tagged_out_file_name):
    """
    Hilfsfunktion welche die Bi-Gramme und Emissionen mithilfe von read_nested_dict liest
    und draus folgend die Funktion tag_file mit diesen Parametern aufruft.
    """

    bi_grams = read_nested_dict(bi_gramm_file_name)
    emissions = read_nested_dict(emission_file_name)
    emissions = smooth_emissions(emissions)

    tag_file(validate_file_and_open(untagged_text, 'r'), tagged_out_file_name, bi_grams, emissions)


if __name__ == "__main__":
    """
    Die Main-Funktion erwartet das Arguemnte übergeben werden, wenn keine übergeben werden, wird
    eine Hilfe ausgegeben.
    
    Es gibt verschiedene Modi, diese prüfen immer dass die richtige Anzahl an Argumenten übergeben wird,
    es gibt keine Default-Werte.
    
    Ein Modus ist immer der erste Argument nach pyhton3 project.py <Modus> <Args…>
    
    Die Argumente der verschiedenen Modi, kann man aus print_help() entnehmen.
    
    Der Modus train ruft schlussendlich die Funktion compute_trans_and_emission und speichert das Ergebnis
    in die jeweilige Text Datei. Dieser Modus ist zum trainieren des HMM Models gedacht. train-denoising
    setzt schlussendlich nur die denoising Flag siehe compute_trans_and_emission.
    
    Der Modus tag taggt den Text siehe tag and tag_file für mehr.
    
    Der Modus eval siehe eval_files
    
    Der Modus train-tag-eval oder train-denoising-tag-eval ist das hintereinander ausführen der Modi: 
    train, tag und eval.
    
    Falls ein falscher Modus angeben wird, wird die Hilfe ausgegeben.
    """
    if (len(sys.argv) - 1) <= 1:
        print("Es wurden keine oder zu wenige Argumente übergeben!")
        print_help()
        sys.exit(-1)

    mode = sys.argv[1]

    if mode == "train" or mode == "train-denoising":
        if (len(sys.argv) - 1) != 4:
            print("Die Anzahl der Argumente stimmt nicht!")
            sys.exit(-1)

        training_corpus_file_name = sys.argv[2]
        bi_gram_out_file_name = sys.argv[3]
        emission_out_file_name = sys.argv[4]

        train(training_corpus_file_name, bi_gram_out_file_name, emission_out_file_name, mode == "train-denoising")
    elif mode == "tag":
        if (len(sys.argv) - 1) != 5 and mode == "tag":
            print("Die Anzahl der Argumente stimmt nicht!")
            sys.exit(-1)

        bi_gramm_file_name = sys.argv[2]
        emission_file_name = sys.argv[3]
        untagged_text = sys.argv[4]
        tagged_out_file_name = sys.argv[5]

        tag(bi_gramm_file_name, emission_file_name, untagged_text, tagged_out_file_name)
    elif mode == "eval":
        if (len(sys.argv) - 1) != 4:
            print("Die Anzahl der Argumente stimmt nicht!")
            sys.exit(-1)

        compare_descriptor = validate_file_and_open(sys.argv[2], 'r')
        eval_descriptor = validate_file_and_open(sys.argv[3], 'r')
        diff_results_descriptor = validate_file_and_open(sys.argv[4], 'w')

        eval_files(compare_descriptor, eval_descriptor, diff_results_descriptor)
    elif mode == "train-tag-eval" or mode == "train-denoising-tag-eval":
        if (len(sys.argv) - 1) != 11:
            print("Die Anzahl der Argumente stimmt nicht!")
            sys.exit(-1)

        training_corpus_file_name = sys.argv[2]
        bi_gram_out_file_name = sys.argv[3]
        emission_out_file_name = sys.argv[4]

        train(training_corpus_file_name, bi_gram_out_file_name, emission_out_file_name,
              mode == "train-denoising-tag-eval")

        bi_gramm_file_name = sys.argv[5]
        emission_file_name = sys.argv[6]
        untagged_text = sys.argv[7]
        tagged_out_file_name = sys.argv[8]

        tag(bi_gramm_file_name, emission_file_name, untagged_text, tagged_out_file_name)

        compare_descriptor = validate_file_and_open(sys.argv[9], 'r')
        eval_descriptor = validate_file_and_open(sys.argv[10], 'r')
        diff_results_descriptor = validate_file_and_open(sys.argv[11], 'w')

        eval_files(compare_descriptor, eval_descriptor, diff_results_descriptor)
    else:
        print("Kein bekannter Modus für das Programm")
        print_help()
        sys.exit(-1)
