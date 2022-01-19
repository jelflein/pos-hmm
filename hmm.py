'''
Beispiel-Implementierung des Viterbi-Algorithmus für Hidden Markov Modelle.
'''

from math import log

from classify import classify_word

neginf = float("-inf")  # log-Wahrscheinlchkeit von 0 (minus unendlich)


class State:
    def __init__(self, prob, prev, out):
        self.prob = prob
        self.prev = prev
        self.out = out

    def __repr__(self):
        return 'S(%r, %r, %r)' % (self.prob, self.prev, self.out)


class HMM:
    def __init__(self, ptrans, pemit):
        self._ptrans = ptrans  # nested dict(): ptrans[tag] = { t1: logprob, t2: logprob, ... }
        self._pemit = pemit  # nested dict(): pemit[tag] = { word1: logprob, word2: logprob, ... }
        self._states = [state for state in ptrans]
        self.words = set()

        for key in pemit:
            self.words |= pemit[key].keys()

    def states(self):
        return self._states

    def pemit(self, x, o):  # P(o | x)
        try:
            return self._pemit[x][o]
        except KeyError:
            return neginf

    def ptrans(self, x, y):  # P(y | x)
        try:
            return self._ptrans[x][y]
        except KeyError:
            return neginf

    def maxprob(self, col, state, obs):
        prev = max(col, key=lambda prev: col[prev].prob + self.ptrans(prev, state))
        return State(col[prev].prob + self.ptrans(prev, state) + self.pemit(state, obs), prev, obs)

    def decode(self, observations):
        new_obsv = []

        for i in range(0, len(observations)):
            word = observations[i]

            if i == 0:
                word = "^" + word

            if word not in self.words:
                classified_word = classify_word(word, 0)

                if classified_word is None:
                    print(word)

                new_obsv.append(classified_word[0])
            else:
                classified_word = classify_word(word, 2)

                if classified_word is not None:
                    new_obsv.append(classified_word[0])
                else:
                    new_obsv.append(word)

        # trellis: Liste von Wörterbüchern (dict)
        viterbi = [{'START': State(0.0, False, '')}]

        for obs in new_obsv:
            viterbi.append({state: self.maxprob(viterbi[-1], state, obs) for state in self.states()})

        # Zustand mit der hächsten Wahrscheinlichkeit
        state = max(viterbi[-1], key=lambda state: viterbi[-1][state].prob)

        # ... und dessen Wahrscheinlichkeit
        prob = viterbi[-1][state].prob

        # Zustandsfolge berechnen:
        path = []
        for col in reversed(viterbi[1:]):
            path.append((col[state].out, state))
            state = col[state].prev

        word_tag_result = []
        reversed_path = path[::-1]
        for i in range(0, len(reversed_path)):
            word_tag = reversed_path[i]
            word_tag_result.append((observations[i], word_tag[1]))

        return prob, word_tag_result  # reversed(path)
