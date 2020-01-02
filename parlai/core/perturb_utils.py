# Perutbation operations
import numpy as np
np.random.seed(seed=300)
import spacy
from parlai.core.dict import DictionaryAgent
import math
import random

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load(
    'en_core_web_sm', parser=False, entity=False, matcher=False, add_vectors=False, tagger=True
)

class Perturb(object):
    def __init__(self, opt):
        self.opt = opt
        self.splitter = "\n"
        print("Perturber created !")

    def perturb(self, act):
        if self.opt['perturb'] == 'None':
            return act
        turns = self._get_turns(act)
        if len(turns) < 3:
            print("less than 3 turns")
            print("turns : {}".format(act['text']))
            return act
        if "last_few_only" in self.opt['perturb']:
            max_num_turns_to_retain = int(self.opt['perturb'].split("__")[-1])
            turns = self.last_few_only(turns, max_num_turns_to_retain)
        elif "shuffle" in self.opt['perturb']:
            turns = self.shuffle(turns)
        elif "reverse_utr_order" in self.opt['perturb']:
            turns = self.reverse_utr_order(turns)
        elif "only_last" in self.opt['perturb']:
            turns = self.only_last(turns)
        elif "worddrop" in self.opt['perturb']:
            turns = self.word_drop(turns)
        elif "wordshuf" in self.opt['perturb']:
            turns = self.word_shuf(turns)
        elif "wordreverse" in self.opt['perturb']:
            turns = self.word_reverse(turns)
        elif "verbdrop" in self.opt['perturb']:
            turns = self.verb_drop(turns)
        elif "noundrop" in self.opt['perturb']:
            turns = self.noun_drop(turns)
        elif "drop" in self.opt['perturb']:
            turns = self.drop(turns)
        elif "swap" in self.opt['perturb']:
            turns = self.swap(turns)
        elif "repeat" in self.opt['perturb']:
            turns = self.repeat(turns)
        #My modification from here
        elif "replaceall" in self.opt['perturb']:
            turns = self.replaceall(turns)
        elif "replace10percent" in self.opt['perturb']:
            turns = self.replace10percent(turns)
        elif "replace20percent" in self.opt['perturb']:
            turns = self.replace20percent(turns)
        elif "replace30percent" in self.opt['perturb']:
            turns = self.replace30percent(turns)
        else:
            assert "Invalid perturb mode : {}. Valid : random, drop, swap, repeat".format(self.opt['perturb'])
        self._update_act(turns, act)
        return act

    def swap(self, turns):
        if "first" in self.opt['perturb']:
            pos = [0, 1]
        elif "last" in self.opt['perturb']:
            last_id = len(turns) - 1
            pos = [last_id - 1, last_id]
        else:
            pos = np.random.randint(len(turns), size=2)
        tmp = turns[pos[0]]
        turns[pos[0]] = turns[pos[1]]
        turns[pos[1]] = tmp
        return turns

    def drop(self, turns, pos=None):
        if "first" in self.opt['perturb']:
            pos = 0
        elif "last" in self.opt['perturb']:
            pos = len(turns) - 1
        else:
            pos = np.random.randint(len(turns))
        return [turn for idx, turn in enumerate(turns) if idx != pos]

    def repeat(self, turns, pos=None):
        if "first" in self.opt['perturb']:
            pos = 0
        elif "last" in self.opt['perturb']:
            pos = len(turns) - 1
        else:
            pos = np.random.randint(len(turns))
        return turns[:pos] + [turns[pos]] + turns[pos:]

    def word_drop(self, turns, pos=None):
        if "first" in self.opt['perturb']:
            pos = 0
        elif "last" in self.opt['perturb']:
            pos = len(turns) - 1
        else:
            pos = 'all'

        if pos == 'all':
            modified_turns = []
            for turn in turns:
                word_mask = np.random.binomial(
                    size=len(turn.split()), n=1, p=0.3
                )
                modified_turn = ' '.join(
                    [x for idx, x in enumerate(turn.split()) if word_mask[idx] == 0]
                )
                modified_turns.append(modified_turn)
            return modified_turns
        else:
            # Tune word dropout prob?
            word_mask = np.random.binomial(
                size=len(turns[pos].split()), n=1, p=0.3
            )
            modified_turn = ' '.join(
                [x for idx, x in enumerate(turns[pos].split()) if word_mask[idx] == 0]
            )
            return turns[:pos] + [modified_turn] + turns[pos:]

    def word_shuf(self, turns, pos=None):
        if "first" in self.opt['perturb']:
            pos = 0
        elif "last" in self.opt['perturb']:
            pos = len(turns) - 1
        else:
            pos = 'all'

        if pos == 'all':
            modified_turns = []
            for turn in turns:
                turn = turn.split()
                np.random.shuffle(turn)
                modified_turns.append(' '.join(turn))
            return modified_turns
        else:
            # Tune word dropout prob?
            turn = turns[pos]
            turn = turn.split()
            np.random.shuffle(turn)
            return turns[:pos] + [' '.join(turn)] + turns[pos:]

    def word_reverse(self, turns, pos=None):
        if "first" in self.opt['perturb']:
            pos = 0
        elif "last" in self.opt['perturb']:
            pos = len(turns) - 1
        else:
            pos = 'all'

        if pos == 'all':
            modified_turns = []
            for turn in turns:
                turn = turn.split()[::-1]
                modified_turns.append(' '.join(turn))
            return modified_turns
        else:
            # Tune word dropout prob?
            turn = turns[pos]
            turn = turn.split()[::-1]
            return turns[:pos] + [" ".join(turn)] + turns[pos:]

    def only_last(self, turns):
        return [turns[-1]]

    def reverse_utr_order(self, turns):
        return turns[::-1]

    def shuffle(self, turns):
        np.random.shuffle(turns)
        return turns

    def verb_drop(self, turns):
        if "first" in self.opt['perturb']:
            pos = 0
        elif "last" in self.opt['perturb']:
            pos = len(turns) - 1
        else:
            pos = 'all'

        if pos == 'all':
            modified_turns = []
            for turn in turns:
                processed_turn = nlp(turn)
                modified_turn = ' '.join([x.text for x in processed_turn if x.pos_ != 'VERB'])
                modified_turns.append(modified_turn)
            return modified_turns
        else:
            turn = turns[pos]
            processed_turn = nlp(turn)
            modified_turn = ' '.join([x.text for x in processed_turn if x.pos_ != 'VERB'])
            return turns[:pos] + [modified_turn] + turns[pos:]

    def noun_drop(self, turns):
        if "first" in self.opt['perturb']:
            pos = 0
        elif "last" in self.opt['perturb']:
            pos = len(turns) - 1
        else:
            pos = 'all'

        if pos == 'all':
            modified_turns = []
            for turn in turns:
                processed_turn = nlp(turn)
                modified_turn = ' '.join([x.text for x in processed_turn if x.pos_ != 'NOUN'])
                modified_turns.append(modified_turn)
            return modified_turns
        else:
            turn = turns[pos]
            processed_turn = nlp(turn)
            modified_turn = ' '.join([x.text for x in processed_turn if x.pos_ != 'NOUN'])
            return turns[:pos] + [modified_turn] + turns[pos:]

    def replaceall(self, turns):

        return turns

    def replace10percent(self, turns):
        self.opt['dict_file'] = self.opt['model_file'] + '.dict'
        dictionary = DictionaryAgent(self.opt)
        dictionary_len = len(dictionary)
        modified_turns = []
        for turn in turns:
            turn = turn.split()
            turn_len = len(turn)
            replace_len = math.ceil(turn_len * 0.1)
            replace_target_index = np.random.choice(turn_len, replace_len, replace=False).tolist()
            replace_target_index.sort()
            replace_target_words = [turn[x] for x in replace_target_index]
            replace_words = []
            i = 0
            while i < replace_len:
                replace_word_index = np.random.choice(dictionary_len, 1, replace=False).tolist()
                replace_word = dictionary.ind2tok[replace_word_index[0]]
                if replace_word == replace_target_words[i]:
                    i -= 1
                else:
                    replace_words.append(replace_word)
                i += 1
            for i in range(replace_len):
                turn[replace_target_index[i]] = replace_words[i]
            modified_turns.append(' '.join(turn))
        return modified_turns

    def replace20percent(self, turns):
        self.opt['dict_file'] = self.opt['model_file'] + '.dict'
        dictionary = DictionaryAgent(self.opt)
        dictionary_len = len(dictionary)
        modified_turns = []
        for turn in turns:
            turn = turn.split()
            turn_len = len(turn)
            replace_len = math.ceil(turn_len * 0.2)
            if replace_len == 1 and turn_len > 1:
                replace_len = 2
            replace_target_index = np.random.choice(turn_len, replace_len, replace=False).tolist()
            replace_target_index.sort()
            replace_target_words = [turn[x] for x in replace_target_index]
            replace_words = []
            i = 0
            while i < replace_len:
                replace_word_index = np.random.choice(dictionary_len, 1, replace=False).tolist()
                replace_word = dictionary.ind2tok[replace_word_index[0]]
                if replace_word == replace_target_words[i]:
                    i -= 1
                else:
                    replace_words.append(replace_word)
                i += 1
            for i in range(replace_len):
                turn[replace_target_index[i]] = replace_words[i]
            modified_turns.append(' '.join(turn))
        return modified_turns

    def replace30percent(self, turns):
        self.opt['dict_file'] = self.opt['model_file'] + '.dict'
        dictionary = DictionaryAgent(self.opt)
        dictionary_len = len(dictionary)
        modified_turns = []
        for turn in turns:
            turn = turn.split()
            turn_len = len(turn)
            replace_len = math.ceil(turn_len * 0.3)
            if (replace_len == 1 or replace_len == 2) and turn_len > 2:
                replace_len = 3
            replace_target_index = np.random.choice(turn_len, replace_len, replace=False).tolist()
            replace_target_index.sort()
            replace_target_words = [turn[x] for x in replace_target_index]
            replace_words = []
            i = 0
            while i < replace_len:
                replace_word_index = np.random.choice(dictionary_len, 1, replace=False).tolist()
                replace_word = dictionary.ind2tok[replace_word_index[0]]
                if replace_word == replace_target_words[i]:
                    i -= 1
                else:
                    replace_words.append(replace_word)
                i += 1
            for i in range(replace_len):
                turn[replace_target_index[i]] = replace_words[i]
            modified_turns.append(' '.join(turn))
        return modified_turns

    def last_few_only(self, turns, max_num_turns_to_retain):
        return turns[-max_num_turns_to_retain:]

    def _get_turns(self, act):
        turns = self._filter_out_personas(act['text'].split('\n'))
        return turns

    def _update_act(self, turns, act):
        act['text'] = self.splitter.join(self._get_persona_turns() + turns)

    def _filter_out_personas(self, turns):
        self.persona_turns = [turn for turn in turns if "your persona: " in turn]
        return [turn for turn in turns if "your persona: " not in turn]
    
    def _get_persona_turns(self):
        return self.persona_turns
        
