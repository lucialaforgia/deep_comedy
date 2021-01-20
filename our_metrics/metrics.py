import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import re


from dante_by_tonedrev_syl.syllabification import tone_and_syllabify_verse, is_hendecasyllable
from dante_by_tonedrev_syl.text_processing import special_tokens, prettify_text

from dante_by_tonedrev_syl.tone import ToneTagger

def eval(generated_canto, synalepha=True):
    
    generated_verses_list = add_special_tokens(generated_canto)
    toned_verses_syls = []
    tone_tagger = ToneTagger()

    for v in generated_verses_list:
        syls = tone_and_syllabify_verse(v, special_tokens, tone_tagger, synalepha=synalepha)
        toned_verses_syls.append(syls)
    

    n_strophes = get_n_strophes(generated_canto)
    n_well_formed_terzine = get_well_formed_terzine(generated_canto)
    mean_verse_len, std_verse_len = get_mean_std_verse_length(toned_verses_syls, synalepha)
    last_single_verse = is_last_single_verse_present(generated_canto)
    correct_rhymes = get_well_formed_rhymes(toned_verses_syls, synalepha)

    n_verses = len(get_verses(generated_canto))
    correct_hendecasyllables = get_correct_hendecasyllables(toned_verses_syls, synalepha)

    return {
        'Number of verses': n_verses ,
        'Number of strophes': n_strophes ,
        'Number of well formed terzine': n_well_formed_terzine,
        'Last single verse': last_single_verse,
        'Average syllables per verse': '{:.2f} ± {:.2f}'.format(mean_verse_len, std_verse_len),
        'Hendecasyllabicness score': '{:.4f}'.format(correct_hendecasyllables/n_verses),
        'Rhymeness score': '{:.4f}'.format(correct_rhymes),
    }


def add_special_tokens(generated_canto):
    new_canto_verses = [special_tokens['START_OF_CANTO']]
    new_canto_verses += [special_tokens['START_OF_TERZINA']]
    canto_verses = generated_canto.split('\n')
    for idx, verse in enumerate(canto_verses):
        if verse == '' and idx != len(canto_verses)-2:
            new_canto_verses.append(special_tokens['END_OF_TERZINA'])
            new_canto_verses.append(special_tokens['START_OF_TERZINA'])
        elif idx == len(canto_verses)-2:
            new_canto_verses.append(special_tokens['END_OF_TERZINA'])
        else:
            verse = verse.replace(' ', ' '+special_tokens['WORD_SEP']+' ')
            verse += ' '+special_tokens['END_OF_VERSO']
            new_canto_verses.append(verse)
    new_canto_verses.append(special_tokens['END_OF_CANTO'])
    return new_canto_verses

def get_terzine(generated_canto):
    generated_canto = generated_canto.strip()
    return generated_canto.split('\n\n')


def is_last_single_verse_present(generated_canto):
    terzine = get_terzine(generated_canto)
    return len(terzine[-1].split('\n')) == 1

def get_n_strophes(generated_canto):
    return len(get_terzine(generated_canto))

def get_well_formed_terzine(generated_canto):
    terzine = get_terzine(generated_canto)
    n_well_formed_terzine = 0
    for t in terzine:
        if len(t.split('\n')) == 3:
            n_well_formed_terzine+=1
    return n_well_formed_terzine

def get_verses(generated_canto):
    #remove empty line and extract verses
    generated_canto_list = generated_canto.split("\n")
    verses = [line.strip() for line in generated_canto_list if line != '']
    return verses

def get_mean_std_verse_length(toned_verses_syls, synalepha):
    lengths = []
    for verse_syls in toned_verses_syls:
        syllables = [ s.strip() for s in verse_syls if s.strip() not in special_tokens.values()]
        #syllables = [ s for s in syllables if s != '' and s != '\n']
        if syllables:
            lengths.append(len(syllables))
    lengths = np.array(lengths)
    return np.mean(lengths), np.std(lengths)

def get_correct_hendecasyllables(toned_verses_syls, synalepha):
    count = 0
    for verse_syls in toned_verses_syls:
        if is_hendecasyllable(verse_syls, special_tokens):
            count += 1
    return count


def is_rhyme(verse_1, verse_2):
    rev_1 = verse_1[::-1]
    rev_2 = verse_2[::-1]
    toned_v = r"""(?i)([ÁÀàáÉÈèéÍÌíìÓÒóòÚÙúù]{1})"""
    rhyme_1 = rev_1[:re.search(toned_v, rev_1).start()+1][::-1]
    rhyme_2 = rev_2[:re.search(toned_v, rev_2).start()+1][::-1]
    if rhyme_1 == rhyme_2:
        return True
    else:
        return False

def get_well_formed_rhymes(toned_verses_syls, synalepha):
    verses = [ ''.join(verse_syls) for verse_syls in toned_verses_syls]
    generated_canto = ''.join(verses)
    generated_canto = prettify_text(generated_canto, special_tokens)

    generated_canto_list = generated_canto.split("\n")
    generated_canto_list = [line.strip() for line in generated_canto_list if line != 'CANTO']
    generated_canto = "\n".join(generated_canto_list)


    n_rhymes = 0
    correct_rhymes = 0

    generated_canto = generated_canto.replace('\n\n', '\n').strip()
    generated_canto_list = generated_canto.split("\n")

    triplets = []
    l = None
    for i, verse in enumerate(generated_canto_list):
        if i % 3 == 0:
            if l:
                triplets.append('\n'.join(l))
            l = [verse]
        else:
            l.append(verse)

    triplets.append('\n'.join(l))

    for i in range(0,len(triplets)-1,1):
        t1 = triplets[i].split('\n')
        t2 = triplets[i+1].split('\n')
            
        if i==0:
            n_rhymes+=1
            if is_rhyme(t1[0], t1[2]):
                correct_rhymes+=1        

        n_rhymes+=1
        
        if i == len(triplets)-2 and len(t2) <3 :

            if is_rhyme(t1[1], t2[0]):
                correct_rhymes+=1

        else:
            if is_rhyme(t1[1], t2[0]):
                correct_rhymes+=1/3         

            if is_rhyme(t2[0], t2[2]):
                correct_rhymes+=1/3

            if is_rhyme(t1[1], t2[2]):
                correct_rhymes+=1/3
            

    return correct_rhymes/n_rhymes