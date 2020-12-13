import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyphen
import re
import string
import itertools
from dante_by_tonedrev_syl.text_processing import remove_punctuation, special_tokens, prettify_text
from dante_by_tonedrev_syl.tone import ToneTagger


# Splits a single word into syllables.
def syllabify_word(text):
    return _perform_final_splits(_perform_initial_splits(text))

# Performs the first (easy and unambiguous) phase of syllabification.
def _perform_initial_splits(text):
    return _split_hiatus(_split_dieresis(_split_double_cons(_split_multiple_cons(text))))

# Performs the second (difficult and heuristic) phase of syllabification.
def _perform_final_splits(text):
    # ho aggiunto l'h -> 'richeggio'
    cvcv = r"""(?i)([bcdfglmnpqrstvz][ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu]+)([bcdfglmnpqrstvz]+[ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUuHh]+)"""
    vcv = r"""(?i)([ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu]+)([bcdfglmnpqrstvz]+[ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu]+)"""
#    vv = r"""(?i)(?<=[ÄäAaËëEeÏïIiÖöOoÜüUu])(?=[ÄäAaËëEeÏïIiÖöOoÜüUu])"""

#    vv = r"""(?i)(?<=[AaEeIiOoUu])(?=[AaEeIiOoUu])"""

    vv = r"""(?i)(?<=[ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu])(?=[ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu])"""

    # Split the contoid vocoid - contoid vocoid case (eg. ca-ne). Deterministic.
    out = re.sub(cvcv, r"""\1#\2""", text)
    # Split the vocoid - contoid vocoid case (eg. ae-reo). Deterministic.
    out = re.sub(vcv, r"""\1#\2""", out)

    # Split the vocoid - vocoid case (eg. a-iuola). Heuristic.
    out = _clump_diphthongs(out)
    out = re.sub(vv, r"""#""", out)
    out = re.sub("§", "", out)

    return out

# Splits double consonants (eg. al-legro)
def _split_double_cons(text): # ok
    doubles = re.compile(r"""(?i)(([bcdfglmnpqrstvz])(?=\2)|c(?=q))""")
    return "#".join(doubles.sub(r"""\1@""", text).split("@"))

# Splits multiple consonants, except: impure s (sc, sg, etc.), mute followed by liquide (eg. tr), digrams and trigrams.
def _split_multiple_cons(text):
    impures = re.compile(r"""(?i)(s(?=[bcdfghlmnpqrtvz]))""")
    muteliquide = re.compile(r"""(?i)([bcdfgptv](?=[lr]))""")
    digrams = re.compile(r"""(?i)(g(?=l[iì])|g(?=n[aeiouàèéìòóù])|s(?=c[eèéiì])|[cg](?=h[eèéiì])|[cg](?=i[aàoòuù]))""")
    trigrams = re.compile(r"""(?i)(g(?=li[aàoòuù])|s(?=ci[aàoòuù]))""")
    multicons = re.compile(r"""(?i)([bcdfglmnpqrstvz](?=[bcdfglmnpqrstvz]+))""")

    # Preserve non admissibile splits.
    out ="§".join(impures.sub(r"""\1@""", text).split("@"))
    out = "§".join(muteliquide.sub(r"""\1@""", out).split("@"))
    out = "§".join(digrams.sub(r"""\1@""", out).split("@"))
    out = "§".join(trigrams.sub(r"""\1@""", out).split("@"))
    # Split everything else.
    out = "#".join(multicons.sub(r"""\1@""", out).split("@"))

    return "".join(re.split("§", out))

# Splits dieresis.
def _split_dieresis(text):
    dieresis = re.compile(r"""(?i)([ÄäËëÏïÖöÜü](?=[aeiou])|[aeiou](?=[ÄäËëÏïÖöÜü]))""")
    return "#".join(dieresis.sub(r"""\1@""", text).split("@"))

# Splits SURE hiatuses only. Ambiguous ones are heuristically considered diphthongs.
def _split_hiatus(text):
    # ho tolto cose... i - u caso 'più','guida'
    # e aggiunto ^
    # hiatus = re.compile(r"""(?i)([aeoàèòóé](?=[aeoàèòóé])|[rbd]i(?=[aeou])|tri(?=[aeou])|[ìù](?=[aeiou])|[aeiou](?=[ìù]))""")
    
    # ok
#    hiatus = re.compile(r"""(?i)([aeoàèòóé](?=[aeoàèòóé])|^[rbd]i(?=[aeou])|^tri(?=[aeou])|[ì](?=[aeo])|[aeo](?=[ì])|[ù](?=[aeo])|[aeo](?=[ù]))""")


    hiatus = re.compile(r"""(?i)([aeo](?=[aeo])|^[rbd]i(?=[aeou])|^tri(?=[aeou])|[ì](?=[aeo])|[aeo](?=[ì])|[ù](?=[aeo])|[aeo](?=[ù]))""")

    return "#".join(hiatus.sub(r"""\1@""", text).split("@"))

# Prevents splitting of diphthongs and triphthongs.
def _clump_diphthongs(text):
    diphthong = r"""(?i)(i[aeouàèéòóù]|u[aeioàèéìòó]|[aeouàèéòóù]i|[aeàèé]u)"""
    diphthongsep = r"""(\{.)(.\})"""
    # triphthong = r"""(?i)(i[àèé]i|u[àòó]i|iu[òó]|ui[èéà])""" #nostra
    triphthong = r"""(?i)(i[àèéòó]i|u[àèéòó]i|iu[òó]|ui[èéà])""" #nostra

    triphthongsep = r"""(\{.)(.)(.\})"""

    out = re.sub(triphthong, r"""{\1}""", text)
    out = re.sub(triphthongsep, r"""\1§\2§\3""", out)
    out = re.sub(diphthong, r"""{\1}""", out)
    out = re.sub(diphthongsep, r"""\1§\2""", out)
    out = re.sub(r"""[{}]""", "", out)

    return out

# def is_diphthong(text):
#     diphthong = r"""(?i)(i[aeouàèéòóù]|u[aeioàèéìòó]|[aeouàèéòóù]i|[aeàèé]u)"""
#     if re.search(diphthong, text):
#         return True
#     else:
#         return False

# def is_hiatus(text):
#     hiatus = re.compile(r"""(?i)([aeoàèòóé](?=[aeoàèòóé])|^[rbd]i(?=[aeou])|^tri(?=[aeou])|[ì](?=[aeo])|[aeo](?=[ì])|[ù](?=[aeou])|[aeou](?=[ù]))""")
#     if re.search(hiatus, text):
#         return True
#     else:
#         return False


def is_toned_syl(syl):
    toned_v = r"""(?i)([ÁÀàáÉÈèéÍÌíìÓÒóòÚÙúù]{1})"""
    if len(re.findall(toned_v, syl)) == 1:
        return True
    else:
        return False

def is_hendecasyllable(syllables, special_tokens):
    syllables = [ prettify_text(s, special_tokens).strip() for s in syllables if s not in special_tokens.values() ]
    # ENDECASILLABO A MAIORE O A MINORE
    if len(syllables) > 9:
        return is_toned_syl(syllables[9]) # aggiungere check max len 12
    else:
        return False

# Apply synalepha by need
def _apply_synalepha(syllables, special_tokens):
    
    # if is_hendecasyllable(syllables, special_tokens):
    #     return syllables

    # syllables_cleaned = [ prettify_text(s, special_tokens).strip() for s in syllables ]
    # syllables_cleaned = [ s for s in syllables if s != '' ]
    syllables_cleaned = [ prettify_text(s, special_tokens).strip() for s in syllables if s not in special_tokens.values() ]
    
    if len(syllables_cleaned) <= 9:
        return syllables 

    # SMARAGLIATA    
    vowels = "ÁÀAaàáÉÈEeèéIÍÌiíìOoóòÚÙUuúùHh'" # considerare l'H come vocale???????

    n_synalepha = 0

    i = 1
    while i < (len(syllables) - 1):
        if syllables[i] == special_tokens['WORD_SEP']:
            pre_syl = syllables[i-1]
            next_syl = syllables[i+1]
            if pre_syl[-1] in vowels and next_syl[0] in vowels: # aggiungere is_dittongo (e not is_iato???) NON CORRETTO! USIAMOLI NELLA SMARAGLIATA
#            if pre_syl[-1] in vowels and next_syl[0] in vowels and is_diphthong(''.join([pre_syl[-1], next_syl[0]])):
#            if pre_syl[-1] in vowels and next_syl[0] in vowels and not is_hiatus(''.join([pre_syl[-1], next_syl[0]])):
                i += 1
                n_synalepha+=1
        i+=1

#    print(n_synalepha, syllables)
    x = ['0', '1']
    combinations = [''.join(p) for p in itertools.product(x, repeat=n_synalepha)]
    synalepha_results = { k: [] for k in combinations }
        
    for c in combinations:    
        syn_index = 0
        result = [syllables[0]]
        i = 1
        while i < (len(syllables) - 1) :
            if syllables[i] == special_tokens['WORD_SEP']:
                pre_syl = syllables[i-1]
                next_syl = syllables[i+1]
                if pre_syl[-1] in vowels and next_syl[0] in vowels:
                    if c[syn_index] == '1':
                        result.append(result[-1] + syllables[i] + next_syl)
                        del result[-2]
                        i += 1
                        
                    else:
                        result.append(syllables[i])               
                    syn_index+=1
                else:
                    result.append(syllables[i])               
            else:
                result.append(syllables[i])
            i+=1
        result.append(syllables[-1])
        synalepha_results[c] = result

    # print(synalepha_results)
    # remove WORD_SEP to count the syllables
    
    best_syllabification = [] 
    best_score = -1
    # get best syllabification

    for syllables_list in synalepha_results.values():
        syllables_cleaned = [ prettify_text(s, special_tokens).strip() for s in syllables_list if s not in special_tokens.values() ]
        
        if len(syllables_cleaned) >=10 and len(syllables_cleaned) <= 11 and is_hendecasyllable(syllables_list, special_tokens):
            if best_score < 5:
                best_syllabification = syllables_list
                best_score = 5
        
        if len(syllables_cleaned) >=10 and len(syllables_cleaned) <= 12 and is_hendecasyllable(syllables_list, special_tokens):
            if best_score < 4:
                best_syllabification = syllables_list
                best_score = 4
        
        if is_hendecasyllable(syllables_list, special_tokens):
            if best_score < 3:
                best_syllabification = syllables_list
                best_score = 3

        if len(syllables_cleaned) >=9 and len(syllables_cleaned) <= 12:
            if best_score < 2:
                best_syllabification = syllables_list
                best_score = 2
        
        if len(syllables_cleaned) >=5:
            if best_score < 1:
                best_syllabification = syllables_list
                best_score = 1


    # if not best_syllabification:
    #     best_syllabification = syllables

    return best_syllabification


def _apply_synalepha_v1(syllables, special_tokens):
    
    if is_hendecasyllable(syllables, special_tokens):
        return syllables

    syllables_cleaned = [ prettify_text(s, special_tokens).strip() for s in syllables ]
    syllables_cleaned = [ s for s in syllables if s != '' ]

    if len(syllables_cleaned) <= 9:
        return syllables 


    # SMARAGLIATA    
    vowels = "ÁÀAaàáÉÈEeèéIÍÌiíìOoóòÚÙUuúù'" # considerare l'H come vocale???????

    n_synalepha = 0


    i = 1
    while i < (len(syllables) - 1):
        if syllables[i] == special_tokens['WORD_SEP']:
            pre_syl = syllables[i-1]
            next_syl = syllables[i+1]
            if pre_syl[-1] in vowels and next_syl[0] in vowels: # aggiungere is_dittongo (e not is_iato???) NON CORRETTO! USIAMOLI NELLA SMARAGLIATA
#            if pre_syl[-1] in vowels and next_syl[0] in vowels and is_diphthong(''.join([pre_syl[-1], next_syl[0]])):
#            if pre_syl[-1] in vowels and next_syl[0] in vowels and not is_hiatus(''.join([pre_syl[-1], next_syl[0]])):
                i += 1
                n_synalepha+=1
        i+=1

#    print(n_synalepha, syllables)


    result = syllables
    synalepha_to_apply = 1
    while not is_hendecasyllable(result, special_tokens) and synalepha_to_apply<=n_synalepha:
        
        applied_synalepha = 0
        result = [syllables[0]]
        i = 1
        completed = False
        while i < (len(syllables) - 1) :
            
            if syllables[i] == special_tokens['WORD_SEP']:
                pre_syl = syllables[i-1]
                next_syl = syllables[i+1]
                if completed:
                    result.append(syllables[i])               
                else:

                    if pre_syl[-1] in vowels and next_syl[0] in vowels:
                        result.append(result[-1] + syllables[i] + next_syl)
                        del result[-2]
                        i += 1
                        
                        applied_synalepha+=1
                        if applied_synalepha == synalepha_to_apply:
                            completed = True
                    else:
                        result.append(syllables[i])               
            else:
                result.append(syllables[i])
            i+=1
        result.append(syllables[-1])
        synalepha_to_apply+=1
    # remove WORD_SEP to count the syllables

    return result

def _apply_synalepha_backup(syllables, special_tokens):
    
    vowels = "ÁÀAaàáÉÈEeèéIÍÌiíìOoóòÚÙUuúù'"
#        vowels = "ÁÀAaàáÉÈEeèéIÍÌiíìOoóòÚÙUuúù"

    result = [syllables[0]]
    i = 1
    while i < (len(syllables) - 1):
        if syllables[i] == special_tokens['WORD_SEP']:
            pre_syl = syllables[i-1]
            next_syl = syllables[i+1]
            if pre_syl[-1] in vowels and next_syl[0] in vowels: # aggiungere is_dittongo (e not is_iato???) NON CORRETTO! USIAMOLI NELLA SMARAGLIATA
#            if pre_syl[-1] in vowels and next_syl[0] in vowels and is_diphthong(''.join([pre_syl[-1], next_syl[0]])):
#            if pre_syl[-1] in vowels and next_syl[0] in vowels and not is_hiatus(''.join([pre_syl[-1], next_syl[0]])):
                result.append(result[-1] + syllables[i] + next_syl)
                del result[-2]
                i += 1
            else:
                result.append(syllables[i])               
        else:
            result.append(syllables[i])
        i+=1
    result.append(syllables[-1])

    # remove WORD_SEP to count the syllables

    return result

def remove_tone(syllables, special_tokens):
    toned_vowels = {
        'à': 'a',
        'è': 'e',
        'é': 'e',
        'ì': 'i',
        'ò': 'o',
        'ó': 'o',
        'ù': 'u',
    }
    cleaned_syllables = []
    for i, syl in enumerate(syllables[:-1]):
        syl = syllables[i]
        next_s = syllables[i+1]
        if syl.strip() in special_tokens.values():
            cleaned_syllables.append(syl)
        elif next_s in special_tokens.values() and syl[-1] in toned_vowels.keys():
            cleaned_syllables.append(syl)
        elif special_tokens['WORD_SEP'] in syl:
            new_sub_syls = []
            sub_syls = syl.split(special_tokens['WORD_SEP'])
            for s in sub_syls[:-1]:
                if s[-1] in toned_vowels.keys():
                    new_sub_syls.append(s)
                else:
                    new_sub_syls.append(''.join([toned_vowels[c] if c in toned_vowels.keys() else c for c in s]))

            if next_s in special_tokens.values() and sub_syls[-1] in toned_vowels.keys():
                new_sub_syls.append(sub_syls[-1])
            else:
                new_sub_syls.append(''.join([toned_vowels[c] if c in toned_vowels.keys() else c for c in sub_syls[-1]]))
            new_syl = special_tokens['WORD_SEP'].join(new_sub_syls)
            cleaned_syllables.append(new_syl)

        else: 
            cleaned_s = ''.join([toned_vowels[c] if c in toned_vowels.keys() else c for c in syl])
            cleaned_syllables.append(cleaned_s)

    if syllables[-1] in toned_vowels.keys():
        cleaned_syllables.append(syllables[-1])
    else:
        cleaned_syllables.append(''.join([toned_vowels[c] if c in toned_vowels.keys() else c for c in syllables[-1] ]))
    return cleaned_syllables


def syllabify_verse_prettify(verse, special_tokens, tone_tagger, synalepha=True):
    syllables = syllabify_verse(verse, special_tokens, tone_tagger, synalepha)
    syllables = [ prettify_text(s, special_tokens).strip() for s in syllables ]
    syllables = [ s for s in syllables if s != '' ]
    return syllables


def syllabify_verse(verse, special_tokens, tone_tagger, synalepha=True):
    if verse.strip() == '':
        return []
    if verse in special_tokens.values():
        return [verse]

    words = [ w for w in verse.split() ]

    list_of_syllables = [ syllabify_word(tone_tagger.tone(w)).split('#') if w.strip() not in special_tokens.values() else [w] for w in words ] 
    
    ## [['nel'], ['<word_sep>'], ['mez', 'zo'], ['<word_sep>'], ['del'], ['<word_sep>'], ['cam', 'min'], ['<word_sep>'], ['di'], ['<word_sep>'], ['no', 'stra'], ['<word_sep>'], ['vi', 'ta'], ['<end_of_verso>']]
    
    syllables = []
    for s in list_of_syllables:
        syllables+=s
    ## ['nèl', '<word_sep>', 'mèz', 'zo', '<word_sep>', 'dèl', '<word_sep>', 'cam', 'mìn', '<word_sep>', 'di', '<word_sep>', 'nò', 'stra', '<word_sep>', 'vì', 'ta', '<end_of_verso>']


    # removing usless tones
#    syllables = remove_tone(syllables, special_tokens)
#    print('before',syllables)
#    print()
    
    if synalepha:
        syllables = _apply_synalepha(syllables, special_tokens)

    # removing usless tones
#    syllables = remove_tone(syllables, special_tokens)
#    print('after',syllables)

    return syllables

if __name__ == "__main__":

#    print(special_tokens)

    with open("divina_commedia_accent_cleaned.txt","r") as f:
        divine_comedy = f.read()

#    divine_comedy = prettify_text(divine_comedy,special_tokens)
    divine_comedy_list = divine_comedy.split("\n")

    divine_comedy_list = [ line for line in divine_comedy_list if line.strip() not in special_tokens.values() ]

    tone_tagger = ToneTagger()
    count = 0
    for line in divine_comedy_list[:]:
        syllables = syllabify_verse(line, special_tokens, tone_tagger)

        if not is_hendecasyllable(syllables, special_tokens):
            count+=1
#        print(syllables)
        syllables = [ syl for syl in syllables if syl != special_tokens['WORD_SEP'] ]
        syllables = [ syl for syl in syllables if syl != special_tokens['END_OF_VERSO'] ]
        syllables = [ syl.replace(special_tokens['WORD_SEP'], ' ') for syl in syllables ]

        size = len(syllables)

        if line.strip() not in special_tokens.values():
#            print("\n"+line)
#            if size < 10 or size > 12:

            # if not is_hendecasyllable(syllables, special_tokens):
            print(line.replace(special_tokens['WORD_SEP'], '').replace(special_tokens['END_OF_VERSO'], ''))
            print(size, '-'.join(syllables))
            print()

    print(str(count)+'/'+str(len(divine_comedy_list)) + " verses still wrong")
