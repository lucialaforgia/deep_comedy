import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyphen
import re
import string
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
    cvcv = r"""(?i)([bcdfglmnpqrstvz][ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu]+)([bcdfglmnpqrstvz]+[ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu]+)"""
    vcv = r"""(?i)([ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu]+)([bcdfglmnpqrstvz]+[ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu]+)"""
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
    # hiatus = re.compile(r"""(?i)([aeoàèòóé](?=[aeoàèòóé])|[rbd]i(?=[aeou])|tri(?=[aeou])|[ìù](?=[aeiou])|[aeiou](?=[ìù]))""")

    hiatus = re.compile(r"""(?i)([aeoàèòóé](?=[aeoàèòóé])|[rbd]i(?=[aeou])|tri(?=[aeou])|[ì](?=[aeo])|[aeo](?=[ì])|[ù](?=[aeou])|[aeou](?=[ù]))""")
    return "#".join(hiatus.sub(r"""\1@""", text).split("@"))

# Prevents splitting of diphthongs and triphthongs.
def _clump_diphthongs(text):
    diphthong = r"""(?i)(i[aeouàèéòóù]|u[aeioàèéìòó]|[aeouàèéòóù]i|[aeàèé]u)"""
    diphthongsep = r"""(\{.)(.\})"""
    triphthong = r"""(?i)(i[àèé]i|u[àòó]i|iu[òó]|ui[èéà])""" #nostra
    triphthongsep = r"""(\{.)(.)(.\})"""

    out = re.sub(triphthong, r"""{\1}""", text)
    out = re.sub(triphthongsep, r"""\1§\2§\3""", out)
    out = re.sub(diphthong, r"""{\1}""", out)
    out = re.sub(diphthongsep, r"""\1§\2""", out)
    out = re.sub(r"""[{}]""", "", out)

    return out

def is_diphthong(text):
    diphthong = r"""(?i)(i[aeouàèéòóù]|u[aeioàèéìòó]|[aeouàèéòóù]i|[aeàèé]u)"""
    if re.search(diphthong, text):
        return True
    else:
        return False

def is_hiatus(text):
    hiatus = re.compile(r"""(?i)([aeoàèòóé](?=[aeoàèòóé])|[rbd]i(?=[aeou])|tri(?=[aeou])|[ì](?=[aeo])|[aeo](?=[ì])|[ù](?=[aeou])|[aeou](?=[ù]))""")
    if re.search(hiatus, text):
        return True
    else:
        return False

# Apply synalepha by need
def _apply_synalepha(syllables, special_tokens):
    
    # out = re.sub(triphthong, r"""{\1}""", text)
    # out = re.sub(triphthongsep, r"""\1§\2§\3""", out)
    # out = re.sub(diphthong, r"""{\1}""", out)
    # out = re.sub(diphthongsep, r"""\1§\2""", out)
    # out = re.sub(r"""[{}]""", "", out)

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
    for i in range(len(syllables)-1):
        s = syllables[i]
        next_s = syllables[i+1]
        if s.strip() in special_tokens.values():
            cleaned_syllables.append(s)
        elif next_s in special_tokens.values() and s[-1] in toned_vowels.keys():
            cleaned_syllables.append(s)
        elif special_tokens['WORD_SEP'] in s:
#            sub_syls = s.split(special_tokens['WORD_SEP'])
            cleaned_syllables.append(s)
        else: 
            cleaned_s = ''.join([toned_vowels[c] if c in toned_vowels.keys() else c for c in s])
            cleaned_syllables.append(cleaned_s)
    cleaned_syllables.append(syllables[-1])
    return cleaned_syllables


def syllabify_verse_prettify(verse, special_tokens, tone_tagger, synalepha=True):
    syllables = syllabify_verse(verse, special_tokens, tone_tagger, synalepha)
    syllables = [ prettify_text(s, special_tokens).strip() for s in syllables ]
    syllables = [ s for s in syllables if s != '' ]
    return syllables


def syllabify_verse(verse, special_tokens, tone_tagger, synalepha=True):
    
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

    if synalepha:
        syllables = _apply_synalepha(syllables, special_tokens)

    # removing usless tones
    syllables = remove_tone(syllables, special_tokens)
#    print('after',syllables)

    return syllables

if __name__ == "__main__":

    print(special_tokens)

    with open("divina_commedia_accent_cleaned.txt","r") as f:
        divine_comedy = f.read()


    divine_comedy_list = divine_comedy.split("\n")

    divine_comedy_list = [ line for line in divine_comedy_list if line.strip() not in special_tokens.values() ]

    tone_tagger = ToneTagger()
    count = 0
    for line in divine_comedy_list[:50]:
        syllables = syllabify_verse(line, special_tokens, tone_tagger)
#        print(syllables)
        syllables = [ syl for syl in syllables if syl != special_tokens['WORD_SEP'] ]
        syllables = [ syl for syl in syllables if syl != special_tokens['END_OF_VERSO'] ]
        syllables = [ syl.replace(special_tokens['WORD_SEP'], ' ') for syl in syllables ]

        size = len(syllables)

        if line.strip() not in special_tokens.values():
#            print("\n"+line)
#            if size < 10 or size > 12:
            if size != 11:
                print(line.replace(special_tokens['WORD_SEP'], '').replace(special_tokens['END_OF_VERSO'], ''))
                print(size, '-'.join(syllables))
                print()
                count+=1

    print(str(count)+'/'+str(len(divine_comedy_list)) + " verses still wrong")
