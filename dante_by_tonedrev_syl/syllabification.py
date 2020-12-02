import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyphen
import re
import string
from dante_by_tonedrev_syl.text_processing import remove_punctuation, special_tokens
from dante_by_tonedrev_syl.tone import ToneTagger


# Splits a single word into syllables.
def syllabify_word(text):
    return _perform_final_splits(_perform_initial_splits(text))

# Performs the first (easy and unambiguous) phase of syllabification.
def _perform_initial_splits(text):
    return _split_hiatus(_split_dieresis(_split_double_cons(_split_multiple_cons(text))))

# Performs the second (difficult and heuristic) phase of syllabification.
def _perform_final_splits(text):
    cvcv = r"""(?i)([bcdfglmnpqrstvz][aeiouàèéìóòùÈËÏ]+)([bcdfglmnpqrstvz]+[aeiouàèéìóòùÈËÏ]+)"""
    vcv = r"""(?i)([aeiouàèéìóòùÈËÏ]+)([bcdfglmnpqrstvz]+[aeiouàèéìóòùÈËÏ]+)"""
    vv = r"""(?i)(?<=[aeiouàèéìóòùÈËÏ])(?=[aeiouàèéìóòùÈËÏ])"""

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
    dieresis = re.compile(r"""(?i)([äëïöüËÏ](?=[aeiou])|[aeiou](?=[äëïöüËÏ]))""")
    return "#".join(dieresis.sub(r"""\1@""", text).split("@"))

# Splits SURE hiatuses only. Ambiguous ones are heuristically considered diphthongs.
def _split_hiatus(text):
    hiatus = re.compile(r"""(?i)([aeoàèòóé](?=[aeoàèòóé])|[rbd]i(?=[aeou])|tri(?=[aeou])|[ìù](?=[aeiou])|[aeiou](?=[ìù]))""")
    return "#".join(hiatus.sub(r"""\1@""", text).split("@"))

# Prevents splitting of diphthongs and triphthongs.
def _clump_diphthongs(text):
    diphthong = r"""(?i)(i[,.;:"“”«»?—'`‘’\s]*[aeouàèéòóù]|u[,.;:"“”«»?—'`‘’\s]*[aeioàèéìòó]|[aeouàèéòóù][,.;:"“”«»?—'`‘’\s]*i|[aeàèé][,.;:"“”«»?—'`‘’\s]*u)"""
    diphthongsep = r"""(\{.[,.;:"“”«»?—'`‘’\s]*)(.\})"""
    triphthong = r"""(?i)(i[àèé]i|u[àòó]i|iu[òó]|ui[èéà])""" #nostra
    triphthongsep = r"""(\{.)(.)(.\})"""

    out = re.sub(triphthong, r"""{\1}""", text)
    out = re.sub(triphthongsep, r"""\1§\2§\3""", out)
    out = re.sub(diphthong, r"""{\1}""", out)
    out = re.sub(diphthongsep, r"""\1§\2""", out)
    out = re.sub(r"""[{}]""", "", out)

    return out

def syllabify_verse(verse, special_tokens, tone_tagger, synalepha=True, dieresis=True):
    
    if verse in special_tokens.values():
        return [verse]

    words = [ w for w in verse.split() ]

    list_of_syllables = [ syllabify_word(tone_tagger.tone(w)).split('#') if w.strip() not in special_tokens.values() else [w] for w in words ] 
    
    # syllables = [ dic.inserted(w) for w in words ]
    # #["l'a-mor", 'che', 'mo-ve', 'il', 'so-le', 'e', "l'al-tre", 'stel-le.']
    # #print(syllables)

    ## [['nel'], ['<word_sep>'], ['mez', 'zo'], ['<word_sep>'], ['del'], ['<word_sep>'], ['cam', 'min'], ['<word_sep>'], ['di'], ['<word_sep>'], ['no', 'stra'], ['<word_sep>'], ['vi', 'ta'], ['<end_of_verso>']]
    
    syllables = []
    # tone_topreserve_positions = []
    for s in list_of_syllables:
        syllables+=s


# #    print("syl", syllables)
# #    print(len(syllables))
    if synalepha:
        vowels = "ÁÀAaàáÉÈEeèéIÍÌiíìOoóòÚÙUuúù'"
#        vowels = "ÁÀAaàáÉÈEeèéIÍÌiíìOoóòÚÙUuúù"

        result = [syllables[0]]
        i = 1
        while i < (len(syllables) - 1):
            if syllables[i] == special_tokens['WORD_SEP']:
                pre_syl = syllables[i-1]
                next_syl = syllables[i+1]
                if pre_syl[-1] in vowels and next_syl[0] in vowels: # aggiungere is_dittongo
                    result.append(result[-1] + syllables[i] + next_syl)
                    del result[-2]
                    i += 1
                else:
                    result.append(syllables[i])               
            else:
                result.append(syllables[i])
            i+=1
        result.append(syllables[-1])
        syllables = result



    
# #    print("synalepha", syllables)
# #    print(len(syllables))

#     if dieresis:
#         diereis_vowels = "ÄäËëÏïÖöÜü"
#         result = []
#         for sy in syllables:
#             for v in diereis_vowels:
#                 if v in sy:
#                     index = sy.index(v)
#                     result.append(sy[:index+1])
#                     result.append(sy[index+1:])
#                     break
#             else:
#                 result.append(sy)
#         syllables = result

# #    print("dieresis", syllables)
# #    print(len(syllables))
#     syllables.append(special_tokens.get('END_OF_VERSO'))
    return syllables

if __name__ == "__main__":

    print(special_tokens)

    with open("divina_commedia_accent_cleaned.txt","r") as f:
        divine_comedy = f.read()


    divine_comedy_list = divine_comedy.split("\n")

    divine_comedy_list = [ line for line in divine_comedy_list if line.strip() not in special_tokens.values() ]

    tone_tagger = ToneTagger()
    count = 0
    for line in divine_comedy_list[:]:
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
