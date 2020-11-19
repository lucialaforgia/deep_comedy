import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyphen
import re
import string
from dante_by_rev_syl.text_processing import remove_puctuation

special_tokens = {
    'START_OF_CANTO'   : '<start_of_canto>',
    'END_OF_CANTO'     : '<end_of_canto>',
    'START_OF_TERZINA' : '<start_of_terzina>',
    'END_OF_TERZINA'   : '<end_of_terzina>',
    'END_OF_VERSO'     : '<end_of_verso>',
    'WORD_SEP'         : '<word_sep>'
}

def is_toned_vowel(c):
    toned_vowels = "ÄäÁÀàáËëÉÈèéÏïÍÌíìÖöÓÒóòÜüÚÙúù"
    return c in toned_vowels


def is_diphthong(c1, c2):
    aeo  = "ÄäÁÀàáAaËëÉÈèéEeÖöÓÒóòOo"
    vowels = "ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu"
    if c1 in vowels and c2 =="'":
        return True
    if c2 in vowels and c1 =="'":
        return True
#    if c1 in aeo and c2 in aeo:
#        return False
#    if c1 == c2:
#        return True
    if is_toned_vowel(c1) and c2 in vowels:
        return False
    if not is_toned_vowel(c1) and c2 in vowels: #secondo me questo e' sbagliato "ae" abbiamo deciso che e' sicuramente uno iato ma qua verra' sempre considerato come dittongo
        return True
    return (c1 + c2) in ('ia', 'ie', 'iè', 'ié', 'io', 'iu', 'iú', 'iù',
                        'ua', 'uè', 'ué', 'ue', 'ui', 'uo',  'uò', 'uò',)


def is_iato(c1, c2):
    aeo  = "ÄäÁÀàáAaËëÉÈèéEeÖöÓÒóòOo"
    iu = "ÏïÍÌíìIiÜüÚÙúùUu"
    vowels = "ÄäÁÀàáAaËëÉÈèéEeÏïÍÌíìIiÖöÓÒóòOoÜüÚÙúùUu"
    if c1 in aeo and c2 in aeo:
        return True
#    if c1 in aeo and c2 in iu:
#        return True
    if is_toned_vowel(c1) and c2 in vowels:
        return True

    return False


def contains_iato(syl):
    vowels = "ÁÀAaàáÉÈEeèéIÍÌiíìOoóòÚÙUuúù'"
    for i in range(len(syl) - 1):
        c1 = syl[i]
        c2 = syl[i+1]
        if c1 in vowels and c2 in vowels and is_iato(c1, c2):
            return True
    return False


def syllabify_verse(verse, special_tokens, synalepha=True, dieresis=True):
    
    if verse in special_tokens.values():
        return [verse]

    dic = pyphen.Pyphen(lang='it_IT',left=1, right=1, cache=True)

    words = [ w  for w in verse.split() if w.strip() not in special_tokens.values() ]

    syllables = [ dic.inserted(w) for w in words ]
    #["l'a-mor", 'che', 'mo-ve', 'il', 'so-le', 'e', "l'al-tre", 'stel-le.']
    #print(syllables)

    sep = '-'+special_tokens['WORD_SEP']+'-'
    syllables = sep.join(syllables)
    #l'a-mor-<word_sep>-che-<word_sep>-mo-ve-<word_sep>-il-<word_sep>-so-le-<word_sep>-e-<word_sep>-l'al-tre-<word_sep>-stel-le.
    #print(syllables)

    syllables = syllables.split("-")
    #["l'a", 'mor', '<word_sep>', 'che', '<word_sep>', 'mo', 've', '<word_sep>', 'il', '<word_sep>', 'so', 'le', '<word_sep>', 'e', '<word_sep>', "l'al", 'tre', '<word_sep>', 'stel', 'le.']
    #print(syllables)

#    syllables = [ remove_puctuation(s) for s in syllables ]
    #["l'a", 'mor', '<word_sep>', 'che', '<word_sep>', 'mo', 've', '<word_sep>', 'il', '<word_sep>', 'so', 'le', '<word_sep>', 'e', '<word_sep>', "l'al", 'tre', '<word_sep>', 'stel', 'le']
    #print(syllables)

    syllables = [ s for s in syllables if s != ""]
    #["l'a", 'mor', '<word_sep>', 'che', '<word_sep>', 'mo', 've', '<word_sep>', 'il', '<word_sep>', 'so', 'le', '<word_sep>', 'e', '<word_sep>', "l'al", 'tre', '<word_sep>', 'stel', 'le']
    #print(syllables)

#    print("syl", syllables)
#    print(len(syllables))
    if synalepha:
        vowels = "ÁÀAaàáÉÈEeèéIÍÌiíìOoóòÚÙUuúù'"
#        vowels = "ÁÀAaàáÉÈEeèéIÍÌiíìOoóòÚÙUuúù"

        result = [syllables[0]]
        i = 1
        while i < (len(syllables) - 1):
            if syllables[i] == special_tokens['WORD_SEP']:
                pre_syl = syllables[i-1]
                next_syl = syllables[i+1]
                if pre_syl[-1] in vowels and next_syl[0] in vowels and is_diphthong(pre_syl[-1], next_syl[0]):
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







    result = []
    for syl in syllables:
        if special_tokens['WORD_SEP'] not in syl and contains_iato(syl):
            new_syl = ''

            i = 0
            while i < len(syl) - 1 : 
                c1 = syl[i]
                c2 = syl[i+1]

                if is_iato(c1, c2): # va staccato
                    new_syl+=c1
#                    print("new syl is iato", new_syl)
                    result.append(new_syl)
                    new_syl = c2
                    
                else: # non va staccato
                    new_syl+=c1
#                    new_syl+=c2
#                    print("new syl else", new_syl)

                i+=1
            result.append(new_syl)
        else:
            result.append(syl)       
    
    syllables = result













#    print("synalepha", syllables)
#    print(len(syllables))

    if dieresis:
        diereis_vowels = "ÄäËëÏïÖöÜü"
        result = []
        for sy in syllables:
            for v in diereis_vowels:
                if v in sy:
                    index = sy.index(v)
                    result.append(sy[:index+1])
                    result.append(sy[index+1:])
                    break
            else:
                result.append(sy)
        syllables = result

#    print("dieresis", syllables)
#    print(len(syllables))
    syllables.append(special_tokens.get('END_OF_VERSO'))
    return syllables

if __name__ == "__main__":

    print(special_tokens)

    with open("divina_commedia_accent_cleaned.txt","r") as f:
        divine_comedy = f.read()


    divine_comedy_list = divine_comedy.split("\n")

    divine_comedy_list = [ line for line in divine_comedy_list if line.strip() not in special_tokens.values() ]

    count = 0
    for line in divine_comedy_list[:]:
        syllables = syllabify_verse(line, special_tokens)
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
