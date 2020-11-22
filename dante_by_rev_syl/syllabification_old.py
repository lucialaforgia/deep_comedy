import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyphen
import re
import string
from dante_by_rev_syl.text_processing import remove_punctuation

special_tokens = {
    'START_OF_CANTO'   : '<start_of_canto>',
    'END_OF_CANTO'     : '<end_of_canto>',
    'START_OF_TERZINA' : '<start_of_terzina>',
    'END_OF_TERZINA'   : '<end_of_terzina>',
    'END_OF_VERSO'     : '<end_of_verso>',
    'WORD_SEP'         : '<word_sep>'
}


def syllabify_verse(verse, special_tokens, synalepha=True, dieresis=True):
    
    if verse in special_tokens.values():
        return [verse]

    dic = pyphen.Pyphen(lang='it_IT',left=1, right=2, cache=True)

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

#    syllables = [ remove_punctuation(s) for s in syllables ]
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
                if pre_syl[-1] in vowels and next_syl[0] in vowels :
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
