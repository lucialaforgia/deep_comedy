import re
import string
import os

special_tokens = {
    'START_OF_CANTO'   : '<start_of_canto>',
    'END_OF_CANTO'     : '<end_of_canto>',
    'START_OF_TERZINA' : '<start_of_terzina>',
    'END_OF_TERZINA'   : '<end_of_terzina>',
    'END_OF_VERSO'     : '<end_of_verso>',
    'WORD_SEP'         : '<word_sep>'
}

def replace_chars(divine_comedy):
    # replace rare and strange chars
#    divine_comedy = divine_comedy.replace("Ä", "A")
#    divine_comedy = divine_comedy.replace("ä", "a")
#    divine_comedy = divine_comedy.replace("Ë", "E")
#    divine_comedy = divine_comedy.replace("ë", "è")
#    divine_comedy = divine_comedy.replace("Ï", "I")
#    divine_comedy = divine_comedy.replace("ï", "i")
#    divine_comedy = divine_comedy.replace("Ö", "O")
#    divine_comedy = divine_comedy.replace("ö", "o")
#    divine_comedy = divine_comedy.replace("Ü", "U")
#    divine_comedy = divine_comedy.replace("ü", "u")

#    divine_comedy = divine_comedy.replace("é", "è")
#    divine_comedy = divine_comedy.replace("ó", "ò")

    divine_comedy = divine_comedy.replace("•", "")

    divine_comedy = divine_comedy.replace("(", "")
    divine_comedy = divine_comedy.replace(")", "")
    divine_comedy = divine_comedy.replace("(", "-")
    divine_comedy = divine_comedy.replace(")", "-")
    divine_comedy = divine_comedy.replace("’", "\'")
    divine_comedy = divine_comedy.replace("‘", "\'")
    divine_comedy = divine_comedy.replace("”", "\"")
    divine_comedy = divine_comedy.replace("“", "\"")
    divine_comedy = divine_comedy.replace("—", "")
    divine_comedy = divine_comedy.replace("«", "\"")
    divine_comedy = divine_comedy.replace("»", "\"")

    return divine_comedy


def get_corpus(divine_comedy):
    # remove text before and after the divine comedy
    start = divine_comedy.find("INFERNO") 
    divina_end = "l'amor che move il sole e l'altre stelle."
    end = divine_comedy.find(divina_end)+len(divina_end)
    divine_comedy = divine_comedy[start:end]
    return divine_comedy

def remove_cantica_title(divine_comedy):
    # remove cantica title
    divine_comedy = divine_comedy.replace("INFERNO", "")
    divine_comedy = divine_comedy.replace("PURGATORIO", "")
    divine_comedy = divine_comedy.replace("PARADISO", "")
    return divine_comedy

def remove_all_punctuation(divine_comedy):
    # remove all punctuation
#    divine_comedy = re.sub(r'\'',' ', divine_comedy)
    divine_comedy = re.sub('[%s]'% re.escape(string.punctuation),' ', divine_comedy )
    divine_comedy = re.sub(r' +',' ', divine_comedy)
    return divine_comedy

def remove_punctuation(divine_comedy):
    # remove punctuation
    divine_comedy = re.sub('[%s]'% re.escape('!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'),'', divine_comedy)
    divine_comedy = re.sub(r' +',' ', divine_comedy)
    return divine_comedy

def remove_empty_lines(divine_comedy):
    # remove unuseful empty lines and spaces
    divine_comedy_list = divine_comedy.split("\n")
    divine_comedy_list = [line.strip() for line in divine_comedy_list]
    divine_comedy = "\n".join(divine_comedy_list)
    divine_comedy = re.sub('\n{3,}', '\n', divine_comedy)
    divine_comedy_list = divine_comedy.split("\n")
    if divine_comedy_list[0] == "":
        divine_comedy_list = divine_comedy_list[1:]
    divine_comedy = "\n".join(divine_comedy_list)
    return divine_comedy

def add_special_tokens(divine_comedy, special_tokens):
    # add special tokens to text

    divine_comedy = divine_comedy + "\n" + special_tokens['END_OF_CANTO']

    divine_comedy_list = divine_comedy.split("\n")

    divine_comedy_list = [special_tokens['END_OF_CANTO']+ "\n"+ special_tokens['START_OF_CANTO']+"\n"+special_tokens['START_OF_TERZINA'] \
            if re.search(r'Canto [A-Z]+',line) else line for line in divine_comedy_list]
    divine_comedy = "\n".join(divine_comedy_list)
    divine_comedy_list = divine_comedy.split("\n")

    del divine_comedy_list[0] # to remove the first end_of_canto token
    #modify the separatores in verse
    divine_comedy = "\n".join(divine_comedy_list)
    divine_comedy = divine_comedy.replace(" ", " "+special_tokens['WORD_SEP']+" ")
    #modify the separatores between verses
    divine_comedy_list = divine_comedy.split("\n")
    divine_comedy_list = [line if line == "" or line in special_tokens.values() else line + ' ' + special_tokens['END_OF_VERSO'] for line in divine_comedy_list]

    temp = []
    for i, line in enumerate(divine_comedy_list):
        if line != "":
            temp.append(line)
        else:
            if i >= 3 and divine_comedy_list[i-1].endswith(special_tokens['END_OF_VERSO']) \
                    and divine_comedy_list[i-2].endswith(special_tokens['END_OF_VERSO']) \
                    and divine_comedy_list[i-3].endswith(special_tokens['END_OF_VERSO']):
                temp.append(special_tokens['END_OF_TERZINA'])
            if i < len(divine_comedy_list) - 3 and divine_comedy_list[i+1].endswith(special_tokens['END_OF_VERSO']) \
                    and divine_comedy_list[i+2].endswith(special_tokens['END_OF_VERSO']) \
                    and divine_comedy_list[i+3].endswith(special_tokens['END_OF_VERSO']):
                temp.append(special_tokens['START_OF_TERZINA'])
    divine_comedy_list = temp
    divine_comedy = "\n".join(divine_comedy_list)

    return divine_comedy

def remove_newlines(divine_comedy):
    divine_comedy_list = divine_comedy.split("\n")
    divine_comedy = "".join(divine_comedy_list)
#    divine_comedy = " ".join(divine_comedy_list)
    return divine_comedy

def prettify_text(text, special_tokens):
    text = remove_newlines(text)
    text = text.replace(special_tokens['END_OF_VERSO'], "\n")
    text = text.replace(special_tokens['START_OF_TERZINA'], "")
    text = text.replace(special_tokens['END_OF_TERZINA'], "\n")
    text = text.replace(special_tokens['START_OF_CANTO'], "CANTO\n")
    text = text.replace(special_tokens['END_OF_CANTO'], "\n")
    text = text.replace(special_tokens['WORD_SEP'], " ")
    text = re.sub(r' +',' ', text)
    # text_list = text.split("\n")
    # text_list = [line.strip() for line in text_list]
    # text = "\n".join(text_list)

    return text



def clean_comedy(divine_comedy, special_tokens):
    divine_comedy = replace_chars(divine_comedy)
    divine_comedy = get_corpus(divine_comedy)
    divine_comedy = remove_cantica_title(divine_comedy)
    divine_comedy = remove_punctuation(divine_comedy)
    divine_comedy = remove_empty_lines(divine_comedy)
    divine_comedy = add_special_tokens(divine_comedy, special_tokens)

#    divine_comedy = remove_newlines(divine_comedy)

    divine_comedy = divine_comedy.lower()

    return divine_comedy

def save_comedy_cleaned(divine_comedy):
    divine_comedy_file_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "divina_commedia_cleaned.txt") 

#   save cleaned divine comedy in a new file
    with open(divine_comedy_file_output,"w") as f:
        f.write(divine_comedy)


if __name__ == "__main__":
    #read divine comedy from file
    divine_comedy_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "divina_commedia", "divina_commedia_accent_UTF-8.txt")
    
    with open(divine_comedy_file,"r") as f:
        divine_comedy = f.read()

    divine_comedy = clean_comedy(divine_comedy, special_tokens)

    save_comedy_cleaned(divine_comedy)

    print(divine_comedy[:1000])
    print(special_tokens)
    print(prettify_text(divine_comedy[:1000], special_tokens))




