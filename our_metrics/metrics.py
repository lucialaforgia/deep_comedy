import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from dante_by_rev_syl.syllabification import syllabify_verse
from dante_by_rev_syl.text_processing import special_tokens
#from dante_by_tonedrev_syl.syllabification import syllabify_verse #dovremo usare questo appena scritto :)



def eval(generated_canto, synalepha=True, dieresis=True):

    n_strophes = get_n_strophes(generated_canto)
    n_well_formed_terzine = get_well_formed_terzine(generated_canto)
    avg_verse_len = get_average_verse_length(generated_canto)
    last_single_verse = is_last_single_verse_present(generated_canto)


    return {
        'Number of strophes': n_strophes ,
        'Number of well formed terzine': n_well_formed_terzine,
        'Last single verse': last_single_verse,
        'Average verse length': avg_verse_len,
        'Hendecasyllabicness by tone': 0,
        'Average rhymeness': 0,
    }
def get_terzine(generated_canto):
    generated_canto = generated_canto.strip()
    return generated_canto.split('\n\n')


def is_last_single_verse_present(generated_canto):
    terzine = get_terzine(generated_canto)
    return len(terzine[-1].split('\n')) == 1

def get_n_strophes(generated_canto):
    return len(get_terzine(generated_canto))

def get_well_formed_terzine(generated_canto):
    terzine = generated_canto.split('\n\n')
    n_well_formed_terzine = 0
    for t in terzine:
        if len(t.split('\n')) == 3:
            n_well_formed_terzine+=1
    return n_well_formed_terzine


def get_average_verse_length(generated_canto):
    #remove empty line and extract verses
    generated_canto_list = generated_canto.split("\n")
    verses = [line.strip() for line in generated_canto_list if line != '']
#    generated_canto_list = [line.strip() for line in generated_canto_list if line != '']
#    generated_canto = "\n".join(generated_canto_list)
#    verses = generated_canto.split('\n')
    syls_counter = 0
    for v in verses:
        syls_counter+=len(syllabify_verse(v, special_tokens))
    return syls_counter/len(verses)
