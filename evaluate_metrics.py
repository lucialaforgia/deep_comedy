import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'other_metrics') )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re

import other_metrics.metrics as om
from other_metrics.ngrams_plagiarism import ngrams_plagiarism

#from dante_by_char.text_processing import clean_comedy, prettify_text, special_tokens
#from dante_by_syl.text_processing import clean_comedy, prettify_text, special_tokens
#from dante_by_word.text_processing import clean_comedy, prettify_text, special_tokens
from dante_by_rev_syl.text_processing import clean_comedy, prettify_text, special_tokens


if __name__ == '__main__':

    generated_canto_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_canto.txt") 
    divine_comedy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

    with open(generated_canto_file,"r", encoding='utf-8') as f:
        generated_canto = f.read()

    with open(divine_comedy_file,"r", encoding='utf-8') as f:
        divine_comedy = f.read()

    divine_comedy = clean_comedy(divine_comedy, special_tokens)
    divine_comedy = prettify_text(divine_comedy, special_tokens)
    
    generated_canto = generated_canto.replace('CANTO', '')

    print(divine_comedy)
#    print(generated_canto)


    evaluation_results = {}
    evaluation_results = om.eval(generated_canto, verbose=False, synalepha=True, permissive=False, rhyme_threshold=1.0)

    ngrams_plagiarism_score = ngrams_plagiarism(generated_canto, divine_comedy, n=4)

    evaluation_results['Plagiarism'] =  ngrams_plagiarism_score

    for k, v in evaluation_results.items():
        print('{}: {}'.format(k, v))









