import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'other_metrics') )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re

import other_metrics.metrics as om
from other_metrics.ngrams_plagiarism import ngrams_plagiarism

import our_metrics.metrics as m

#from dante_by_char.text_processing import clean_comedy, prettify_text, special_tokens
#from dante_by_syl.text_processing import clean_comedy, prettify_text, special_tokens
#from dante_by_word.text_processing import clean_comedy, prettify_text, special_tokens


from dante_by_rev_syl.text_processing import clean_comedy, prettify_text, special_tokens, remove_all_punctuation



def evaluate_other_metrics(generated_canto, divine_comedy):

    generated_canto_list = generated_canto.split("\n")
    generated_canto_list = [line.strip() for line in generated_canto_list if line != 'CANTO']
    generated_canto = "\n".join(generated_canto_list)

    divine_comedy_list = divine_comedy.split("\n")
    divine_comedy_list = [line.strip() for line in divine_comedy_list if line != 'CANTO']
    divine_comedy = "\n".join(divine_comedy_list)


    evaluation_results = {}
    evaluation_results = om.eval(generated_canto, verbose=False, synalepha=True, permissive=False, rhyme_threshold=1.0)

    ngrams_plagiarism_score = ngrams_plagiarism(generated_canto, divine_comedy, n=4)

    evaluation_results['Plagiarism'] =  ngrams_plagiarism_score

    return evaluation_results


def evaluate_our_metrics(generated_canto, divine_comedy):

    generated_canto_list = generated_canto.split("\n")
    generated_canto_list = [line.strip() for line in generated_canto_list if line != 'CANTO']
    generated_canto = "\n".join(generated_canto_list)

    divine_comedy_list = divine_comedy.split("\n")
    divine_comedy_list = [line.strip() for line in divine_comedy_list if line != 'CANTO']
    divine_comedy = "\n".join(divine_comedy_list)

    evaluation_results = {}

    evaluation_results = m.eval(generated_canto, synalepha=True)

    return evaluation_results

if __name__ == '__main__':

    generated_canto_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_canto.txt") 
    evaluation_results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.txt") 
    divine_comedy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

    with open(generated_canto_file,"r", encoding='utf-8') as f:
        generated_canto = f.read()

    with open(divine_comedy_file,"r", encoding='utf-8') as f:
        divine_comedy = f.read()

    divine_comedy = clean_comedy(divine_comedy, special_tokens)
    divine_comedy = prettify_text(divine_comedy, special_tokens)
    divine_comedy = remove_all_punctuation(divine_comedy)
    
    evaluation_results = {}
#    print(divine_comedy)
#    print(generated_canto)


    ###### Test our metrics on divine comedy ########
    print('\nEVALUATING OUR METRICS ON WHOLE DIVINE COMEDY...')
    evaluation_results = evaluate_our_metrics(divine_comedy, divine_comedy)

    f = open(evaluation_results_file, "w", encoding='utf-8')
    f.write('EVALUATION RESULTS:\n')
    print('\nOUR METRICS ON WHOLE DIVINE COMEDY:')
    f.write('\nOUR METRICS ON WHOLE DIVINE COMEDY:\n')
    for k, v in evaluation_results.items():
        print('{}: {}'.format(k, v))
        f.write('{}: {}\n'.format(k, v))
    f.close()
    ##################################################
    


    # Test our metrics on one canto of divine comedy #
    n_canto = 1
    print('\nEVALUATING OUR METRICS ON {} CANTO OF DIVINE COMEDY...'.format(n_canto))
    divine_comedy_canto_list = divine_comedy.split('CANTO')
    divine_comedy_canto_list = [line.strip() for line in divine_comedy_canto_list if line != '']

    evaluation_results = evaluate_our_metrics(divine_comedy_canto_list[n_canto-1], divine_comedy)

    f = open(evaluation_results_file, "a", encoding='utf-8')
    print('\nOUR METRICS ON {} CANTO OF DIVINE COMEDY:'.format(n_canto))
    f.write('\nOUR METRICS ON {} CANTO OF DIVINE COMEDY:\n'.format(n_canto))
    for k, v in evaluation_results.items():
        print('{}: {}'.format(k, v))
        f.write('{}: {}\n'.format(k, v))
    f.close()
    
    ##################################################
    


    ## Evaluation other metrics on generated canto ### 
    print('\nEVALUATING OTHER METRICS ON GENERATED CANTO...')
    evaluation_results = evaluate_other_metrics(generated_canto, divine_comedy)

    f = open(evaluation_results_file, "a", encoding='utf-8')
    print('\nOTHER METRICS ON GENERATED CANTO:')
    f.write('\nOTHER METRICS ON GENERATED CANTO:\n')
    for k, v in evaluation_results.items():
        print('{}: {}'.format(k, v))
        f.write('{}: {}\n'.format(k, v))
    f.close()


    ##################################################

    ### Evaluation our metrics on generated canto ####
    print('\nEVALUATING OUR METRICS ON GENERATED CANTO...')
    evaluation_results = evaluate_our_metrics(generated_canto, divine_comedy)

    f = open(evaluation_results_file, "a", encoding='utf-8')
    print('\nOUR METRICS ON GENERATED CANTO:')
    f.write('\nOUR METRICS ON GENERATED CANTO:\n')
    for k, v in evaluation_results.items():
        print('{}: {}'.format(k, v))
        f.write('{}: {}\n'.format(k, v))
    f.close()

    ##################################################

    
