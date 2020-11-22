import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import requests
import json
import csv
import string
import re
import pyphen
import spacy
import time

from dante_by_word.text_processing import clean_comedy, prettify_text, special_tokens, remove_all_punctuation
#from utils import save_vocab, load_vocab

working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')

divine_comedy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "divina_commedia", "divina_commedia_accent_UTF-8.txt") 

with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)
divine_comedy = remove_all_punctuation(divine_comedy)


vocab = sorted(list(set(divine_comedy.split())))

# Path where the vocab will be saved
accents_vocab_file = os.path.join(working_dir, 'accent_vocab.csv')

print("Vocab size:", len(vocab))

api_url = "https://api.dictionaryapi.dev/api/v2/entries/it/{word}"

headers = { 
    "Cache-Control": "no-cache",
    "Pragma": "no-cache"
}


pyphen_dic = pyphen.Pyphen(lang='it_IT',left=1, right=1, cache=True)

nlp = spacy.load("it_core_news_sm")

f_accents_vocab = open(accents_vocab_file, "w")

writer = csv.writer(f_accents_vocab, delimiter='\t', lineterminator='\n')
row = ['word', 'searched_word', 'found_word', 'syllables', 'toned_word', 'mode']
writer.writerow(row)

sl_t = 0.5

to_lemmatize = []

for w in vocab:
    searched_word = w
    print('Searching '+ searched_word+'...')
    try:
        json_response = requests.get(api_url.format(word=searched_word), headers=headers)
        dic = json.loads(json_response.content)
    except:
        to_lemmatize.append(w)
        continue
    if 'No Definitions Found' not in str(json_response.content):
        for d in dic:
            try:
                found_word = d['word']
                if searched_word == found_word:
                    try:
                        syls = d['phonetic'].replace('·', '-')
                        toned_word = d['phonetic'].replace('·', '')
                        row = [w, searched_word, found_word, syls, toned_word, 'ok']
                        print('{}\t{}\t{}\t{}\t{}\t{}'.format(*row))
                        writer.writerow(row)
                    except:
                        try:
                            syls = d['phonetics'][0]['text'].replace('·', '-')
                            toned_word = d['phonetics'][0]['text'].replace('·', '')
                            row = [w, searched_word, found_word, syls, toned_word, 'ok']
                            print('{}\t{}\t{}\t{}\t{}\t{}'.format(*row))
                            writer.writerow(row)
                        except:
                            to_lemmatize.append(w)
#                            syls = pyphen_dic.inserted(w)
#                            toned_word = 'no_toned_word'
#                            row = [w, searched_word, found_word, syls, toned_word, 'no_tone_pyphen_split']
                else:
                    to_lemmatize.append(w)
    #                row = ['_error', 'search '+to_search, 'found '+found_word]
    #                writer.writerow(row)
            except:
                to_lemmatize.append(w)
#                row = [w, '_error', '_error', '_error', '_error']
#                writer.writerow(row)
#                print('{}\t{}\t{}\t{}\t{}'.format(*row))
            break
    else:
        to_lemmatize.append(w)
    f_accents_vocab.flush()
    time.sleep(sl_t)

to_search = []

print("START LEMMATIZATION...")
for w in to_lemmatize:
    print('Searching '+ w+'...')
    doc = nlp(w)
    if len(doc) == 0:
#        row = [w, 'no_lemma', 'not_found', '_error', '_error']
#        print('{}\t{}\t{}\t{}\t{}'.format(*row))
#        writer.writerow(row)
        to_search.append(w)
        continue
    lemma = doc[0].lemma_

    try:
        json_response = requests.get(api_url.format(word=lemma), headers=headers)
        dic = json.loads(json_response.content)
    except:
        to_search.append(w)
        continue

    if 'No Definitions Found' not in str(json_response.content):
        for d in dic:
            try:
                found_word = d['word']
                if lemma == found_word:
                    try:
                        syls = d['phonetic'].replace('·', '-')
                        toned_word = d['phonetic'].replace('·', '')
                        row = [w, lemma, found_word, syls, toned_word, 'toned_lemma_ok']
                        writer.writerow(row)
                        print('{}\t{}\t{}\t{}\t{}\t{}'.format(*row))
                    except:
                        try:
                            syls = d['phonetics'][0]['text'].replace('·', '-')
                            toned_word = d['phonetics'][0]['text'].replace('·', '')
                            row = [w, lemma, found_word, syls, toned_word, 'toned_lemma_ok']
                            writer.writerow(row)
                            print('{}\t{}\t{}\t{}\t{}\t{}'.format(*row))
                        except:
                            to_search.append(w)
#                            syls = pyphen_dic.inserted(w)
#                            toned_word = 'no_toned_word'
#                            row = [w, lemma, found_word, syls, toned_word, 'no_tone_pyphen_split']
                else:
                    to_search.append(w)
            except:
                to_search.append(w)
#                row = [w, '_error', '_error', '_error', '_error']
#                writer.writerow(row)
#                print('{}\t{}\t{}\t{}\t{}'.format(*row))
            break
    else:
        to_search.append(w)
    time.sleep(sl_t)
    f_accents_vocab.flush()



to_split = []
print('START FINAL SEARCH...')

for w in to_search:
    searched_word = w
    print('Searching '+ searched_word+'...')

    try:
        json_response = requests.get(api_url.format(word=searched_word), headers=headers)
        dic = json.loads(json_response.content)
    except:
        to_split.append(w)
        continue
    if 'No Definitions Found' not in str(json_response.content):
        for d in dic:
            try:
                found_word = d['word']
                try:
                    syls = d['phonetic'].replace('·', '-')
                    toned_word = d['phonetic'].replace('·', '')
                    row = [w, searched_word, found_word, syls, toned_word, 'toned_found']
                    print('{}\t{}\t{}\t{}\t{}\t{}'.format(*row))
                    writer.writerow(row)
                except:
                    try:
                        syls = d['phonetics'][0]['text'].replace('·', '-')
                        toned_word = d['phonetics'][0]['text'].replace('·', '')
                        row = [w, searched_word, found_word, syls, toned_word, 'toned_found']
                        print('{}\t{}\t{}\t{}\t{}\t{}'.format(*row))
                        writer.writerow(row)
                    except:
                        to_split.append(w)
#                        syls = pyphen_dic.inserted(w)
#                        toned_word = 'no_toned_word'
            except:
                to_split.append(w)
#                row = [w, searched_word, found_word, '_error', '_error']
#                writer.writerow(row)
#                print('{}\t{}\t{}\t{}\t{}'.format(*row))
            break
    else:
        to_split.append(w)
#        row = [w, '_error', 'not_found', '_error', '_error']
#        writer.writerow(row)
#        print('{}\t{}\t{}\t{}\t{}'.format(*row))

    f_accents_vocab.flush()
    time.sleep(sl_t)


print('START SPLITTING...')

for w in to_split:
    syls = pyphen_dic.inserted(w)
    row = [w, 'not_searched', 'not_founded', syls, 'no_toned_word', 'no_tone_pyphen_split']
    print('{}\t{}\t{}\t{}\t{}\t{}'.format(*row))
    writer.writerow(row)
    f_accents_vocab.flush()

