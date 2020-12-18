import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from dante_by_tonedrev_syl.text_processing import clean_comedy, prettify_text, special_tokens, remove_all_punctuation
from dante_by_tonedrev_syl.tone import ToneTagger


working_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dante_by_tonedrev_syl')

divine_comedy_file = os.path.join(os.path.dirname(working_dir), "divina_commedia", "divina_commedia_accent_UTF-8.txt") 


with open(divine_comedy_file,"r") as f:
    divine_comedy = f.read()

divine_comedy = clean_comedy(divine_comedy, special_tokens)
divine_comedy = prettify_text(divine_comedy, special_tokens)
#divine_comedy = remove_all_punctuation(divine_comedy)

tone_tagger = ToneTagger()

print("\nMODEL: {}\n".format(tone_tagger.model_filename))

#os.makedirs(os.path.join(logs_dir, model_filename), exist_ok = True) 

# output_file = os.path.join(logs_dir, model_filename, "output.txt")
# raw_output_file = os.path.join(logs_dir, model_filename, "raw_output.txt")


divine_comedy_words = divine_comedy.split()[:10] + []

for w in divine_comedy_words:
    print(tone_tagger.tone(w), flush=True, end=' ')



# indexes = [i for i, x in enumerate(divine_comedy_verse) if x == special_tokens['END_OF_VERSO'] and i > SEQ_LENGTH]
# index_eov = np.random.choice(indexes)
# start_seq = divine_comedy_verse[index_eov - SEQ_LENGTH:index_eov]

# #print(start_seq)

# generated_text = generate_text(model_tone, special_tokens, vocab_size, char2idx, idx2char, SEQ_LENGTH, start_seq, temperature=1.0)

# #print(prettify_text(generated_text, special_tokens))


# with open(output_file,"w") as f:
#     f.write(prettify_text(generated_text, special_tokens))

# with open(raw_output_file,"w") as f:
#     f.write(generated_text)
