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

divine_comedy_words = divine_comedy.split()[:10] + []

for w in divine_comedy_words:
    print(tone_tagger.tone(w), flush=True, end=' ')

