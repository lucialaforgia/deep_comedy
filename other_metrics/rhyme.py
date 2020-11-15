# coding=utf-8

import re
import syllabification as s

# Rhyme scoring and extraction module. Exploits informations about accents, syllables and heuristics to perform
# the difficult task of determining if two words form a rhyme.

# Computes a rhyming score between two words.
def rhyme_score(w1, w2):
    if w1 == "" or w2 == "": # One of the two words is missing.
        return 0

    pw1 = s.prettify(w1, True)  # preserving accents.
    pw2 = s.prettify(w2, True)
    ppw1 = s.prettify(w1, False)  # removing accents.
    ppw2 = s.prettify(w2, False)
    accent1 = _locate_accent(pw1)
    accent2 = _locate_accent(pw2)

    if accent1 == 0 and accent2 == 0: # Difficult case: no accent is known. Heuristic match.
        out = _heuristic_rhyme(w1, w2)
    elif accent1 == accent2: # Trivial case: both accents in the same position.
        out = _match_syllable(ppw1[accent1:], ppw2[accent1:])
    elif accent1 != 0 and accent2 == 0: # Trivial case: accent1 known.
        out = _match_syllable(ppw1[accent1:], ppw2[accent1:])
    elif accent2 != 0 and accent1 == 0: # Trivial case: accent2 known.
        out = _match_syllable(ppw1[accent2:], ppw2[accent2:])
    else: # Trivial case: both accents are known, but in different positions.
        out = _match_syllable(ppw1[accent1:], ppw2[accent2:])

    return out

# Determines if a word is tronca (accent on the last syllable). Exact cases: word ending with an accented letter (morì) or word ending with a consonant (mangiàr).
# Heuristic: NO other words are considered tronche since the majority of Italian words are piane (accent on the second to last syllable) or sdrucciole (third to last).
def is_tronca(word):
    consonant = re.compile(r"""[bcdfghlmnprstvz]""")
    accentlastsyl = re.compile(r""".*#[^#]*[àèéìóòù][^#]*""")
    w = s.prettify(word, True)
    out = False

    if w == "": # The "word" was actually composed by punctuation only.
        out = False
    elif consonant.match(w[-1]):
        out = True
    else:
        sw = s.syllabify_word(w)
        if sw.count("#") == 0:
            out = True
        elif accentlastsyl.match(sw):
            out = True
        else:
            out = False

    return out

# Not used:
# def _is_piana(word): # Most common case.
#     return not (_is_tronca(word) or _is_sdrucciola(word))
#
# def _is_sdrucciola(word): # Detected only if the accent is marked.
#     accentlastsyl = re.compile(r""".*[àèéìóòù].*#.*#.*""") # The accent is marked and followed by at least two hashes.
#     return accentlastsyl.match(s.syllabify_word(s.prettify(word, True)))

# Returns the accent position FROM THE END of the word (eg. mangiò -> -1, dormìre -> -3).
# NOTE: prettification is done by the caller, since it could change accent position.
def _locate_accent(word):
    accent = re.compile(r"""[àèéìóòù]""")
    match = accent.search(word)
    if match:
        pos = match.start()
    else:
        pos = len(word)

    return pos - len(word)

# Determines a rhyming score if the two words don't have accents.
def _heuristic_rhyme(w1, w2):
    pw1 = s.prettify(w1, False)
    pw2 = s.prettify(w2, False)

    sw1 = s.syllabify_word(pw1).split("#")
    sw2 = s.syllabify_word(pw2).split("#")

    # Approximate match:
    if is_tronca(w1) and is_tronca(w2): # Both words are tronche: match only the last syllable from the vowel.
        out = _match_syllable(sw1[-1], sw2[-1])
    else: # Both words are piane: match exactly the last syllable and the last-but-one from the vowel.
        ssw1 = "".join(sw1[-2:])
        ssw2 = "".join(sw2[-2:])
        out = _match_syllable(ssw1, ssw2)

    return out

# Computes a score based on how many letters match from the end of the two strings, up to the last vowel of the first vocoid (eg. "men#te" vs. "can#te" tries to match ente and ante, computing a score of 0.75, while "iuo#la" vs. "suo#la" tries to match ola and ola, computing a score of 1.0).
# HEURISTIC: since no accent is known, the match is as PERMISSIVE as possible (ie. matches from the LAST vowel of a diphthong). This rhymes correctly "quivi/sorgivi" (while a restrictive heuristic wouldn't).
# The computed score is the sum of all matching characters (truncated at the first difference), weighted exponentially with the distance from the putative beginning of the rhyme.
# As such, it's a score which can scale well on different matching lengths (eg. "più" and "fu" have a score similar to "frangente" and "assolutamente"), at the expenses of not having a "natural" meaning which could be easier to threshold.
def _match_syllable(s1, s2):
    lastvowel = re.compile(r"""[aeiou](?![aeiou])""") # Inside a syllable vowels can only be together, so only the NEXT character needs to be checked.

    match1 = lastvowel.search(s1)
    match2 = lastvowel.search(s2)

    if match1 and match2:
        ss1 = s1[match1.start():]
        ss2 = s2[match2.start():]

        # maxlength = len(ss2) if len(ss1) < len(ss2) else len(ss1) # The two lengths could be different.
        minlength = len(ss1) if len(ss1) < len(ss2) else len(ss2)
        out = 0.0
        a = (ss1 if len(ss1) < len(ss2) else ss2)[::-1] # reverse.
        b = (ss2 if len(ss1) < len(ss2) else ss1)[::-1]

        i = 0
        while (i < minlength) and (a[i] == b[i]): # Iterate only over the shared part of the string.
            out += 2.0 ** (minlength - i)
            i += 1
        out /= 2 ** minlength
    else:
        out = 0.0

    return out
