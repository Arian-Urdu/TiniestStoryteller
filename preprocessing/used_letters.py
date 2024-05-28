import re

# normalise text
def process_string(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'(m|re|s|d|ll|ve)", r" '\1", text)

    # Remove punctuation marks except apostrophes
    text = re.sub(r'[^\w\s\']', '', text)

    return text

# create list of words in text
def string2list(text):
    text = process_string(text)
    words = text.split()

    return words

# ---------------------------------------------------------------------------
# list all used letters in dataset

used_letters = {'a', 'b', 'c'}

for story in filtered_dataset['text']:
    word_list = string2list(story)
    for word in word_list:
        for letter in word:
            used_letters.add(letter)

print(used_letters)

# {'â', 'ä', "'", 'ñ', 'ã'}


forbidden_letters = {'ê', '_', 'ò', '²', 'š', 'ž', 'ª', 'á', 'ÿ', 'ï', '½', 'ä', 'æ', 'ˆ', 'ã', 'ñ', 'º', '³', '¾', '¼', 'é', 'â', 'œ', 'µ', 'î', 'è', 'å', 'ç', '¹', 'ð'}
# {'â', 'ñ', 'ä', 'ã', "'"}


# ---------------------------------------------------------------------------
# list all stories containing any of the forbidden letters

# forbidden_stories = []
#
# for story in dataset['text']:
#     processed_story = process_string(story)
#     if any(letter in processed_story for letter in forbidden_letters):
#         forbidden_stories.append([processed_story])

# print(len(forbidden_stories) / len(dataset['text']))
# approx 6%