import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# My observations on the similarities between cat, apple, monkey, and banana:
# The words cat and monkey have high similarity - likely because both are animals. 
# Likewise, banana and apple also show high levels of similarity, because both are types of fruit. 
# Cat has low similarity with either fruit word (apple and banana). This is because there is little to link these words to a cat. 
# However, monkey has high similarity with banana (but lower with apple). Possibly because monkeys are known to love bananas!
# Words compared with themselves (e.g. monkey/monkey) all get 1. Unsurprisng, since they are identical words! 

sentence_to_compare = "Why is my cat on the car"

sentences = ["Where did my dog go", 
             "Hello, there is my car",
             "I\'ve lost my car in my car", 
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# My own examples
word4 = nlp("parrot")
word5 = nlp("pear")
word6 = nlp("mouse")
word7 = nlp("cheese")

# Notes on running example file. 
# When run with "en_core_web_sm", the example file gave a warning that it may not give useful similarity judgements.
# This did not happen when I ran the file with the "en_core_web_md" - the file ran without any issues here.  
# This is likely because is a larger model (has word vectors loaded and is more suited to give useful similarity judgements.