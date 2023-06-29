# USING SPACY
import spacy

nlp = spacy.load ('en_core_web_sm')

text = ("This is Python Interpretor. It helps in encoding code written by user to help the machine understand it. Delhi is the capital of India. Lucknow is the capital of Uttar Pradesh. Ram went to Nigeria for his graduation in Computer Science and Engineering.")

doc = nlp(text)

sentence = list(doc.sents)
print (sentence)

ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print (ents)

# USING NLTK
import nltk
nltk.download('words')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perpeptron_tagger')
nltk.download('state_union')

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
train_text = state_union.raw() 
sample_text = state_union.raw("This is Python Interpretor. It helps in encoding code written by user to help the machine understand it. Delhi is the capital of India. Lucknow is the capital of Uttar Pradesh. Ram went to Nigeria for his graduation in Computer Science and Engineering.")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def get_named_entity():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary = false)
            namedEnt.draw()
    except:
        pass
get_named_entity()