import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion in 2018")

print(doc)
print(doc.ents)
dates = [ent for ent in doc.ents if ent.label_ == 'DATE']
print(dates)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)