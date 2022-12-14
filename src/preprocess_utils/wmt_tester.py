from preprocess_wmt_train_data import mask_sentence_one_ss_random_span
import spacy

def main():
    sentences = ['The Nielsen analysis shows Mr. Romney has most frequently advertised on the Microsoft Network and Bell South Internet home pages, as well as on The Drudge Report and FoxNews.com.']
    print(sentences)
    spacy_ner = spacy.load("en_core_web_sm", disable=['tagger', 'parser','tok2vec', 'attribute_ruler', 'lemmatizer'])
    spacy_res = spacy_ner.pipe(sentences)
    mean_length = 3
    mask_pct = 0.15
    print(spacy_res)

    for s in spacy_res:
        print(s)
        sentence, train_answer, val_answer = mask_sentence_one_ss_random_span(sentence, mean_length, mask_pct)
        print(sentence)
        print(train_answer)
        print(val_answer)
