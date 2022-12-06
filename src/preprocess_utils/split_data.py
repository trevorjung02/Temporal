import json
import random

def main():
    queries = {
        'P54': 'plays for',
        'P39': 'holds the position of',
        'P108': 'works for',
        'P102': 'is a member of the',
        'P286': 'is the head coach of',
        'P69': 'attended',
        'P488': 'is the chair of',
        'P6': 'is the head of the government of',
        'P127': 'is owned by'
    }
    ending = {'P286', 'P488', 'P6'}
    datasets = {'train':[], 'val':[], 'test':[]}
    queries_by_sub = {}
    data = []
    for version in ["train", "val", "test"]:    
        with open(f"data/templama_old/templama_{version}.json") as f:
            data.extend(json.load(f))
    for i in range(len(data)):
        line = data[i]
        pattern = queries[line['relation']]
        index = line['query'].find(pattern)
        if line['relation'] not in ending:
            subject = line['query'][:index].strip()
        else:
            subject = line['query'][index + len(pattern):-1].strip()
        if subject not in queries_by_sub:
            queries_by_sub[subject] = []    
        queries_by_sub[subject].append(i)

    subjects = list(queries_by_sub.keys())
    random.shuffle(subjects)
    split = {'train': 0.2, 'val': 0.1, 'test': 0.7}
    i = 0 
    for v in datasets:
        while len(datasets[v]) < len(data) * split[v] and i < len(subjects):
            for query in queries_by_sub[subjects[i]]:
                datasets[v].append(data[query])
            i += 1
        with open(f"data/templama/templama_{v}.json", mode = 'w') as f:
            json.dump(datasets[v], f)

if __name__ == "__main__":
    main()