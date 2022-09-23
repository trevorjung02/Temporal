import json
import csv
import copy

def main():
    version ='val'

    with open("qa_data/temp.dev.jsonl") as f:
        json_list = list(f)
    data = [json.loads(s) for s in json_list]
    # print(data)
    dataset = []
    ids_to_answers = {}
    years = ['2018-', '2019+']

    for i in range(len(data)):
        row = data[i]
        date = row['date'].split()[-1]
        index = len(dataset)
        answers = []
        for answer in row['answer']:
            answers.append(answer)

        query = row['edited_question']
        res = [index, date, query, ';'.join(answers)]
        dataset.append(res)
        ids_to_answers[str(index)] = answers

    with open(f"data/situatedqa/sqa_{version}_full.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["id", "date", "input", "output"])
        for row in dataset:
            w.writerow(row)

    with open(f"data/situatedqa/sqa_{version}_full_answers.json", "w", encoding='utf-8') as f:
        json.dump(ids_to_answers, f, ensure_ascii=False)
    
    with open(f"data/situatedqa/sqa_{version}_full_prefixed.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["id", "date", "input", "output"])
        for row in dataset:
            w.writerow(row)


if __name__ == "__main__":
    main()