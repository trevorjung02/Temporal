import json
import csv

def main():
    version = 'val'
    # with open(f"data/templama/templama_{version}.json") as f:
    #     lines = f.read().splitlines()

    # with open(f"data/templama/templama_{version}.json", "w") as f:
    #     for line in lines:
    #         f.write(line + ",\n")

    with open(f"data/templama/templama_{version}.json") as f:
        data = json.load(f)
    dataset = []
    ids_to_answers = {}
    for i in range(len(data)):
        row = data[i]
        answers = []
        for answer in row['answer']:
            answers.append(answer['name'])
        date = row['date']
        if int(date) > 2018:
            continue
        query = row['query'].replace('_X_', '<extra_id_0>')
        res = [i, date, query, ';'.join(answers)]
        dataset.append(res)
        ids_to_answers[str(i)] = answers

    with open(f"data/templama/templama_{version}_full.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["id", "date", "input", "output"])
        for row in dataset:
            w.writerow(row)

    with open(f"data/templama/templama_{version}_full_answers.json", "w", encoding='utf-8') as f:
        json.dump(ids_to_answers, f, ensure_ascii=False)
    
    with open(f"data/templama/templama_{version}_full_prefixed.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["id", "date", "input", "output"])
        for row in dataset:
            row[2] = f"In {row[1]}, " + row[2]
            w.writerow(row)


if __name__ == "__main__":
    main()