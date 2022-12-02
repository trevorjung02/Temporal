import json
import csv
import copy

def main():
    version ='val'

    with open("qa_data/temp.dev.jsonl") as f:
        json_list = list(f)
    data = [json.loads(s) for s in json_list]
    # print(data)
    datasets = {}
    ids_to_answers = {}
    years = ['2018-', '2019+']
    for year in years:
        datasets[year] = []
        ids_to_answers[year] = {}

    for i in range(len(data)):
        row = data[i]
        date = row['date'].split()[-1]
        if int(date) < 2019:
            dataset_version = '2018-'
        else:
            dataset_version = '2019+'
        index = len(datasets[dataset_version])
        answers = []
        for answer in row['answer']:
            answers.append(answer)

        query = row['edited_question']
        res = [index, date, query, ';'.join(answers)]
        datasets[dataset_version].append(res)
        ids_to_answers[dataset_version][str(index)] = answers

    for year in datasets: 
        with open(f"data/situatedqa/sqa_{version}_{year}.csv", "w") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            for row in datasets[year]:
                w.writerow(row)

        with open(f"data/situatedqa/sqa_{version}_{year}_answers.json", "w", encoding='utf-8') as f:
            json.dump(ids_to_answers[year], f, ensure_ascii=False)

        with open(f"data/situatedqa/sqa_{version}_{year}_prefixed.csv", "w") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            for row in datasets[year]:
                w.writerow(row)


if __name__ == "__main__":
    main()