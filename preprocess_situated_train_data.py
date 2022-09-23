import json
import csv

def main():
    with open("qa_data/temp.train.jsonl") as f:
        json_list = list(f)
    data = [json.loads(s) for s in json_list]
    # print(data)
    datasets = {}
    years = ['2018-', '2019+']
    for year in years:
        datasets[year] = []
    max_answer_len = 0
    max_question_len = 0
    min_year = None
    max_year = None
    for i in range(len(data)):
        row = data[i]
        date = row['date'].split()[-1]
        if int(date) < 2019:
            dataset_version = '2018-'
        else:
            dataset_version = '2019+'
        index = len(datasets[dataset_version])

        if min_year is None or min_year > int(date):
            min_year = int(date)
        if max_year is None or max_year < int(date):
            max_year = int(date)

        query = row['edited_question']
        answer = row['answer'][0]
        res = [index, date, query, answer]
        datasets[dataset_version].append(res)

        max_question_len = max(max_question_len, len(query.split()))
        max_answer_len = max(max_answer_len, len(answer.split()))
    print(f"max answer length = {max_answer_len}")
    print(f"max query length = {max_question_len}")
    print(f"min year = f{min_year}")
    print(f"max year = f{max_year}")

    for year in datasets: 
        with open(f"data/situatedqa/sqa_train_{year}.csv", "w") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            for row in datasets[year]:
                w.writerow(row)

    for year in datasets: 
        with open(f"data/situatedqa/sqa_train_{year}_prefixed.csv", "w") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            for row in datasets[year]:
                w.writerow(row)

    with open(f"data/situatedqa/sqa_train_full.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["id", "date", "input", "output"])
        for year in datasets: 
            for row in datasets[year]:
                w.writerow(row)

    with open(f"data/situatedqa/sqa_train_full_prefixed.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["id", "date", "input", "output"])
        for year in datasets: 
            for row in datasets[year]:
                w.writerow(row)


if __name__ == "__main__":
    main()