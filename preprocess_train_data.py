import json
import csv

def main():
    # with open("data/templama_train.json") as f:
    #     lines = f.read().splitlines()

    # with open("data/templama_train.json", "w") as f:
    #     for line in lines:
    #         f.write(line + ",\n")

    with open("data/templama/templama_train.json") as f:
        data = json.load(f)
    datasets = {}
    years = list(range(2010, 2021))
    for year in years:
        datasets[str(year)] = []
    max_answer_len = 0
    max_question_len = 0
    for i in range(len(data)):
        row = data[i]

        date = row['date']
        index = len(datasets[date])
        query = row['query'].replace('_X_', '<extra_id_0>')
        answer = f"<extra_id_0> {row['answer'][0]['name']} <extra_id_1>"
        res = [index, date, query, answer]
        datasets[date].append(res)

        max_question_len = max(max_question_len, len(query.split()))
        max_answer_len = max(max_answer_len, len(answer.split()))
    print(f"max answer length = {max_answer_len}")
    print(f"max query length = {max_question_len}")

    for year in datasets: 
        with open(f"data/templama/templama_train_{year}.csv", "w") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            for row in datasets[year]:
                w.writerow(row)
        
        with open(f"data/templama/templama_train_{year}_prefixed.csv", "w") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            for row in datasets[year]:
                row_copy = row.copy()
                row_copy[2] = f"In {year}, " + row[2]
                w.writerow(row_copy)

    with open(f"data/templama/templama_train_full_prefixed.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["id", "date", "input", "output"])
        for year in datasets: 
            if int(year) > 2018:
                break
            for row in datasets[year]:
                row_copy = row.copy()
                row_copy[2] = f"In {year}, " + row[2]
                w.writerow(row_copy)


if __name__ == "__main__":
    main()