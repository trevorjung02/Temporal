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
    datasets = {}
    ids_to_answers = {}
    years = list(range(2010, 2021))
    for year in years:
        datasets[str(year)] = []
        ids_to_answers[str(year)] = {}

    for i in range(len(data)):
        row = data[i]
        answers = []
        for answer in row['answer']:
            answers.append(answer['name'])
        date = row['date']
        index = len(datasets[date])
        res = [index, i, row['date'], row['query'].replace('_X_', '<extra_id_0>'), ';'.join(answers)]
        datasets[date].append(res)
        ids_to_answers[date][str(index)] = answers

    for year in datasets: 
        with open(f"data/templama/templama_{version}_{year}.csv", "w") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "original", "date", "input", "output"])
            for row in datasets[year]:
                w.writerow(row)

        with open(f"data/templama/templama_{version}_{year}_answers.json", "w", encoding='utf-8') as f:
            json.dump(ids_to_answers[year], f, ensure_ascii=False)


if __name__ == "__main__":
    main()