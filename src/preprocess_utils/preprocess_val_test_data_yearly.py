import json
import csv
import copy

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
        query = row['query'].replace('_X_', '<extra_id_0>')
        res = [index, row['date'], query, ';'.join(answers)]
        datasets[date].append(res)
        ids_to_answers[date][str(index)] = answers

    for year in datasets: 
        dataset_len = len(datasets[year])
        for i in range(dataset_len-dataset_len % 32, dataset_len):
            del ids_to_answers[year][str(i)]
        datasets[year] = datasets[year][:dataset_len-dataset_len%32]
        with open(f"data/templama/templama_{version}_{year}.csv", "w") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            for row in datasets[year]:
                w.writerow(row)

        with open(f"data/templama/templama_{version}_{year}_answers.json", "w", encoding='utf-8') as f:
            json.dump(ids_to_answers[year], f, ensure_ascii=False)

        with open(f"data/templama/templama_{version}_{year}_prefixed.csv", "w") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["id", "date", "input", "output"])
            for row in datasets[year]:
                temp = copy.deepcopy(row)
                temp[2] = f"In {year}, " + temp[2]
                w.writerow(temp)

    dataset = []
    ids_to_answers_full = {}
    for year in datasets: 
        if int(year) > 2019:
            break
        for row in datasets[year]:
            index = len(dataset)
            row_copy = row.copy()
            row_copy[2] = f"In {year}, " + row[2]
            row_copy[0] = index
            dataset.append(row_copy)
            ids_to_answers_full[index] = row_copy[3].split(';')

    with open(f"data/templama/templama_{version}_full_answers.json", "w", encoding='utf-8') as f:
        json.dump(ids_to_answers_full, f, ensure_ascii=False)
    
    with open(f"data/templama/templama_{version}_full_prefixed.csv", "w") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["id", "date", "input", "output"])
        for row in dataset:
            w.writerow(row)

if __name__ == "__main__":
    main()