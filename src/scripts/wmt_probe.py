import json

def main(): 
    year = 2007
    path = f"wmt_counts/{year}.json"
    with open(path) as f:
        counts = json.load(f)
    print(counts['University'])

if __name__ == "__main__":
    main()