def main():
    with open("data/templama/templama_train_2010.csv") as f:
        lines = f.read().splitlines()

    with open("data/templama/templama_train_debug.csv", "w") as f:
        for i in range(50):
            f.write(lines[i] + "\n") 

if __name__ == "__main__":
    main()