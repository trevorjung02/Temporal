import torch

def main():
    ckpt_path = ".pl_auto_save.ckpt"
    ckpt = torch.load(ckpt_path)
    print(ckpt)

if __name__ == "__main__":
    main()