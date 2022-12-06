import os
import re
import torch

def extract_adapters(dir):
    pattern = 'em_score=(\d.\d*)'
    checkpoint_path = None
    max_em = 0
    exists = False
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f) and os.path.splitext(f)[1] == '.ckpt' and not 'AdapterWeights' in f:
            em = float(re.search(pattern, filename).group(1))
            if em > max_em:
                max_em = em
                checkpoint_path = f
                output_path = checkpoint_path.replace('.ckpt', '_AdapterWeights.ckpt')
                if os.path.isfile(output_path):
                    exists = True
                else:
                    exists = False
    print(f"checkpoint path = {checkpoint_path}")

    if not exists:
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'kadapter' not in key:
                del state_dict[key]
            else:
                # new_key = key.replace('enc_kadapter.', '')
                new_key = key.replace('model.enc_kadapter.', '')
                state_dict[new_key] = state_dict.pop(key)

        torch.save(state_dict, output_path)
    return output_path
