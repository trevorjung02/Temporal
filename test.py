# import required module
import os
import re
# assign directory
directory = 'outputs/adapters_2010_2freeze_158'
 
# iterate over files in
# that directory
args.output_dir += '_' + ''.join(map(str, args.adapter_config['adapter_list']))
pattern = 'em_score=(\d.\d*)'
checkpoint_path = None
max_em = 0
for filename in os.listdir(args.output_dir):
    f = os.path.join(args.output_dir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        em = re.search(pattern, filename).group(0)
        if em > max_em:
            max_em = em
            checkpoint_path = f
args.checkpoint_path = checkpoint_path
print(args.checkpoint_path)