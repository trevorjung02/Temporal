import subprocess

command = 'sbatch myscript.sh'
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
