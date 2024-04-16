import subprocess

proc = subprocess.Popen(['./sampling_io'], stdin=subprocess.PIPE, stdout = subprocess.PIPE, text=True, shell=True)

while True:
    value = input('python Please input m1 mv2 v: ')
    value = value + '\n'
    proc.stdin.write(value)
    proc.stdin.flush()
    result = proc.stdout.readline()
    print(result)