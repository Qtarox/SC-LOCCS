import numpy as np

# load the key and plaintext in global variables
DATA_PATH='E:\\data\\'
REAL_KEY=open(DATA_PATH+str('key_.txt'),'r').readline()
PTS=[]
with open(DATA_PATH+str('pt_.txt'),'r') as f:
    lines=f.readlines()
    for line in lines:
        PTS.append(line)

def main():
    
    return 0

if __name__=='__main__':
    main()