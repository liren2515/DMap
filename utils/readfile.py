import os, sys
import pickle as pkl
import json 

def load_pkl(path):
    f = open(path, 'rb')
    data = pkl.load(f)
    f.close()
    return data

def dump_pkl(data, path):
    f = open(path, 'wb')
    pkl.dump(data, f)
    f.close()
    return 

def write_txt(lines, path):
    f = open(path, 'w')
    for line in lines:
        f.write(line+'\n')
    f.close()
    return

def read_txt(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    lines = [i.split() for i in lines]
    return lines

def write_json(dicts, path):
    with open(path, "w") as outfile: 
        json.dump(dicts, outfile, indent=4)
    return