# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:49:14 2023

@author: belie
"""

from os import path, listdir
from .utils import get_random_string

def extract_id_from_line(lines):
    return [line.split(".")[0] for line in lines[:lines.index("\n")]]

def extract_id_from_readme(speaker_dir, list_files):    
    readmes = [l for l in list_files if l.startswith("README.")]
    if len(readmes) > 0:        
        with open(path.join(speaker_dir, readmes[0]), 'r') as f:
            lines = f.readlines()
            return extract_id_from_line(lines)
    else:
        return []
    
def get_id_from_file(file_name):
    return path.splitext(file_name)[0].split("_")[-1]

def prepare_file_list(speaker_dir, speaker_id=None):
    
    if speaker_id is None:
        speaker_id = get_random_string()
    
    list_files = listdir(speaker_dir)
    list_files.sort()
    
    idx2rej = extract_id_from_readme(speaker_dir, list_files)
    
    keys = [".ema", ".wav", ".lar", ".lab"]
    emas, audios, eggs, segs = [[l for l in list_files if l.endswith(key) and
                                 get_id_from_file(l) not in idx2rej] for 
                                key in keys]
    return [write_line(speaker_id, e, a, l, s, speaker_dir) for e, a, l, s in
             zip(emas, audios, eggs, segs)]
        
def read_file_list(file):
    lines = read_data_file(file)
    return [[l.split("\t")[n] for l in lines] for n in range(6)]

def read_data_file(file):
    with open(file, "r") as f:
        return [x.replace("\n", "") for x in f.readlines()]

def write_line(speaker_id, ema_file, audio_file, egg_file, seg_file,
               speaker_dir):
    return "\t".join(["_".join([speaker_id, get_id_from_file(ema_file)]),
                      speaker_id,
                      path.join(speaker_dir, ema_file),
                      path.join(speaker_dir, audio_file),
                      path.join(speaker_dir, egg_file),
                      path.join(speaker_dir, seg_file)]) + "\n"
    
def write_list_to_file(file, speaker_dirs, speaker_ids=None):
    lines = []
    if speaker_ids is None:
        speaker_ids = [get_random_string() for s in speaker_dirs]
    for speaker_dir, speaker_id in zip(speaker_dirs, speaker_ids):
        lines += prepare_file_list(speaker_dir, speaker_id)
    with open(file, "w") as f:
        f.writelines(lines)


