# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:36:34 2023

@author: belie
"""

import ema2proba as ema
from os import path, listdir, environ

from importlib.resources import files

def get_repo_path():
    return files("ema2proba").parent.parent

data_dir = get_repo_path()
corpus_dir = path.join(data_dir, "ema2proba", "corpus")

lines = []
speaker_dirs = []
speaker_ids = []
for speaker in ["male", "female"]:
    if speaker == "male":
        speaker_name = "msak0_v1.1"
    elif speaker == "female":
        speaker_name = "fsew0_v1.1"
    speaker_dir = path.join(data_dir, speaker_name)
    speaker_dirs.append(speaker_dir)
    speaker_ids.append(speaker_name)
    list_file = path.join(corpus_dir, speaker_name + ".rlist")
    
    ema.write_list_to_file(list_file, [speaker_dir], [speaker_name])

list_file = path.join(corpus_dir, "fsew0_v1.1_msak0_v1.1.rlist")
ema.write_list_to_file(list_file, speaker_dirs, speaker_ids)


