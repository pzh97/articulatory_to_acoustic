# Articulatory to Acoustic Mapping
## Data Preparation
Input data: ema->articulatory data; laryngography data->voicing information. These two information will be concatenated into a single matrix of size rowX23. 23 will be the features.
Ouput data: textgrids containing phonetic segmentation information. This will be done by the ```mfa``` aligner, a command line tool.
