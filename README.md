# Articulatory to Acoustic Mapping
## Data Preparation
Input data: ema->articulatory data; laryngography data->voicing information. These two information will be concatenated into a single matrix of size rowX23. 23 will be the features.
Ouput data: textgrids containing phonetic segmentation information. This will be done by the ```mfa``` aligner, a command line tool.
## Model Training 
This will be a classification model learned in a supervised way. 
## Running 
```
bash rif.sh
```
```
conda activate aligner
````
```
mfa validate ./corpus english_mfa english_mfa
```
```
mfa align ./corpus english_mfa english_mfa ./corpus/textgrids
```
