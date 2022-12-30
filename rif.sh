#!/bin/sh
mkdir ./wav
cp -r ./mocha_timit/$1_v1.1/*.wav ./wav
echo "copying finished!"
for i in $(seq -f "%03g" 1 460)
do
    sph2pipe -f rif ./wav/$1_$i.wav ./wav/test_$i.wav
done
echo "The SPH wav files have already been converted to RIF wav files"
find ./wav -name '$1_*.wav' -delete
for file in ./wav/test_*.wav
do
    mv "$file" "${file/test_/$1_}"
done
echo "renaming finished!"
cp -r ./wav/*.wav ./corpus
echo "corpus built!"
rm -r ./wav
echo "cleaning finished!"
