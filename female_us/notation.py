import os
import textgrids

script = []
for transcription in sorted(os.listdir("./textgrids")):
    if transcription.endswith('.TextGrid'):
        script.append(transcription)
with open('notation.txt', 'w') as f:
    for t in script:
        grid = textgrids.TextGrid('./textgrids/' + t)
        for phon in grid['phones']:
            l = phon.text.transcode()
            f.write(l)
            f.write('\t')

        f.write('\n')
        
with open('mocha-timit.txt') as reader, open('mocha-timit.txt', 'r+') as writer:
  for line in reader:
    if line.strip():
      writer.write(line)
  writer.truncate()
