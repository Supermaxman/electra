
import argparse
from util import utils
import os
import json


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-dir", required=True,
                      help="Location of pre-training text files.")
  parser.add_argument("--output-dir", required=True,
                      help="Where to write out the tfrecords.")
  args = parser.parse_args()

  utils.rmkdir(args.output_dir)

  fnames = sorted(os.listdir(args.corpus_dir))

  for file_no, fname in enumerate(fnames):
    input_file = os.path.join(args.corpus_dir, fname)
    output_file = os.path.join(args.output_dir, fname)
    print(f'Writing {fname}...')
    with open(input_file, 'r') as fi:
      with open(output_file, 'w') as fo:
        for line in fi:
          line = line.strip().replace('\n', '')
          if line:
            line = json.loads(line)
            article_text = ' '.join(line['article_text']).replace('\n', '')
            abstract_text = ' '.join(line['abstract_text']).replace('<S>', '').replace('\n', '')
            # empty lines to split docs
            text = f'{abstract_text} {article_text} \n\n'
            fo.write(text)


if __name__ == "__main__":
  main()
