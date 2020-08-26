
import argparse
from util import utils
import os
import tensorflow.compat.v1 as tf
import json


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-dir", required=True,
                      help="Location of pre-training text files.")
  parser.add_argument("--output-dir", required=True,
                      help="Where to write out the tfrecords.")
  args = parser.parse_args()

  utils.rmkdir(args.output_dir)

  fnames = sorted(tf.io.gfile.listdir(args.corpus_dir))

  for file_no, fname in enumerate(fnames):
    input_file = os.path.join(args.corpus_dir, fname)
    output_file = os.path.join(args.output_dir, fname)
    with tf.io.gfile.GFile(input_file) as fi:
      with tf.io.gfile.GFile(output_file) as fo:
        for line in fi:
          line = line.strip()
          line = json.loads(line)
          text = ' '.join(line['article_text'])
          fo.write(text + '\n')


if __name__ == "__main__":
  main()
