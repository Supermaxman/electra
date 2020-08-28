
import argparse
from tokenizers import BertWordPieceTokenizer
from pathlib import Path

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-filelist-path", required=True,
                      help="Location of pre-training text files.")
  args = parser.parse_args()

  paths = []
  with open(args.corpus_filelist_path) as f:
    for line in f:
      line = line.strip()
      if line:
        paths.append(line)

  print(f'Nrof files: {len(paths)}')

  # Initialize a tokenizer
  tokenizer = BertWordPieceTokenizer(
    lowercase=False
  )

  # Customize training
  tokenizer.train(
    files=paths,
    vocab_size=40_000,
    min_frequency=10,
  )

  # Save files to disk
  tokenizer.save_model(".", "vocab.txt")


if __name__ == "__main__":
  main()
