
import argparse
from tokenizers import BertWordPieceTokenizer
from pathlib import Path

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-dir", required=True,
                      help="Location of pre-training text files.")
  args = parser.parse_args()

  paths = list([str(x) for x in Path(args.corpus_dir).rglob("*.[tT][xX][tT]")])
  print(f'Nrof files: {len(paths)}')

  # Initialize a tokenizer
  tokenizer = BertWordPieceTokenizer(
    lowercase=True
  )

  # Customize training
  tokenizer.train(
    files=paths,
    vocab_size=30_000,
    min_frequency=10,
  )

  # Save files to disk
  tokenizer.save_model(".", "vocab.txt")


if __name__ == "__main__":
  main()
