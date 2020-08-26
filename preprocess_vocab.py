
import argparse
from tokenizers import BertWordPieceTokenizer


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-path", required=True,
                      help="Location of pre-training text files.")
  parser.add_argument("--vocab-path", required=True,
                      help="Where to write out the tfrecords.")
  args = parser.parse_args()

  paths = [args.corpus_path]

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
  tokenizer.save_model(".", "vocab.json")


if __name__ == "__main__":
  main()
