
import argparse
from pathlib import Path
from tqdm import tqdm


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-dir", required=True,
                      help="Location of pre-training text files.")
  args = parser.parse_args()

  count = 0
  with open('files.txt', 'w') as f:
    for path in tqdm(Path(args.corpus_dir).rglob('*.txt')):
      if path.is_file():
        f.write(str(path)+'\n')
        count += 1

  print(f'Nrof files: {count}')


if __name__ == "__main__":
  main()
