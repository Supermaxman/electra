
import argparse
import os


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-path", required=True,
                      help="Location of pre-training text files.")
  parser.add_argument("--output-dir", required=True,
                      help="Where to write out the tfrecords.")
  parser.add_argument("--num-files", type=int, default=5,
                      help="Where to write out the tfrecords.")
  args = parser.parse_args()

  idx = 0
  fo_list = [open(os.path.join(args.output_dir, f'{i}.txt'), 'w') for i in range(args.num_files)]
  with open(args.corpus_path, 'r') as fi:
    for line in fi:
      line = line.strip()
      if line:
        fo_list[idx % args.num_files].write(line + '\n\n')
        idx += 1

  for file in fo_list:
    file.close()


if __name__ == "__main__":
  main()
