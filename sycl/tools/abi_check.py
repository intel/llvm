import argparse
import os
import subprocess
import sys


def get_llvm_bin_path():
  if 'LLVM_BIN_PATH' in os.environ:
    return os.environ['LLVM_BIN_PATH']
  return ""


def parse_objdump_output(output):
  lines = output.decode().strip().split("\n")
  lines = lines[4:]
  table = list(map(lambda line: line.split(" ")[-1].strip(), lines))
  return table


def dump_symbols(target_path, output):
  with open(output, "w") as out:
    objdump_out = subprocess.check_output([get_llvm_bin_path()+"llvm-objdump", "-t", target_path])
    symbols = parse_objdump_output(objdump_out)
    out.write("\n".join(symbols))


def check_symbols(ref_path, target_path):
  with open(ref_path, "r") as ref:
    ref_symbols = []
    for line in ref:
      if not line.startswith('#'):
        ref_symbols.append(line.strip())

    objdump_out = subprocess.check_output([get_llvm_bin_path()+"llvm-objdump", "-t", target_path])
    symbols = parse_objdump_output(objdump_out)

    missing_symbols = []

    for symbol in ref_symbols:
      if symbol in symbols:
        symbols.remove(symbol)
      else:
        missing_symbols.append(symbol)

    correct_return = True
    if missing_symbols:
      correct_return = False
      print('The following symbols are missing from the new object file:\n' + "\n".join(missing_symbols))

    if symbols:
      correct_return = False
      print('The following symbols are new to the object file:\n' + "\n".join(symbols))

    if not correct_return:
      sys.exit(-1)


def main():
  parser = argparse.ArgumentParser(description='ABI checker utility.')
  parser.add_argument('--mode', type=str, choices=['check_symbols', 'dump_symbols'], help='ABI checking mode', required=True)
  parser.add_argument('--reference', type=str, help='Reference ABI dump')
  parser.add_argument('--output', type=str, help='Output for dump modes')
  parser.add_argument('target_library', type=str)

  args = parser.parse_args()

  if args.mode == 'check_symbols':
    check_symbols(args.reference, args.target_library)
  elif args.mode == 'dump_symbols':
    dump_symbols(args.target_library, args.output)


if __name__ == "__main__":
  main()

