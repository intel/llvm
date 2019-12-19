import argparse
import os
import subprocess
import sys
import re


def get_llvm_bin_path():
  if 'LLVM_BIN_PATH' in os.environ:
    return os.environ['LLVM_BIN_PATH']
  return ""


def match_symbol(sym_binding, sym_type):
  if sym_binding is None or sym_type is None:
    return False
  if not sym_type.group() == "Function":
    return False
  if not sym_binding.group() == "Global":
    return False
  return True


def parse_readobj_output(output):
  symbols = re.findall(r"Symbol \{[\n\s\w:\.\-\(\)]*\}", output.decode().strip())
  parsed_symbols = []
  for sym in symbols:
    sym_binding = re.search(r"(?<=Binding:\s)[\w]+", sym)
    sym_type = re.search(r"(?<=Type:\s)[\w]+", sym)
    name = re.search(r"(?<=Name:\s)[\w]+", sym)
    if match_symbol(sym_binding, sym_type):
      parsed_symbols.append(name.group())
  return parsed_symbols


def dump_symbols(target_path, output):
  with open(output, "w") as out:
    readobj_out = subprocess.check_output([get_llvm_bin_path()+"llvm-readobj", "-t", target_path])
    symbols = parse_readobj_output(readobj_out)
    out.write("\n".join(symbols))


def check_symbols(ref_path, target_path):
  with open(ref_path, "r") as ref:
    ref_symbols = []
    for line in ref:
      if not line.startswith('#') and line.strip():
        ref_symbols.append(line.strip())

    readobj_out = subprocess.check_output([get_llvm_bin_path()+"llvm-readobj", "-t", target_path])
    symbols = parse_readobj_output(readobj_out)

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

