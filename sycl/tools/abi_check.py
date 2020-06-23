#!/usr/bin/env python
#
# Compare symbols that are exported from the binary against a known snapshot.
# Return an error if there are new or missing symbols in the library.
#
import argparse
import os
import subprocess
import sys
import re


def get_llvm_bin_path():
  if 'LLVM_BIN_PATH' in os.environ:
    return os.environ['LLVM_BIN_PATH']
  return ""


def match_symbol(sym_binding, sym_type, sym_section):
  if sym_binding is None or sym_type is None or sym_section is None:
    return False
  if not sym_type.group() == "Function":
    return False
  if not (sym_binding.group() == "Global" or sym_binding.group() == "Weak"):
    return False
  if not sym_section.group() == ".text":
    return False
  return True


def parse_readobj_output(output):
  if os.name == 'nt':
    return re.findall(r"(?<=Symbol: )[\w\_\?\@\$]+", output.decode().strip())
  else:
    symbols = re.findall(r"Symbol \{[\n\s\w:\.\-\(\)]*\}",
                         output.decode().strip())
    parsed_symbols = []
    for sym in symbols:
      sym_binding = re.search(r"(?<=Binding:\s)[\w]+", sym)
      sym_type = re.search(r"(?<=Type:\s)[\w]+", sym)
      sym_section = re.search(r"(?<=Section:\s)[\.\w]+", sym)
      name = re.search(r"(?<=Name:\s)[\w]+", sym)
      if match_symbol(sym_binding, sym_type, sym_section):
        parsed_symbols.append(name.group())
    return parsed_symbols


def dump_symbols(target_path, output):
  with open(output, "w") as out:
    readobj_out = subprocess.check_output([get_llvm_bin_path()+"llvm-readobj",
                                           "-t", target_path])
    symbols = parse_readobj_output(readobj_out)
    symbols.sort()
    out.write("\n".join(symbols))


def compare_results(ref_records, records):
  missing_records = set(ref_records).difference(set(records))
  new_records = set(records).difference(set(ref_records))

  return (missing_records, new_records)


# Dumps symbols from from binary at target_path and compares with a snapshot
# stored at ref_path. Reports new and absent symbols (if there are any).
def check_symbols(ref_path, target_path):
  with open(ref_path, "r") as ref:
    ref_symbols = []
    for line in ref:
      if not line.startswith('#') and line.strip():
        ref_symbols.append(line.strip())

    readobj_out = subprocess.check_output([get_llvm_bin_path()+"llvm-readobj",
                                           "-t", target_path])
    symbols = parse_readobj_output(readobj_out)

    missing_symbols, new_symbols = compare_results(ref_symbols, symbols)

    correct_return = True
    if missing_symbols:
      correct_return = False
      print('The following symbols are missing from the new object file:\n')
      print("\n".join(missing_symbols))

    if new_symbols:
      correct_return = False
      print('The following symbols are new to the object file:\n')
      print("\n".join(new_symbols))

    if not correct_return:
      sys.exit(-1)


def main():
  parser = argparse.ArgumentParser(description='ABI checker utility.')
  parser.add_argument('--mode', type=str,
                      choices=['check_symbols', 'dump_symbols'],
                      help='ABI checking mode', required=True)
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

