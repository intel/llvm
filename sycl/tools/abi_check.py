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
  symbols = re.findall(r"Symbol \{[\n\s\w:\.\-\(\)]*\}",
                       output.decode().strip())
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
    readobj_out = subprocess.check_output([get_llvm_bin_path()+"llvm-readobj",
                                           "-t", target_path])
    symbols = parse_readobj_output(readobj_out)
    out.write("\n".join(symbols))


def compare_results(ref_records, records):
  missing_records = []

  for record in ref_records:
    if record in records:
      records.remove(record)
    else:
      missing_records.append(record)
  return (missing_records, records)


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

def check_vtable(ref_path, target_path):
  with open(ref_path, "r") as ref:
    ref_records = []
    for line in ref:
      if not line.startswith('#') and line.strip():
        ref_records.append(line.strip())

    cxxdump_out = subprocess.check_output([get_llvm_bin_path()+"llvm-cxxdump",
                                           target_path])
    records = cxxdump_out.decode().strip().split('\n')

    missing_records, new_records = compare_results(ref_records, records)

    correct_return = True
    if missing_records:
      correct_return = False
      print('The following records are missing from the new object file:\n')
      print("\n".join(missing_records))

    if new_records:
      correct_return = False
      print('The following records are new to the object file:\n')
      print("\n".join(new_records))

    if not correct_return:
      sys.exit(-1)


def dump_vtable(target_path, output):
  with open(output, "w") as out:
    cxxdump_out = subprocess.check_output([get_llvm_bin_path()+"llvm-cxxdump",
                                           target_path])
    out.write(cxxdump_out.decode())


def main():
  parser = argparse.ArgumentParser(description='ABI checker utility.')
  parser.add_argument('--mode', type=str,
                      choices=['check_symbols', 'dump_symbols',
                               'check_vtable', 'dump_vtable'],
                      help='ABI checking mode', required=True)
  parser.add_argument('--reference', type=str, help='Reference ABI dump')
  parser.add_argument('--output', type=str, help='Output for dump modes')
  parser.add_argument('target_library', type=str)

  args = parser.parse_args()

  if args.mode == 'check_symbols':
    check_symbols(args.reference, args.target_library)
  elif args.mode == 'dump_symbols':
    dump_symbols(args.target_library, args.output)
  elif args.mode == 'check_vtable':
    check_vtable(args.reference, args.target_library)
  elif args.mode == 'dump_vtable':
    dump_vtable(args.target_library, args.output)


if __name__ == "__main__":
  main()

