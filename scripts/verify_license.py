"""
 Copyright (C) 2023 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import sys
import argparse

def verify_file_has_license(file):
    with open(file, 'r') as in_file:
        contents = in_file.read(400)
        if "SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception" not in contents:
            raise Exception(f"{file} does not contain a license!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+', default=[])
    args = parser.parse_args()
    for file in args.files:
        verify_file_has_license(file)


if __name__ == "__main__":
    sys.exit(main())
