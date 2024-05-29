import os
import re
import sys


def main(directory, max_count, target_file=None):
    total_count = 0
    pattern = re.compile(r"//\.instCount (\d+)")
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".asm") and (target_file is None or file == target_file):
                print('Checking file: ', file)
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        match = pattern.search(line)
                        if match:
                            total_count += int(match.group(1))

    print("Total instruction count: ", total_count)
    if total_count > max_count * 1.1:  # 10% tolerance
        print("Instruction count exceeded threshold")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)