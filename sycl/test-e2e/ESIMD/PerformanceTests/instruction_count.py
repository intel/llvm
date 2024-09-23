import os
import re
import sys


def main(directory, max_count, target_file):
    total_count = 0
    pattern = re.compile(r"//\.instCount (\d+)")

    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist.")
        sys.exit(1)

    try:
        target_found = False
        for root, dirs, files in os.walk(directory):
            for file in files:
                print("File: ", file)
                if file.endswith(".asm") and re.search(target_file + "$", file):
                    target_found = True
                    print("Checking file: ", file)
                    try:
                        with open(os.path.join(root, file), "r") as f:
                            for line in f:
                                match = pattern.search(line)
                                if match:
                                    total_count += int(match.group(1))
                    except IOError:
                        print(f"Failed to open file: {file}")
                        sys.exit(2)
                    break
        if not target_found:
            raise FileNotFoundError(f"Target file {target_file} was not found")
    except FileNotFoundError as e:
        print(e)
        sys.exit(3)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(4)

    print("Instruction count: ", total_count)
    if total_count > max_count * 1.05:  # 5% tolerance
        print(
            f"Instruction count exceeded threshold. Baseline is {max_count}. 5% threshold is {max_count * 1.05}. Current is {total_count}."
        )
        print(
            f"Percentage difference is {((total_count - max_count) / max_count) * 100}%, the tolerance is 5%."
        )
        sys.exit(1)
    elif total_count < max_count * 0.95:  # ask for baseline to be updated
        print(
            f"Instruction count is below the 95% threshold. Baseline is {max_count}. Current is {total_count}."
        )
        print(
            f"Percentage difference is {((total_count - max_count) / max_count) * 100}%"
        )
        print("Please update the baseline.")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])