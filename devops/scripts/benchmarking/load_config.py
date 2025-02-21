from common import Configuration, Validate
import sys

# TODO better frontend / use argparse
if __name__ == "__main__":

    def usage_and_exit():
        print(f"Usage: {sys.argv[0]} <path to /devops> [config | constants]")
        print(
            "Generate commands to export configuration options/constants as an environment variable."
        )
        exit(1)

    if len(sys.argv) != 3:
        usage_and_exit()

    if not Validate.filepath(sys.argv[1]):
        print(f"Not a valid filepath: {sys.argv[1]}", file=sys.stderr)
        exit(1)
    # If the filepath provided passed filepath validation, then it is clean
    sanitized_filepath = sys.argv[1]

    # Load configuration
    config = Configuration(sanitized_filepath)
    if sys.argv[2] == "config":
        print(config.export_shell_configs())
    elif sys.argv[2] == "constants":
        print(config.export_shell_constants())
    else:
        usage_and_exit()
