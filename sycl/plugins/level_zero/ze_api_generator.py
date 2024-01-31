import re
import sys


def camel_to_snake(src):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", src).lower()


def snake_to_camel(src):
    temp = src.split("_")
    return "".join(x.title() for x in temp)


def extract_ze_apis(header):
    """
    Emit file with contents of
    _ZE_API(api_name, api_domain, cb, param_type)
    """
    api = open("ze_api.def", "w")

    matches = re.finditer(
        r"typedef struct _ze_([_a-z]+)_callbacks_t\n\{\n([a-zA-Z_;\s\n]+)\n\} ze_([_a-z]+)_callbacks_t;",
        header,
    )

    for match in matches:
        api_domain = snake_to_camel(match.group(1))
        for l in match.group(2).splitlines():
            parts = l.split()
            api_match = re.match(r"ze_pfn([a-zA-Z]+)Cb_t", parts[0])
            api_name_tail = api_match.group(1)
            api_name = "ze" + api_name_tail

            param_type = "ze_" + camel_to_snake(api_name_tail) + "_params_t"

            cb = "pfn" + api_name_tail.replace(api_domain, "") + "Cb"

            api.write(
                "_ZE_API({}, {}, {}, {})\n".format(api_name, api_domain, cb, param_type)
            )

    api.close()


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        header = f.read()
        extract_ze_apis(header)
