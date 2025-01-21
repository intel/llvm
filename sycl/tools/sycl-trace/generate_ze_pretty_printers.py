import re
import sys


def camel_to_snake(src):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", src).lower()


def snake_to_camel(src):
    temp = src.split("_")
    return "".join(x.title() for x in temp)


def generate_ze_pretty_printers(header):
    """
    Emit file with contents of
    _ZE_API(api_name, api_domain, cb, param_type)
    """
    printers = open("ze_printers.def", "w")

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

            search_str = r"typedef struct _{}\n{{\n([0-9\sa-zA-Z_\*;\n]*)}}".format(
                param_type
            )
            args = re.search(search_str, header)

            args = args.group(1).replace("\n", "").strip().split(";")[:-1]

            printers.write(
                "case static_cast<uint32_t>(ZEApiKind::{}): {{\n".format(api_name)
            )
            printers.write(
                "const auto *Args = reinterpret_cast<{}*>(Data->args_data);\n".format(
                    param_type
                )
            )
            for arg in args:
                arg_name = arg.strip().split(" ")[-1].replace("*", "")
                arg_types = [x.strip() for x in arg.strip().split(" ")[:-1]]
                printers.write("PrintOffset();\n")
                scalar = ["size_t*", "void**", "uint32_t*", "uint64_t*"]
                if any(item in scalar for item in arg_types):
                    printers.write(
                        'std::cout << "{}: " << *(Args->{}) << "\\n";\n'.format(
                            arg_name[1:], arg_name
                        )
                    )
                else:
                    printers.write(
                        '  std::cout << "{}: " << Args->{} << "\\n";\n'.format(
                            arg_name, arg_name
                        )
                    )
            printers.write("break;\n")
            printers.write("}\n")

    printers.close()


if __name__ == "__main__":
    """
    Usage: python generate_pi_pretty_printers.py path/to/ze_api.h
    """
    with open(sys.argv[1], "r") as f:
        header = f.read()
        generate_ze_pretty_printers(header)
