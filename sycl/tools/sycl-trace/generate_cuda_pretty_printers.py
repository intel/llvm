import re
import sys


def generate_ze_pretty_printers(header):
    printers = open("cuda_printers.def", "w")

    matches = re.finditer(
        r"typedef struct (cu[a-zA-Z_]+)_params_st {\n([a-zA-Z\s\n;\*0-9]+)}", header
    )

    for match in matches:
        api_name = match.group(1)
        args = match.group(2).replace("\n", "").strip().split(";")[:-1]

        printers.write(
            "case static_cast<uint32_t>(CUPTI_DRIVER_TRACE_CBID_{}): {{\n".format(
                api_name
            )
        )
        printers.write(
            "const auto *Args = reinterpret_cast<const {}_params_st*>(Data->args_data);\n".format(
                api_name
            )
        )
        printers.write("(void)Args;\n")
        for arg in args:
            arg_name = arg.split(" ")[-1].replace("*", "")
            arg_types = [x.strip() for x in arg.strip().split(" ")[:-1]]
            printers.write("PrintOffset();\n")
            printable = ["size_t", "void", "int", "char"]
            if any(item in printable for item in arg_types):
                printers.write(
                    'std::cout << "{}: " << Args->{} << "\\n";\n'.format(
                        arg_name, arg_name
                    )
                )
            else:
                printers.write(
                    'std::cout << "{}: <non-printable>" << "\\n";\n'.format(arg_name)
                )

        printers.write("break;\n")
        printers.write("}\n")

    printers.close()


if __name__ == "__main__":
    """
    Usage: python generate_cuda_pretty_printers.py path/to/generated_cuda_meta.h
    """
    with open(sys.argv[1], "r") as f:
        header = f.read()
        generate_ze_pretty_printers(header)
