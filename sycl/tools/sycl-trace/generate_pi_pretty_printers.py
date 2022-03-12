import re
import sys

def generate_pi_pretty_printers(header):
    hdr = open("pi_structs.hpp", "w")
    # TODO copyright
    hdr.write("#pragma once\n")
    printers = open("pi_printers.def", "w")

    matches = re.finditer(r'(pi[a-zA-Z]+)\(([a-zA-Z\_\s,\*0-9]*)\);', header)

    for match in matches:
        api_name = str(match.group(1))

        if api_name == 'piPluginInit':
            continue

        all_args = match.group(2).replace('\n', '').split(',')

        hdr.write("struct __attribute__((packed)) " + api_name + "_args {\n")

        for arg in all_args:
            hdr.write(arg.strip() + ";\n")

        hdr.write("};\n")

        arg_names = []

        for arg in all_args:
            name = arg.split(" ")[-1].replace('*', '')
            arg_names.append(name)

        printers.write("case static_cast<uint32_t>(sycl::detail::PiApiKind::{}): {{\n".format(api_name))
        printers.write("const auto *Args = reinterpret_cast<{}_args*>(Data->args_data);\n".format(api_name))
        for name in arg_names:
            printers.write('std::cout << "    {}: " << Args->{} << "\\n";\n'.format(name, name))
        printers.write("break;\n")
        printers.write("}\n")

if __name__ == "__main__":
    """
    Usage: python generate_pi_pretty_printers.py path/to/pi.h
    """
    with open(sys.argv[1], 'r') as f:
        header = f.read()
        generate_pi_pretty_printers(header)
