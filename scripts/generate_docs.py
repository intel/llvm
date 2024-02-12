"""
 Copyright (C) 2022-2024 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
import os
import subprocess
import util
import re

RE_ENABLE   = r"^\#\#\s*\-\-validate\s*\=\s*on$"
RE_DISABLE  = r"^\#\#\s*\-\-validate\s*\=\s*off$"

RE_PYCODE_BLOCK_BEGIN = r"^\<\%$"
RE_PYCODE_BLOCK_END   = r"^\%\>$"

RE_INVALID_TAG_FORMAT  = r".*(\$\w).*"
RE_EXTRACT_TAG_NAME    = r"\$\{(\w)\}"
RE_PROPER_TAG_FORMAT   = r".*"+RE_EXTRACT_TAG_NAME+r".*"

RE_CODE_BLOCK_BEGIN = r"\s*..\sparsed-literal::"

RE_EXTRACT_NAME     = r"\$\{\w\}\w+"
RE_EXTRACT_PARAMS   = r"\w+\((.*)\)\;"

RE_VERSION_BEGIN    = r"\%if\s+(ver\s+[\<\=\>]+\s+[-+]?\d*\.\d+|\d+).*"
RE_VERSION_END      = r"\%endif\s+#\s+.*"


"""
    determines if the symbol is known
"""
def _find_symbol_type(name, meta):
    for group in meta:
        if name in meta[group]:
            return group

    if name.isupper():
        for enum in meta['enum']:
            if name in meta['enum'][enum]['etors']:
                return 'etor'

    return None

"""
    fix up tag for template (e.g. $x to ${x})
"""
def _fixup_tag(name):
    return re.sub(r"\$(?P<tag>\w)", r"${\g<tag>}", name)

"""
    find the enum type name for a given enumerator
"""
def _find_enum_from_etor(etor, meta):
    for name in meta['enum']:
        if etor in meta['enum'][name]['etors']:
            return _fixup_tag(name)

    return None

"""
    make restructedtext reference from symbol.
"""
def _make_ref(symbol, symbol_type, meta):
    if not re.match(r"function|struct|union|enum|etor", symbol_type):
        return ""

    ref = _fixup_tag(symbol)
    if re.match("etor", symbol_type):
        target = _find_enum_from_etor(symbol, meta)
        if target:
            ref = ":ref:`" + ref + " <" + target.replace("_", "-") + ">`"
        else:
            print("%s(%s) : error : enum symbol not found for etor %s"%(fin, iline+1, symbol))
            raise Exception()
    elif not re.match("function", symbol_type):
        ref = ":ref:`" + ref.replace("_", "-") + "`"
    else:
        ref = ":ref:`" + ref + "`"

    return ref

"""
    generate a valid reStructuredText file
"""
def _generate_valid_rst(fin, fout, namespace, tags, ver, rev, meta):
    ver=float(ver)
    enable = True
    code_block = False

    print("Generating %s..."%fout)

    error = False

    outlines = []
    iter = enumerate(util.textRead(fin))
    for iline, line in iter:

        if re.match(RE_ENABLE, line) or re.match(RE_PYCODE_BLOCK_END, line):
            enable = True
        elif re.match(RE_DISABLE, line) or re.match(RE_PYCODE_BLOCK_BEGIN, line):
            enable = False

        elif re.match(RE_CODE_BLOCK_BEGIN, line):
            code_block = True
        elif re.match(r'^\w', line) and code_block: # code is always indented
            code_block = False

        elif re.match(RE_VERSION_BEGIN, line):
            enable = eval(re.sub(RE_VERSION_BEGIN, r"\1", line))
        elif re.match(RE_VERSION_END, line):
            enable = True

        if not enable:
            outlines.append(line)
            continue

        if re.match(RE_INVALID_TAG_FORMAT, line):
            print("%s(%s) : error : invalid %s tag used"%(fin, iline+1, re.sub(RE_INVALID_TAG_FORMAT, r"\1", line)))
            error = True

        newline = line # new line will contain proper tags for reStructuredText if needed.
        if re.match(RE_PROPER_TAG_FORMAT, line):
            words = re.findall(RE_EXTRACT_NAME, line)

            newline = ""
            for word in words:
                symbol = re.sub(RE_EXTRACT_TAG_NAME, r"$\1", word)
                if symbol:
                    symbol_type = _find_symbol_type(symbol, meta)
                    if not symbol_type:
                        print("%s(%s) : error : symbol '%s' not found"%(fin, iline+1, symbol))
                        error = True
                        continue

                    if code_block and 'function' == symbol_type:
                        # If function is split across multiple lines
                        # then join lines until a ';' is encountered.
                        try:
                            line = line.rstrip()
                            while not line.endswith(';'):
                                _, n_line = next(iter)
                                line = line + n_line.strip()
                            line += '\n'
                        except StopIteration:
                            print(f"Function {line[:100]} was not terminated by a ';' character.")
                            error = True
                    
                        words = re.sub(RE_EXTRACT_PARAMS, r"\1", line)
                        words = line.split(",")
                        if len(words) != len(meta['function'][symbol]['params']):
                            print("%s(%s) : error : %s parameter count mismatch - %s actual vs. %s expected"%(fin, iline+1, symbol, len(words), len(meta['function'][symbol]['params'])))
                            print("line = %s"%line)
                            error = True

                    ref = _make_ref(symbol, symbol_type, meta)
                    if ref:
                        tuple = line.partition(word)

                        newline += tuple[0] + tuple[1].replace(word, ref)
                        if tuple[2] and not re.match(r'\s', tuple[2]):
                            # reStructuredText requires an escape character after references that are not followed by whitespace.
                            newline += "\\"
                        line = tuple[2]
                    else:
                        # ignore reference links for specific types that have no API documentation for them.
                        if not re.match(r"env|handle|typedef|macro", symbol_type):
                            print("%s(%s) : warning : reference link %s (type=%s) not used."%(fin, iline+1, symbol, symbol_type))
                else:
                    print("%s(%s) : warning : reference link %s not used."%(fin, iline+1, word))

            newline += line

        outlines.append(newline)

    if error:
        raise Exception('Error during reStructuredText generation.')

    util.writelines(os.path.abspath(fout), outlines)

    return util.makoWrite(os.path.abspath(fout), fout,
                          ver=ver,
                          namespace=namespace,
                          tags=tags,
                          meta=meta)

"""
Entry-point:
    generate restructuredtext documents from templates
"""
def generate_rst(docpath, section, namespace, tags, ver, rev, specs, meta):
    srcpath = os.path.join("./", section)
    dstpath = os.path.join(docpath, "source", section)

    loc = 0
    util.makePath(dstpath)
    util.removeFiles(dstpath, "*.rst")
    for fin in util.findFiles(srcpath, "*.rst"):
        fout = os.path.join(dstpath, os.path.basename(fin))
        loc += _generate_valid_rst(os.path.abspath(fin), fout, namespace, tags, ver, rev, meta)

    print("Generated %s lines of reStructuredText (rst).\n"%loc)

    if (loc > 0):
        fin = os.path.join("templates", "api_listing.mako")
        fout = os.path.join(dstpath, "api.rst")
        groupname = os.path.basename(dstpath).capitalize()
        util.makoWrite(fin, fout,
            namespace=namespace,
            groupname=groupname,
            ver=ver,
            rev=rev,
            tags=tags,
            meta=meta,
            specs=specs)

"""
Entry-point:
    generate doxygen xml and setup source path.
"""
def generate_common(dstpath, sections, ver, rev):
    htmlpath = os.path.join(dstpath, "html")
    latexpath = os.path.join(dstpath, "latex")
    xmlpath = os.path.join(dstpath, "xml")
    sourcepath = os.path.join(dstpath, "source")
    util.removePath(htmlpath)
    util.removePath(latexpath)
    util.removePath(xmlpath)
    util.makePath(xmlpath)

    # Generate sphinx configuration file with version.
    loc = 0
    for fn in ["conf.py", "index.rst", "api.rst", "exp-features.rst"]:
        loc += util.makoWrite(
            "./templates/%s.mako" % fn,
            os.path.join(sourcepath, fn),
            ver=ver,
            rev=rev,
            sourcepath=sourcepath,
            sections=sections)

    # Doxygen generates XML files needed by sphinx breathe plugin for API documentation.
    print("Generating doxygen...")
    proc = subprocess.Popen(["doxygen", "Doxyfile"], stderr=subprocess.PIPE)
    proc.wait()
    output = proc.stderr.read().decode()
    print(output)
    if re.search(r"warning: explicit link request to \'.*\' could not be resolved", output) and 'CI' in os.environ:
        raise Exception("Doxygen has unresolved references")

"""
Entry-point:
    generate HTML files using reStructuredText (rst) and Doxygen template
"""
def generate_html(dstpath):
    sourcepath = os.path.join(dstpath, "source")

    print("Generating HTML...")
    result = subprocess.run(["sphinx-build", "-M", "html", sourcepath, "../docs" ], stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("sphinx-build returned non-zero error code.")
        print("--- output ---")
        print(result.stderr.decode())
        raise Exception("Failed to generate html documentation.")

"""
Entry-point:
    generate PDF file using generated LaTeX files
"""
def generate_pdf(dstpath):
    sourcepath = os.path.join(dstpath, "source")

    print("Generating PDF...")
    result = subprocess.run(["sphinx-build", "-b", "pdf", sourcepath, "../docs/latex"], stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("sphinx-build returned non-zero error code.")
        print("--- output ---")
        print(result.stderr.decode())
        raise Exception("Failed to generate pdf documentation.")

"""
Entry-point:
    prepare doc folder for documentation.
"""
def prepare(docpath, gen_rst, gen_html, ver):
    if gen_html:
        htmlpath = os.path.join(docpath, "html")
        if util.exists(htmlpath):
            util.removePath(htmlpath)
    
    # if generating rst then assume everything in docs is invalid and clean it.
    if gen_rst:
        if util.exists(docpath):
            util.removePath(docpath)

    docsourcepath = os.path.join(docpath, "source")
    util.copyTree("./assets/html/_static",    os.path.join(docsourcepath, "_static"))
    util.copyTree("./assets/html/_templates", os.path.join(docsourcepath, "_templates"))
    util.copyTree("./assets/images",          os.path.join(docsourcepath, "images"))
    
