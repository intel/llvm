#!/usr/bin/env python3
"""
 Copyright (C) 2022 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
import argparse
import re
import util
import parse_specs
import generate_code
import generate_docs
import os
import sys
import platform
import time
import subprocess
from version import Version

"""
    helper for adding mutually-exclusive boolean arguments "--name" and "--!name"
"""
def add_argument(parser, name, help, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, help="Enable "+help, action="store_true")
    group.add_argument("--!" + name, dest=name, help="Disable "+help, action="store_false")
    parser.set_defaults(**{name:default})

"""
    helper for cleaning previously generated files
"""
def clean():
    util.removePath("../include")
    util.makePath("../include")
    util.removePath("../build")
    util.makePath("../build")

"""
    help for updating spec documentation
"""
def update_spec(target):
    inc = "%s/source/elements/l0/include" % target
    src = "%s/source/elements/l0/source" % target
    util.copyTree("../include", inc)
    util.copyTree("../docs/source", src)
    util.removePath("%s/experimental" % inc)
    util.removePath("%s/experimental" % src)


"""
    helper for running cmake windows build
"""
def build():
    if "Windows" == platform.system():
        result = os.system('cmake -B ../build/ -S .. -G "Visual Studio 16 2019" -A x64')
    else:
        result = -1 #todo
    if result == 0:
        result = os.system('cmake --build ../build --clean-first')
    return result == 0

"""
    helper for getting revision number from git repository
    revision is number of commits since tag 'v0'
"""
def revision():
    return '0'
    result = subprocess.run(['git', 'describe', '--tags', '--dirty'], cwd=os.path.dirname(os.path.abspath(__file__)), stdout=subprocess.PIPE)
    if result.returncode:
        print('ERROR: Could not get revision number from git', file=sys.stderr)
        return '0'

    items = result.stdout.decode().strip().split('-')
    tag = items[0][1:] # remove 'v'
    print("Version is %s" % tag)
    count = 0
    if len(items) > 1 and items[1].isdigit():
        count = int(items[1])

    # Bump count if any local files are dirty.
    # Keeps the count the same after doing a commit (assuming all dirty files are committed)
    if 'dirty' in items[-1]:
        count += 1
    return '%s.%s'%(tag, count)


"""
    helper for getting the default version from the project() command in the
    root CMakeLists.txt file
"""
def get_version_from_cmakelists():
    cmakelists_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'CMakeLists.txt'))
    with open(cmakelists_path, 'r') as cmakelists_file:
        for line in cmakelists_file.readlines():
            line = line.strip()
            if line.startswith('project('):
                return Version(re.findall(r'\d+\.\d+', line)[0])
    raise Exception(f'unable to read project version from {cmakelists_path}')


"""
Main entry:
    Do everything...
"""
def main():
    # phase 0: parse cmdline arguments
    configParser = util.configRead("config.ini")

    parser = argparse.ArgumentParser()
    for section in configParser.sections():
        add_argument(parser, section, "generation of C/C++ '%s' files."%section, True)
    add_argument(parser, "clean", "cleaning previous generated files.")
    add_argument(parser, "build", "running cmake to generate and build projects.", False)
    add_argument(parser, "debug", "dump intermediate data to disk.")
    add_argument(parser, "html", "generation of HTML files.", True)
    add_argument(parser, "pdf", "generation of PDF file.")
    add_argument(parser, "rst", "generation of reStructuredText files.", True)
    parser.add_argument("--update_spec", type=str, help="root of integrated spec directory to update")
    parser.add_argument(
        "--ver",
        type=parse_specs.Version,
        default=get_version_from_cmakelists(),
        required=False,
        help="specification version to generate.",
    )
    parser.add_argument("--api-json", type=str, default="unified_runtime.json", required=False, help="json output file for the spec")
    parser.add_argument("--clang-format", type=str, default="clang-format", required=False, help="path to clang-format executable")
    parser.add_argument('--fast-mode', action='store_true', help='Disable sections which are slow to render')
    args = vars(parser.parse_args())
    args['rev'] = revision()

    print('Version', args['ver'])

    start = time.time()

    # phase 1: extract configuration info from ini file
    input = {
        'configs': [],
        'specs'  : [],
        'meta'   : {},
        'ref'    : {}
        }

    for section in configParser.sections():
        input['configs'].append({
            'name'     : section,
            'namespace': configParser.get(section,'namespace'),
            'tags'     : {'$'+key : configParser.get(section,key) for key in configParser.get(section,'tags').split(",")},
            })

    # phase 2: parse specs
    for config in input['configs']:
        specs, input['meta'], input['ref'] = parse_specs.parse(config['name'], args['ver'], config['tags'], input['meta'], input['ref'])
        input['specs'].append(specs)

    util.jsonWrite(args['api_json'], input)

    # phase 3: generate files
    if args['clean']:
        clean()

    incpath = os.path.join("../include/")
    srcpath = os.path.join("../source/")
    docpath = os.path.join("../docs/")

    generate_docs.prepare(docpath, args['rst'], args['html'], args['ver'])

    for idx, specs in enumerate(input['specs']):
        config = input['configs'][idx]
        if args[config['name']]:

            generate_code.generate_api(incpath, srcpath, config['namespace'], config['tags'], args['ver'], args['rev'], specs, input['meta'])

            # clang-format ur_api.h
            proc = subprocess.run([args['clang_format'], "--style=file", "-i" , "ur_api.h"], stderr=subprocess.PIPE, cwd=incpath)
            if proc.returncode != 0:
                print("-- clang-format failed with non-zero return code. --")
                print(proc.stderr.decode())
                raise Exception("Failed to format ur_api.h")

            if args['rst']:
                generate_docs.generate_rst(docpath, config['name'], config['namespace'], config['tags'], args['ver'], args['rev'], specs, input['meta'], args['fast_mode'])

        if util.makeErrorCount():
            print("\n%s Errors found during generation, stopping execution!"%util.makeErrorCount())
            return

    if args['debug']:
        util.makoFileListWrite("generated.json")

    # phase 4: build code
    if args['build']:
        if not build():
            print("\nBuild failed, stopping execution!")
            return

    # phase 5: prep for publication of html or pdf
    if args['html'] or args['pdf']:
        generate_docs.generate_common(docpath, configParser.sections(), args['ver'], args['rev'])

    # phase 5: publish documentation
    if args['html']:
        generate_docs.generate_html(docpath)

    if args['pdf']:
        generate_docs.generate_pdf(docpath)

    if args['update_spec']:
        update_spec(args['update_spec'])

    print("\nCompleted in %.1f seconds!"%(time.time() - start))


if __name__ == '__main__':
    main()
