"""
 Copyright (C) 2022 Intel Corporation

 SPDX-License-Identifier: MIT

"""
import argparse
import util
import parse_specs
import generate_code
import generate_docs
import os, sys, platform
import time
import subprocess

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
    return '0.5'
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
    parser.add_argument("--ver", type=str, default="0.5", required=False, help="specification version to generate.")
    args = vars(parser.parse_args())
    args['rev'] = revision()

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

    if args['debug']:
        util.jsonWrite("input.json", input)

    util.jsonWrite("level_zero.json", input['ref'])

    # phase 3: generate files
    if args['clean']:
        clean()

    incpath = os.path.join("../include/")
    srcpath = os.path.join("../source/")
    docpath = os.path.join("../docs/")

    generate_docs.prepare(docpath, args['rst'], args['html'], args['ver'])
    generate_docs.generate_ref(docpath, input['ref'])

    for idx, specs in enumerate(input['specs']):
        config = input['configs'][idx]
        if args[config['name']]:

            generate_code.generate_api(incpath, srcpath, config['namespace'], config['tags'], args['ver'], args['rev'], specs, input['meta'])

            if args['rst']:
                generate_docs.generate_rst(docpath, config['name'], config['namespace'], config['tags'], args['ver'], args['rev'], specs, input['meta'])

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
# END OF FILE
