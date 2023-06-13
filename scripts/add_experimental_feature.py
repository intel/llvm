"""
 Copyright (C) 2023 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
import argparse
import sys
from util import makoWrite
import re

def verify_kebab_case(input: str) -> bool:
    # kebab case regex from https://github.com/expandjs/expandjs/blob/master/lib/kebabCaseRegex.js
    kebab_case_re = r"^([a-z](?![\d])|[\d](?![a-z]))+(-?([a-z](?![\d])|[\d](?![a-z])))*$|^$"
    pattern = re.compile(kebab_case_re)
    if pattern.match(input) is None:
        return False
    return True


def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("name", help="must be lowercase and kebab case i.e. command-buffer", type=str)
    args = argParser.parse_args()

    if not verify_kebab_case(args.name):
        print("Name must be lowercase and kebab-case i.e. command-buffer.")
        sys.exit(1)

    exp_feat_name = args.name
    
    out_yml_name = "exp-%s.yml" % exp_feat_name
    out_rst_name = "EXP-%s.rst" % exp_feat_name.upper()

    yaml_template_path = "./scripts/templates/%s" % "exp_feat.yml.mako"
    rst_template_path = "./scripts/templates/%s" % "exp_feat.rst.mako"
    out_yml_path = "./scripts/core/%s" % out_yml_name
    out_rst_path = "./scripts/core/%s" % out_rst_name

    makoWrite(yaml_template_path, out_yml_path, name=exp_feat_name)
    makoWrite(rst_template_path, out_rst_path, name=exp_feat_name)


    print("Successfully generated the template files needed for %s." % exp_feat_name)
    print("""
You can now implement your feature in the following files:
    * %s 
    * %s
""" % (out_yml_name, out_rst_name))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(130)
