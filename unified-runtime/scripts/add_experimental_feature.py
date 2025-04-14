# Copyright (C) 2023 Intel Corporation
#
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import sys
from util import makoWrite
import re
import subprocess


def verify_kebab_case(input: str) -> bool:
    kebab_case_re = r"[a-z0-9]+(?:-[a-z0-9]+)*"
    pattern = re.compile(kebab_case_re)
    if pattern.match(input) is None:
        return False
    return True


def get_user_name_email_from_git_config():
    proc = subprocess.run(["git", "config", "user.name"], stdout=subprocess.PIPE)
    if proc.returncode != 0:
        raise Exception("Failed to get user name from git config.")
    user_name = proc.stdout.decode().strip()

    proc = subprocess.run(["git", "config", "user.email"], stdout=subprocess.PIPE)
    if proc.returncode != 0:
        raise Exception("Failed to get user email from git config.")
    user_email = proc.stdout.decode().strip()

    return user_name, user_email


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "name", help="must be lowercase and kebab case i.e. command-buffer", type=str
    )
    argParser.add_argument(
        "--dry_run",
        help="run the script without generating any files",
        action="store_true",
    )
    args = argParser.parse_args()

    if not verify_kebab_case(args.name):
        print("Name must be lowercase and kebab-case i.e. command-buffer.")
        sys.exit(1)

    user_name, user_email = get_user_name_email_from_git_config()
    user = {"email": user_email, "name": user_name}

    exp_feat_name = args.name

    out_yml_name = f"exp-{exp_feat_name}.yml"
    out_rst_name = f"EXP-{exp_feat_name.upper()}.rst"

    yaml_template_path = "./scripts/templates/exp_feat.yml.mako"
    rst_template_path = "./scripts/templates/exp_feat.rst.mako"
    out_yml_path = f"./scripts/core/{out_yml_name}"
    out_rst_path = f"./scripts/core/{out_rst_name}"

    if not args.dry_run:
        makoWrite(yaml_template_path, out_yml_path, name=exp_feat_name)
        makoWrite(rst_template_path, out_rst_path, name=exp_feat_name, user=user)

    print(
        f"""\
Successfully generated the template files needed for {exp_feat_name}.

You can now implement your feature in the following files:
    * {out_yml_name} 
    * {out_rst_name}
"""
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(130)
