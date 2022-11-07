import os
import sys
import argparse
import re
import fileinput
from distutils import dir_util
import util

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)


"""
Entry-point:
    publishes HTML for GitLab pages
"""
def publish_gitlab_html():
    src_html_dir = os.path.join(root_dir, "docs", "html")
    src_img_dir = os.path.join(root_dir, "images")
    tmp_dir = os.path.join(root_dir, ".public")
    tmp_img_dir = os.path.join(root_dir, ".public/images")
    publishing_dir = os.path.join(root_dir, "public")

    # Remove dest dirs
    if os.path.exists(tmp_dir):
        print("Deleting temp dir: %s" % tmp_dir)
        util.removePath(tmp_dir)
    if os.path.exists(publishing_dir):
        print("Deleting publishing dir: %s" % publishing_dir)
        util.removePath(publishing_dir)

    # Copy over generated content to new folder
    print("Copying html files from '%s' to '%s'" % (src_html_dir, tmp_dir))
    dir_util.copy_tree(src_html_dir, tmp_dir)

    # Fixes html files by converting paths relative to root html folder instead of repo
    print("Fixing paths in html files in '%s' to be relative to root..." % (tmp_dir))
    regex_pattern = re.compile(r'\.\.[\/|\\]images')
    files = util.findFiles(tmp_dir, "*.html")
    print("Found %s files" % (len(files)))
    with fileinput.FileInput(files=files, inplace=True) as f:
        for line in f:
            print(re.sub(regex_pattern, './images', line), end='')

    # Publish new folder to GitLab Pages folder (/public)
    print("Publishing to GitLab pages by renaming '%s' to '%s'" % (tmp_dir, publishing_dir))
    os.rename(tmp_dir, publishing_dir)


"""
Entry-point:
    main()
"""
def main(args=sys.argv[1:]):
    # Define args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish-html",
        help="Publish html",
        action="store_true")

    # Parse args
    options = parser.parse_args(args)

    # Publish GitLab html
    if options.publish_html:
        try:
            publish_gitlab_html()
        except Exception as e:
            print(e)
            print("Failed")
            return 1

    print("Done")
    return 0


if __name__ == '__main__':
    sys.exit(main())
# END OF FILE
