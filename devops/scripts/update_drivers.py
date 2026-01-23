from urllib.request import urlopen
import json
import sys
import os
import re
import argparse


def get_latest_release(repo, allow_prerelease=True):
    url = "https://api.github.com/repos/" + repo + "/releases"
    releases_raw = urlopen(url).read()
    releases = json.loads(releases_raw)
    if allow_prerelease:
        return releases[0]
    # The GitHub API doesn't allow us to filter prereleases
    # in the query so do it manually.
    for release in releases:
        if release["prerelease"] == False:
            return release
    raise ValueError("No prereleases required but no releases found")

def get_latest_workflow_runs(repo, workflow_name):
    action_runs = urlopen(
        "https://api.github.com/repos/"
        + repo
        + "/actions/workflows/"
        + workflow_name
        + ".yml/runs"
    ).read()
    return json.loads(action_runs)["workflow_runs"][0]


def get_artifacts_download_url(repo, name):
    artifacts = urlopen(
        "https://api.github.com/repos/" + repo + "/actions/artifacts?name=" + name
    ).read()
    return json.loads(artifacts)["artifacts"][0]["archive_download_url"]


def uplift_linux_igfx_driver(config, platform_tag):

    compute_runtime = get_latest_release('intel/compute-runtime')

    config[platform_tag]['compute_runtime']['github_tag'] = compute_runtime['tag_name']
    config[platform_tag]['compute_runtime']['version'] = compute_runtime['tag_name']
    config[platform_tag]['compute_runtime']['url'] = 'https://github.com/intel/compute-runtime/releases/tag/' + compute_runtime['tag_name']

    m = re.search(
        re.escape("https://github.com/intel/intel-graphics-compiler/releases/tag/")
        + r"(v[\.0-9]+)",
        compute_runtime["body"],
    )
    if m is not None:
        ver = m.group(1)
        config[platform_tag]["igc"]["github_tag"] = ver
        config[platform_tag]["igc"]["version"] = ver
        config[platform_tag]["igc"]["url"] = (
            "https://github.com/intel/intel-graphics-compiler/releases/tag/" + ver
        )

    cm = get_latest_release("intel/cm-compiler", allow_prerelease=False)
    config[platform_tag]["cm"]["github_tag"] = cm["tag_name"]
    config[platform_tag]["cm"]["version"] = cm["tag_name"].replace("cmclang-", "")
    config[platform_tag]["cm"]["url"] = (
        "https://github.com/intel/cm-compiler/releases/tag/" + cm["tag_name"]
    )

    level_zero = get_latest_release("oneapi-src/level-zero", allow_prerelease=False)
    config[platform_tag]["level_zero"]["github_tag"] = level_zero["tag_name"]
    config[platform_tag]["level_zero"]["version"] = level_zero["tag_name"]
    config[platform_tag]["level_zero"]["url"] = (
        "https://github.com/oneapi-src/level-zero/releases/tag/"
        + level_zero["tag_name"]
    )

    return config


def main(platform_tag):
    script = os.path.dirname(os.path.realpath(__file__))
    config_name = os.path.join(script, "..", "dependencies.json")
    config = {}

    with open(config_name, "r") as f:
        config = json.loads(f.read())
        config = uplift_linux_igfx_driver(config, platform_tag)

    with open(config_name, "w") as f:
        json.dump(config, f, indent=2)
        f.write('\n')

    return config[platform_tag]['compute_runtime']['version']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("platform_tag")
    args = parser.parse_args()
    sys.stdout.write(main(args.platform_tag) + "\n")
