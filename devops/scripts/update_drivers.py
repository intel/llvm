from urllib.request import urlopen
import json
import sys
import os
import re


def get_latest_release(repo):
    releases = urlopen("https://api.github.com/repos/" + repo + "/releases").read()
    return json.loads(releases)[0]


def uplift_linux_igfx_driver(config, platform_tag):
    compute_runtime = get_latest_release('intel/compute-runtime')

    config[platform_tag]['compute_runtime']['github_tag'] = compute_runtime['tag_name']
    config[platform_tag]['compute_runtime']['version'] = compute_runtime['tag_name']
    config[platform_tag]['compute_runtime']['url'] = 'https://github.com/intel/compute-runtime/releases/tag/' + compute_runtime['tag_name']

    for a in compute_runtime['assets']:
        if a['name'].endswith('.sum'):
            deps = str(urlopen(a['browser_download_url']).read())
            m = re.search(r"intel-igc-core_([0-9\.]*)_amd64", deps)
            if m is not None:
                ver = m.group(1)
                config[platform_tag]['igc']['github_tag'] = 'igc-' + ver
                config[platform_tag]['igc']['version'] = ver
                config[platform_tag]['igc']['url'] = 'https://github.com/intel/intel-graphics-compiler/releases/tag/igc-' + ver
                break

    cm = get_latest_release('intel/cm-compiler')
    config[platform_tag]['cm']['github_tag'] = cm['tag_name']
    config[platform_tag]['cm']['version'] = cm['tag_name'].replace('cmclang-', '')
    config[platform_tag]['cm']['url'] = 'https://github.com/intel/cm-compiler/releases/tag/' + cm['tag_name']

    level_zero = get_latest_release('oneapi-src/level-zero')
    config[platform_tag]['level_zero']['github_tag'] = level_zero['tag_name']
    config[platform_tag]['level_zero']['version'] = level_zero['tag_name']
    config[platform_tag]['level_zero']['url'] = 'https://github.com/oneapi-src/level-zero/releases/tag/' + level_zero['tag_name']

    return config


def main(platform_tag):
    script = os.path.dirname(os.path.realpath(__file__))
    config_name = os.path.join(script, '..', 'dependencies.json')
    config = {}

    with open(config_name, "r") as f:
        config = json.loads(f.read())
        config = uplift_linux_igfx_driver(config, platform_tag)

    with open(config_name, "w") as f:
        json.dump(config, f, indent=2)

    return config[platform_tag]['compute_runtime']['version']


if __name__ == '__main__':
    platform_tag = sys.argv[1] if len(sys.argv) > 1 else "ERROR_PLATFORM"
    sys.stdout.write(main(platform_tag) + '\n')
