from urllib.request import urlopen
import json
import sys
import os
import re


def get_latest_release(repo):
    releases = urlopen("https://api.github.com/repos/" + repo + "/releases").read()
    return json.loads(releases)[0]


def uplift_linux_igfx_driver(config):
    compute_runtime = get_latest_release('intel/compute-runtime')

    config['linux_staging']['compute_runtime']['github_tag'] = compute_runtime['tag_name']
    config['linux_staging']['compute_runtime']['version'] = compute_runtime['tag_name']
    config['linux_staging']['compute_runtime']['url'] = 'https://github.com/intel/compute-runtime/releases/tag/' + compute_runtime['tag_name']

    for a in compute_runtime['assets']:
        if a['name'].endswith('.sum'):
            deps = str(urlopen(a['browser_download_url']).read())
            print(a['browser_download_url'])
            print(deps)
            m = re.search(r"intel-igc-core_([0-9\.]*)_amd64", deps)
            if m is not None:
                ver = m.group()
                print("IGC MATCH")
                config['linux_staging']['igc']['github_tag'] = 'igc-' + ver
                config['linux_staging']['igc']['version'] = ver
                config['linux_staging']['igc']['url'] = 'https://github.com/intel/intel-graphics-compiler/releases/tag/igc-' + ver
                break

    cm = get_latest_release('intel/cm-compiler')
    config['linux_staging']['cm']['github_tag'] = cm['tag_name']
    config['linux_staging']['cm']['version'] = cm['tag_name'].replace('cmclang-', '')
    config['linux_staging']['cm']['url'] = 'https://github.com/intel/cm-compiler/releases/tag/' + cm['tag_name']

    return config


def main():
    script = os.path.dirname(os.path.realpath(__file__))
    config_name = os.path.join(script, '..', 'dependencies.json')
    config = {}

    with open(config_name, "r") as f:
        config = json.loads(f.read())
        config = uplift_linux_igfx_driver(config)

    with open(config_name, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
