from urllib.request import urlopen
import json
import sys
import os

def uplift_linux_dev_igfx_driver(config, platform_tag):
    action_runs = urlopen("https://api.github.com/repos/intel/intel-graphics-compiler/actions/runs").read()
    workflow_runs = json.loads(action_runs)['workflow_runs']

    for run in workflow_runs:
        if run['name'] != 'Build IGC':
            continue
        if run['status'] != 'completed':
            continue
        if run['conclusion'] != 'success':
            continue
        config[platform_tag]['igc_dev']['github_hash'] = run['head_sha'][:7]
        break

    return config

def main(platform_tag):
    script = os.path.dirname(os.path.realpath(__file__))
    config_name = os.path.join(script, '..', 'dependencies.json')
    config = {}

    with open(config_name, "r") as f:
        config = json.loads(f.read())
        config = uplift_linux_dev_igfx_driver(config, platform_tag)

    with open(config_name, "w") as f:
        json.dump(config, f, indent=2)
        f.write('\n')

    return config[platform_tag]['igc_dev']['github_hash']


if __name__ == '__main__':
    platform_tag = sys.argv[1] if len(sys.argv) > 1 else "ERROR_PLATFORM"
    sys.stdout.write(main(platform_tag) + '\n')
