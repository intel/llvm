import json
import sys
import os

with open(sys.argv[1]) as f:
    config = json.load(f)
    inputs = json.load(os.environ['GHA_INPUT'])

    enabled_lts_configs = inputs.lts.slit(";")

    lts_config = []

    for c in config['lts']:
        if c.config in enabled_lts_configs:
            lts_config.append(c)

    lts_str = json.dumps(lts_config)

    for k, v in inputs.items():
        lts_str = lts_str.replace('${{ inputs.{} }}'.format(k), v)

    lts_str = lts_str.replace('%', '%25')
    lts_str = lts_str.replace('\n', '%0A')
    lts_str = lts_str.replace('\r', '%0D')

    print("::set-ouput name=lts::{}".format(lts_str))
