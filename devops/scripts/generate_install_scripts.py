import json
import sys
import os

script = os.path.dirname(os.path.realpath(__file__))
config_name = os.path.join(script, '..', 'dependencies.json')
linux_script = os.path.join(script, 'install_drivers.sh')

config_file = open(config_name, "r")
config = json.loads(config_file.read())

if sys.argv[1] == "linux":
    template = open(linux_script, "r")
    out_file = open(sys.argv[2], "w")

    script = str(template.read())
    script = script.replace("$compute_runtime_tag", "\"" + config['linux']['compute_runtime']['github_tag'] + "\"")
    script = script.replace("$igc_tag", "\"" + config['linux']['igc']['github_tag'] + "\"")
    script = script.replace("$cm_tag", "\"" + config['linux']['cm']['github_tag'] + "\"")
    script = script.replace("$tbb_tag", "\"" + config['linux']['tbb']['github_tag'] + "\"")
    script = script.replace("$cpu_tag", "\"" + config['linux']['oclcpu']['github_tag'] + "\"")
    script = script.replace("$fpgaemu_tag", "\"" + config['linux']['fpgaemu']['github_tag'] + "\"")

    out_file.write(script)
    out_file.close()

else:
    print("Unknown platform {}".format(sys.argv[1]))
