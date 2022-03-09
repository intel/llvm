module.exports = ({core, process}) => {
  const fs = require('fs');
  fs.readFile('./test_configs.json', 'utf8', (err, data) => {
    if (err) {
      console.log(`Error reading file from disk: ${err}`);
    } else {
      const driverNew =
          JSON.parse(fs.readFileSync('./dependencies.json', 'utf8'));
      const driverOld =
          JSON.parse(fs.readFileSync('./dependencies.sycl.json', 'utf8'));
      const testConfigs = JSON.parse(data);
      const inputs = JSON.parse(process.env.GHA_INPUTS);
      const needsDrivers =
          driverNew["linux"]["compute_runtime"]["version"] !==
              driverOld["linux"]["compute_runtime"]["version"] ||
          driverNew["linux"]["igc"]["version"] !==
              driverOld["linux"]["igc"]["version"] ||
          driverNew["linux"]["cm"]["version"] !==
              driverOld["linux"]["cm"]["version"] ||
          driverNew["linux"]["tbb"]["version"] !==
              driverOld["linux"]["tbb"]["version"] ||
          driverNew["linux"]["oclcpu"]["version"] !==
              driverOld["linux"]["oclcpu"]["version"] ||
          driverNew["linux"]["fpgaemu"]["version"] !==
              driverOld["linux"]["fpgaemu"]["version"];

      const ltsConfigs = inputs.lts_config.split(';');

      const enabledLTSConfigs = [];

      testConfigs.lts.forEach(v => {
        if (ltsConfigs.includes(v.config)) {
          if (needsDrivers) {
            v["env"] = [
              {
                "compute_runtime_tag" :
                    driverNew["linux"]["compute_runtime"]["github_tag"]
              },
              {"igc_tag" : driverNew["linux"]["igc"]["github_tag"]},
              {"cm_tag" : driverNew["linux"]["cm"]["github_tag"]},
              {"tbb_tag" : driverNew["linux"]["tbb"]["github_tag"]},
              {"cpu_tag" : driverNew["linux"]["oclcpu"]["github_tag"]},
              {"fpgaemu_tag" : driverNew["linux"]["fpgaemu"]["github_tag"]},
            ];
          } else {
            v["env"] = [];
          }
          enabledLTSConfigs.push(v);
        }
      });

      let ltsString = JSON.stringify(enabledLTSConfigs);
      console.log(ltsString);

      for (let [key, value] of Object.entries(inputs)) {
        ltsString = ltsString.replaceAll("${{ inputs." + key + " }}", value);
      }
      ltsString = ltsString.replaceAll(
          "ghcr.io/intel/llvm/ubuntu2004_intel_drivers:latest",
          "ghcr.io/intel/llvm/ubuntu2004_base:latest");

      core.setOutput('lts', ltsString);
    }
  });
}
