module.exports = ({core, process}) => {
  const fs = require('fs');
  fs.readFile('./test_configs.json', 'utf8', (err, data) => {
    if (err) {
      console.error(`Error reading file from disk: ${err}`);
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
          driverNew["linux"]["level_zero"]["version"] !==
              driverOld["linux"]["level_zero"]["version"] ||
          driverNew["linux"]["tbb"]["version"] !==
              driverOld["linux"]["tbb"]["version"] ||
          driverNew["linux"]["oclcpu"]["version"] !==
              driverOld["linux"]["oclcpu"]["version"] ||
          driverNew["linux"]["fpgaemu"]["version"] !==
              driverOld["linux"]["fpgaemu"]["version"];

      const ltsConfigs = inputs.lts_config.split(';');

      const enabledLTSLxConfigs = [];
      const enabledLTSWnConfigs = [];
      const enabledLTSAWSConfigs = [];

      // Process LTS (LLVM Test Suite)

      testConfigs.lts.forEach(v => {
        if (ltsConfigs.includes(v.config)) {
          if (needsDrivers) {
            v["env"] = {
              "compute_runtime_tag" :
                  driverNew["linux"]["compute_runtime"]["github_tag"],
              "igc_tag" : driverNew["linux"]["igc"]["github_tag"],
              "cm_tag" : driverNew["linux"]["cm"]["github_tag"],
              "level_zero_tag" : driverNew["linux"]["level_zero"]["github_tag"],
              "tbb_tag" : driverNew["linux"]["tbb"]["github_tag"],
              "cpu_tag" : driverNew["linux"]["oclcpu"]["github_tag"],
              "fpgaemu_tag" : driverNew["linux"]["fpgaemu"]["github_tag"],
            };
          } else {
            v["env"] = {};
          }
          if (v["runs-on"].includes("Windows"))
            enabledLTSWnConfigs.push(v);
          else if (v["runs-on"].includes("Linux"))
            enabledLTSLxConfigs.push(v);
          else
            console.error("runs-on OS is not recognized");
          if (v["aws-type"]) enabledLTSAWSConfigs.push(v);
        }
      });

      let ltsLxString = JSON.stringify(enabledLTSLxConfigs);
      let ltsWnString = JSON.stringify(enabledLTSWnConfigs);
      let ltsAWSString = JSON.stringify(enabledLTSAWSConfigs);
      console.log("Linux LTS config:")
      console.log(ltsLxString);
      console.log("Windows LTS config:")
      console.log(ltsWnString);
      console.log("Linux AWS LTS config:")
      console.log(ltsAWSString)

      // drivers update is supported on Linux only
      for (let [key, value] of Object.entries(inputs)) {
        ltsLxString =
            ltsLxString.replaceAll("${{ inputs." + key + " }}", value);
        ltsAWSString = ltsAWSString.replaceAll("${{ inputs." + key + " }}", value);
      }
      if (needsDrivers) {
        ltsLxString = ltsLxString.replaceAll(
            "ghcr.io/intel/llvm/ubuntu2004_intel_drivers:latest",
            "ghcr.io/intel/llvm/ubuntu2004_base:latest");
        ltsAWSString = ltsAWSString.replaceAll(
            "ghcr.io/intel/llvm/ubuntu2004_intel_drivers:latest",
            "ghcr.io/intel/llvm/ubuntu2004_base:latest");
      }

      core.setOutput('lts_lx_matrix', ltsLxString);
      core.setOutput('lts_wn_matrix', ltsWnString);
      core.setOutput('lts_aws_matrix', ltsAWSString);

      // Process CTS (Conformance Test Suite)

      const ctsConfigs = inputs.cts_config.split(';');

      const enabledCTSConfigs = [];

      testConfigs.cts.forEach(v => {
        if (ctsConfigs.includes(v.config)) {
          if (needsDrivers) {
            v["env"] = {
              "compute_runtime_tag" :
                  driverNew["linux"]["compute_runtime"]["github_tag"],
              "igc_tag" : driverNew["linux"]["igc"]["github_tag"],
              "cm_tag" : driverNew["linux"]["cm"]["github_tag"],
              "level_zero_tag" : driverNew["linux"]["level_zero"]["github_tag"],
              "tbb_tag" : driverNew["linux"]["tbb"]["github_tag"],
              "cpu_tag" : driverNew["linux"]["oclcpu"]["github_tag"],
              "fpgaemu_tag" : driverNew["linux"]["fpgaemu"]["github_tag"],
            };
          } else {
            v["env"] = {};
          }
          enabledCTSConfigs.push(v);
        }
      });

      let ctsString = JSON.stringify(enabledCTSConfigs);
      console.log("CTS config:")
      console.log(ctsString);

      for (let [key, value] of Object.entries(inputs)) {
        ctsString = ctsString.replaceAll("${{ inputs." + key + " }}", value);
      }
      if (needsDrivers) {
        ctsString = ctsString.replaceAll(
            "ghcr.io/intel/llvm/ubuntu2004_intel_drivers:latest",
            "ghcr.io/intel/llvm/ubuntu2004_base:latest");
      }

      core.setOutput('cts_matrix', ctsString);
    }
  });
}
