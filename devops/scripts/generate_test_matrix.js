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

      const e2eConfigs = inputs.e2e_config.split(';');

      const enabledE2ELxConfigs = [];
      const enabledE2EWnConfigs = [];
      const enabledE2EAWSConfigs = [];

      // Process E2E (SYCL End-to-End tests)

      testConfigs.e2e.forEach(v => {
        if (e2eConfigs.includes(v.config)) {
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
            enabledE2EWnConfigs.push(v);
          else if (v["runs-on"].includes("Linux"))
            enabledE2ELxConfigs.push(v);
          else
            console.error("runs-on OS is not recognized");
          if (v["aws-type"]) enabledE2EAWSConfigs.push(v);
        }
      });

      let e2eLxString = JSON.stringify(enabledE2ELxConfigs);
      let e2eWnString = JSON.stringify(enabledE2EWnConfigs);
      let e2eAWSString = JSON.stringify(enabledE2EAWSConfigs);
      console.log("Linux E2E config:")
      console.log(e2eLxString);
      console.log("Windows E2E config:")
      console.log(e2eWnString);
      console.log("Linux AWS E2E config:")
      console.log(e2eAWSString)

      // drivers update is supported on Linux only
      for (let [key, value] of Object.entries(inputs)) {
        e2eLxString =
            e2eLxString.replaceAll("${{ inputs." + key + " }}", value);
        e2eAWSString = e2eAWSString.replaceAll("${{ inputs." + key + " }}", value);
      }
      if (needsDrivers) {
        e2eLxString = e2eLxString.replaceAll(
            "ghcr.io/intel/llvm/ubuntu2204_intel_drivers:latest",
            "ghcr.io/intel/llvm/ubuntu2204_base:latest");
        e2eAWSString = e2eAWSString.replaceAll(
            "ghcr.io/intel/llvm/ubuntu2204_intel_drivers:latest",
            "ghcr.io/intel/llvm/ubuntu2204_base:latest");
      }

      core.setOutput('e2e_lx_matrix', e2eLxString);
      core.setOutput('e2e_wn_matrix', e2eWnString);
      core.setOutput('e2e_aws_matrix', e2eAWSString);

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
            "ghcr.io/intel/llvm/ubuntu2204_intel_drivers:latest",
            "ghcr.io/intel/llvm/ubuntu2204_base:latest");
      }

      core.setOutput('cts_matrix', ctsString);
    }
  });
}
