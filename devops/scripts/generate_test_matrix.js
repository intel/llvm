module.exports = ({core, process}) => {
  const fs = require('fs');
  fs.readFile('./test_configs.json', 'utf8', (err, data) => {
    if (err) {
      console.log(`Error reading file from disk: ${err}`);
    } else {
      const testConfigs = JSON.parse(data);
      const inputs = JSON.parse(process.env.GHA_INPUTS);

      const ltsConfigs = inputs.lts_config.split(';');

      const enabledLTSConfigs = [];

      testConfigs.lts.forEach(v => {
        if (ltsConfigs.includes(v.config)) {
          enabledLTSConfigs.push(v);
        }
      });

      let ltsString = JSON.stringify(enabledLTSConfigs);
      console.log(ltsString);

      for (let [key, value] of Object.entries(inputs)) {
        ltsString = ltsString.replaceAll("${{ inputs." + key + " }}", value);
      }

      core.setOutput('lts', ltsString);
    }
  });
}
