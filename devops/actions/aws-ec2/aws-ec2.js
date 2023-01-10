const core   = require('@actions/core');
const github = require('@actions/github');
const AWS    = require('aws-sdk');

// shortcut to reference current repo
const repo = `${github.context.repo.owner}/${github.context.repo.repo}`;

// get github registration token that allows to register new runner based on
// GH_PERSONAL_ACCESS_TOKEN github user api key
async function getGithubRegToken() {
  core.info("Preparing Github SDK API");
  const octokit = github.getOctokit(core.getInput("GH_PERSONAL_ACCESS_TOKEN"));

  try {
    core.info(`Getting Github Actions Runner registration token for ${repo} repo`);
    const response = await octokit.request(`POST /repos/${repo}/actions/runners/registration-token`);
    core.info("Got Github Actions Runner registration token");
    return response.data.token;
  } catch (error) {
    core.error("Error getting Github Actions Runner registration token");
    throw error;
  }
}

// add delay before retrying promise one more time
function rejectDelay(reason) {
  return new Promise(function(resolve, reject) {
    setTimeout(reject.bind(null, reason), 10 * 1000);
  });
}

// we better keep GH_PERSONAL_ACCESS_TOKEN here and do not pass it to AWS EC2
// userscript so it will keep secret
let reg_token;

// starts AWS EC2 instance that will spawn Github runner for a given label
async function start(param_type, param_label, param_ami, param_spot, param_disk, param_timebomb, param_onejob) {
  const ec2 = new AWS.EC2();

  reg_token = reg_token ? reg_token : await getGithubRegToken();
  const ec2types     = typeof param_type     === 'string' ? [ param_type ] : param_type;
  const label        = typeof param_label    === 'string' ? param_label : param_label[0];
  const ec2ami       = typeof param_ami      !== 'undefined' ? param_ami : "ami-0966bccbb521ccb24";
  const ec2spot      = typeof param_spot     !== 'undefined' ? (param_spot === "false" ? false : true) : true;
  const ec2disk      = typeof param_disk     !== 'undefined' ? param_disk : "/dev/sda1:16";
  const timebomb     = typeof param_timebomb !== 'undefined' ? param_timebomb : "1h";
  const onejob       = typeof param_onejob   !== 'undefined' ? (param_onejob === "false" ? false : true) : true;
  // ephemeral runner will exit after one job so we will terminate instance sooner
  const ephemeral_str = onejob ? "--ephemeral" : "";

  let ec2id; // AWS EC2 instance id
  // last error that will be thrown in case all our attemps in instance creation will fails
  let last_error;
  // loop for spot/ondemand instances
  for (let spot of (ec2spot ? [ 1, 0 ] : [ 0 ])) {
    const spot_str = spot ? "spot" : "on-demand";
    for (let ec2type of ec2types) { // iterate for provided instance types
      const setup_github_actions_runner = [
        `#!/bin/bash -x`, `mkdir actions-runner`, `cd actions-runner`,
        // we can not place runner into AMI image since it is updated often and
        // latest version in required to connect to github
        `export RUNNER_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | sed -n \'s,.*"tag_name": "v\\(.*\\)".*,\\1,p\')`,
        `curl -O -L https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-linux-x64-$RUNNER_VERSION.tar.gz || shutdown -h now`,
        `tar xf ./actions-runner-linux-x64-$RUNNER_VERSION.tar.gz || shutdown -h now`,
        `su gh_runner -c "./config.sh --unattended ${ephemeral_str} --url https://github.com/${repo} --token ${reg_token} --name ${label}_${ec2type}_${spot_str} --labels ${label} --replace || shutdown -h now"`,
        // timebomb to avoid paying for stale AWS instances
        `(sleep ${timebomb}; su gh_runner -c "./config.sh remove --token ${reg_token}"; shutdown -h now) &`,
        `su gh_runner -c "./run.sh"`,
        `su gh_runner -c "./config.sh remove --token ${reg_token}"`,
        // in case we launch insance with InstanceInitiatedShutdownBehavior = "terminate" it will terminate instance here as well
        `shutdown -h now`
      ];
      try {
        let params = {
          ImageId:      ec2ami,
          InstanceType: ec2type,
          UserData:     Buffer.from(setup_github_actions_runner.join('\n')).toString('base64'),
          MinCount:     1,
          MaxCount:     1,
          InstanceInitiatedShutdownBehavior: "terminate",
          TagSpecifications: [
            { ResourceType: "instance", Tags: [ {Key: "Label", Value: label} ] }
          ]
        };
        if (spot) params.InstanceMarketOptions = { MarketType: "spot" };
        if (ec2disk) {
          const items = ec2disk.split(':');
          params.BlockDeviceMappings = [ {DeviceName: items[0], Ebs: {VolumeSize: items[1]}} ];
        }
        const result = await ec2.runInstances(params).promise();
        ec2id = result.Instances[0].InstanceId;
        core.info(`Created AWS EC2 ${spot_str} instance ${ec2id} of ${ec2type} type with ${label} label`);
        break;
      } catch (error) {
        core.warning(`Error creating AWS EC2 ${spot_str} instance of ${ec2type} type with ${label} label`);
        last_error = error;
      }
    }
    // we already created instance and do not need to iterate these loops
    if (ec2id) break;
  }
  if (!ec2id && last_error) {
    core.error(`Error creating AWS EC2 instance with ${label} label`);
    throw last_error;
  }

  // wait untill instance will be found running before continuing (spot instance
  // can be created but never run and will be in pending state untill
  // termination)
  let p = ec2.waitFor("instanceRunning", {
    Filters: [ { Name: "tag:Label", Values: [ label ] } ]
  }).promise();
  for (let i = 0; i < 2; i++) {
    p = p.catch(function() {
      core.warning(`Error searching for running AWS EC2 instance ${ec2id} with ${label} label. Will retry.`);
    }).catch(rejectDelay);
  }
  p = p.then(function() {
    core.info(`Found running AWS EC2 instance ${ec2id} with ${label} label`);
  }).catch(function(error) {
    core.error(`Error searching for running AWS EC2 instance ${ec2id} with ${label} label`);
    throw error;
  });
}

// terminate (completely remove) AWS EC instances (normally one instance) with
// given tag label and also remove all Github actions runners (normally one
// runner) with that label
async function stop(param_label) {
  // last error that will be thrown in case something will break here
  let last_error;
  const ec2 = new AWS.EC2();

  const label = typeof param_label === 'string' ? param_label : param_label[0];

  // find AWS EC2 instances with tag label
  let instances;
  try {
    instances = await ec2.describeInstances({
      Filters: [ { Name: "tag:Label", Values: [ label ] } ]
    }).promise();
    core.info(`Searched for AWS EC2 instance with label ${label}`);
  } catch (error) {
    core.error(`Error searching for AWS EC2 instance with label ${label}: ${error}`);
    last_error = error;
  }

  // remove all found AWS EC2 instances
  if (instances)
    for (const reservation of instances.Reservations) {
      for (const instance of reservation.Instances) {
        try {
          await ec2.terminateInstances({ InstanceIds: [ instance.InstanceId ] }).promise();
          core.info(`Terminated AWS EC2 instance ${instance.InstanceId} with label ${label}`);
        } catch (error) {
          core.error(`Error terminating AWS EC2 instance ${instance.InstanceId} with label ${label}: ${error}`);
          last_error = error;
        }
      }
    }

  // find all Github action runners
  core.info("Preparing Github SDK API");
  const octokit = github.getOctokit(core.getInput("GH_PERSONAL_ACCESS_TOKEN"));
  let runners;
  try {
    runners = await octokit.paginate(`GET /repos/${repo}/actions/runners`);
    core.info(`Searched for Github action runners with label ${label}`);
  } catch (error) {
    core.info(`Error searching for Github action runners with label ${label}`);
    last_error = error;
  }

  // remove Github action runners with specified label
  if (runners)
    for (runner of runners) {
      let label_found = false;
      for (label_obj of runner.labels)
        if (label_obj.name == label) {
          label_found = true;
          break;
        }
      if (!label_found) continue;
      let p = octokit.request(`DELETE /repos/${repo}/actions/runners/${runner.id}`);
      // retry deletion up to 5 times (with 10 seconds delay) sincec Github can
      // not remove runners still marked as active (with running job)
      for (let i = 0; i < 5; i++) {
        p = p.catch(function() {
          core.warning(`Error removing Github self-hosted runner ${runner.id} with ${label}. Will retry.`);
        }).catch(rejectDelay);
      }
      p = p.then(function() {
        core.info(`Removed Github self-hosted runner ${runner.id} with ${label}`);
      }).catch(function(error) {
        core.error(`Error removing Github self-hosted runner ${runner.id} with ${label}: ${error}`);
        last_error = error;
      });
    }

  if (last_error) throw last_error;
}

(async function() {
  try {
    // provide AWS credentials
    AWS.config.update({
      accessKeyId:     core.getInput("AWS_ACCESS_KEY"),
      secretAccessKey: core.getInput("AWS_SECRET_KEY"),
      region:          core.getInput("aws-region")
    });
    // mode is start or stop
    const mode = core.getInput("mode");
    const runs_on_list = core.getInput("runs-on-list") ? JSON.parse(core.getInput("runs-on-list")) : [];

    if (mode == "start") {
      for (let c of runs_on_list) {
        const raw_label = c["runs-on"];
        if (c["aws-type"]) {
          await start(c["aws-type"], raw_label, c["aws-ami"], c["aws-spot"], c["aws-disk"], c["aws-timebomb"], c["one-job"]);
        } else core.info(`Skipping ${raw_label} config`);
      }
    } else if (mode == "stop") {
      // last error that will be thrown in case something will break here
      let last_error;
      for (let c of runs_on_list) {
        const raw_label = c["runs-on"];
        try {
          if (c["aws-type"]) {
            await stop(raw_label);
          } else core.info(`Skipping ${raw_label} config`);
        } catch (error) {
          core.error(`Error removing runner with ${raw_label}: ${error}`);
          last_error = error;
        }
      }
      if (last_error) throw last_error;
    }
  } catch (error) {
    core.error(error);
    core.setFailed(error.message);
  }
})();
