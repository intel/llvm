These scripts enable a Python virtual environment used for the CI scripts. This Python can also be used for local development as well.

The only requirements of the system are:
* Windows platform
* Git is installed

Once setup, a Python virtual environment with all necessary PIP modules will be enabled. Then you simply run your script as follows (from anywhere):

 ```
 (pyvirtualenv) local> python run.py ...
 ```

 Once you're done you simply deactivate as outlined below.

 **NOTES:**
 * All dependencies will be installed in the *third_party* dir
 * Add new PIP modules by modifying the *pyenv.cmd* file
 * This leverages *Artifactory* and *irepo*

## Setup

### Activate the python virtual environment

```
local> ci\pyenv.cmd up
```
which will give you prompt as follows:
```
*** Python virtual environment enabled ***
Python 3.6.5

(pyvirtualenv) root>
```

### Updating PIP modules in python virtual environment

Ideally, the PIP modules don't change too often. But if Python complains you can try to update the modules with the following command.

```
local> ci\pyenv.cmd update_pip
```

### Deactivate python virtual environment

```
local> ci\pyenv.cmd down
```

### Reset python virtual environment

This will remove all files associated with the python virtual environment. Useful when reloading new pip modules, for instance.

```
local> ci\pyenv.cmd reset
```

## Troubleshooting

### 'doxygen' is not recognized as an internal or external command

Doxygen was added when pyenv.cmd setup the environment so you need to add it to your path by running the following in your shell:

```console
set PATH=<PATH TO YOUR REPO>\one-api\level_zero\.deps\doxygen;%PATH%
```

*Note:* this dependent on the location specified in *third_party\windows_docs.yml* and may change in the future. Refer to the file for correct location.