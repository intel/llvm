# SYCL and Unified Runtime Benchmark Runner

Scripts for running benchmarks on SYCL and Unified Runtime.

## Benchmarks

- [Velocity Bench](https://github.com/oneapi-src/Velocity-Bench)
- [Compute Benchmarks](https://github.com/intel/compute-benchmarks/)
- [LlamaCpp Benchmarks](https://github.com/ggerganov/llama.cpp)
- [SYCL-Bench](https://github.com/unisa-hpc/sycl-bench)
- [Gromacs](https://gitlab.com/gromacs/gromacs.git)/[Grappa](https://github.com/graeter-group/grappa)
- [BenchDNN](https://github.com/uxlfoundation/oneDNN/tree/main/tests/benchdnn)

## Requirements

* Built compiler to be used for benchmarks.  
Instructions on where to find releases or how to build from sources can be found [here](https://github.com/intel/llvm).

* [Unified Runtime](https://github.com/intel/llvm/tree/sycl/unified-runtime) installed.  
Path to the UR install directory will be required in case of using UR for benchmarking.

* `Python3` is required to install and run benchmarks.

## Building & Running

```bash
$ git clone https://github.com/intel/llvm.git
$ cd llvm/devops/scripts/benchmarks/
$ pip install -r requirements.txt

$ ./main.py ~/benchmarks_workdir/ --sycl ~/llvm/build/ --ur ~/ur_install --adapter adapter_name
```

This last command will **download and build** everything in `~/benchmarks_workdir/`
using the built compiler located in `~/llvm/build/` and
installed Unified Runtime in directory `~/ur_install`,
and then **run** the benchmarks for `adapter_name` adapter.

The scripts will try to reuse the files stored in `~/benchmarks_workdir/`. 
If any dependant projects binaries are already built, they will not be rebuilt
again if their tags match tags specified by benchmarks source code.

>NOTE: By default `level_zero` adapter is used.

>NOTE: Pay attention to the `--ur` parameter. It points directly to the directory where UR is installed.  
To install Unified Runtime in the predefined location, use the `-DCMAKE_INSTALL_PREFIX`.

UR build and install example:
```
$ cmake -DCMAKE_BUILD_TYPE=Release -S~/llvm/unified-runtime -B~/ur_build -DCMAKE_INSTALL_PREFIX=~/ur_install -DUR_BUILD_ADAPTER_L0=ON -DUR_BUILD_ADAPTER_L0_V2=ON
$ cmake --build ~/ur_build -j $(nproc)
$ cmake --install ~/ur_build
```

## Results

By default, the benchmark results are not stored.  
To store them, use the option `--save <name>`. This will make the results available for comparison during the next benchmark runs.  
To indicate a specific results location, use the option `--results-dir <path>`.

### Comparing results

You can compare benchmark results using `--compare` option. The comparison will be presented in a markdown output file (see below). If you want to calculate the relative performance of the new results against the previously saved data, use `--compare <previously_saved_data>` (i.e. `--compare baseline`). In case of comparing only stored data without generating new results, use `--dry-run --compare <name1> --compare <name2> --relative-perf <name1>`, where `name1` indicates the baseline for the relative performance calculation and `--dry-run` prevents the script for running benchmarks. Listing more than two `--compare` options results in displaying only execution time, without statistical analysis.

>NOTE: Baseline_L0, as well as Baseline_L0v2 (for the level-zero adapter v2) is updated automatically during a nightly job.  
The results
are stored [here](https://oneapi-src.github.io/unified-runtime/performance/).

### Output formats
You can display the results in the form of a HTML file by using `--ouptut-html` and a markdown file by using `--output-markdown`. Due to character limits for posting PR comments, the final content of the markdown file might be reduced. In order to obtain the full markdown output, use `--output-markdown full`.

## Logging

The benchmark runner uses a configurable logging system with different log levels that can be set using the `--log-level` command-line option.

Available log levels:
- `debug`
- `info` (default)
- `warning`
- `error`
- `critical`

To set the log level, use the `--log-level` option:
```bash
./main.py ~/benchmarks_workdir/ --sycl ~/llvm/build/ --log-level debug
```

You can also use the `--verbose` flag, which sets the log level to `debug` and overrides any `--log-level` setting:
```bash
./main.py ~/benchmarks_workdir/ --sycl ~/llvm/build/ --verbose
```

## Additional options

In addition to the above parameters, there are also additional options that help run benchmarks and read the results in a more customized way.

`--preset <option>` - limits the types of benchmarks that are run.

The available benchmarks options are:
* `Full` (BenchDNN, Compute, Gromacs, llama, SYCL, Velocity and UMF benchmarks)
* `SYCL` (Compute, llama, SYCL, Velocity)
* `Minimal` (Compute)
* `Normal` (BenchDNN, Compute, Gromacs, llama, Velocity)
* `Gromacs` (Gromacs)
* `OneDNN` (BenchDNN)
* `Test` (Test Suite)

`--filter <regex>` - allows to set the regex pattern to filter benchmarks by name.

For example `--filter "graph_api_*"`

## Running in CI

The benchmarks scripts are used in a GitHub Actions workflow, and can be automatically executed on a preconfigured system against any Pull Request.

![compute benchmarks](workflow.png "Compute Benchmarks CI job")

To execute the benchmarks in CI, navigate to the `Actions` tab and then go to the `SYCL Run Benchmarks` workflow. Here, you will find a list of previous runs and a "Run workflow" button. Upon clicking the button, you will be prompted to fill in a form to customize your benchmark run. Important field is the `PR number`, which is the identifier for the Pull Request against which you want the benchmarks to run. Instead, you can specify `Commit hash` from within intel/llvm repository, or leave both empty to run benchmarks against the branch/tag the workflow started from (the value from dropdown list at the top).

Once all the information is entered, click the "Run workflow" button to initiate a new workflow run. This will execute the benchmarks and then post the results as a comment on the specified Pull Request.

>NOTE: You must be a member of the `oneapi-src` organization to access these features.

## Requirements
### System

Sobel Filter benchmark:

`$ sudo apt-get install libopencv-dev`

### Compute-runtime and IGC

The scripts have an option to build compute-runtime and all related components from source:

`$ ./main.py ~/benchmarks_workdir/ --compute-runtime [tag] --build-igc`

For this to work, the system needs to have the appropriate dependencies installed.

compute-runtime (Ubuntu):

`$ sudo apt-get install cmake g++ git pkg-config`

IGC (Ubuntu):

`$ sudo apt-get install flex bison libz-dev cmake libc6 libstdc++6 python3-pip`


## Performance Tuning

For stable benchmark results and system configuration recommendations, see the
[Performance Tuning Guide](PERFORMANCE_TUNING.md).

## Contribution

The requirements and instructions above are for building the project from source
without any modifications. To make modifications to the framework, please see the
[Contribution Guide](https://github.com/intel/llvm/blob/sycl/devops/scripts/benchmarks/CONTRIB.md)
for more detailed instructions.
