# SYCL & UR Benchmark Suite Contribution Guide

## Architecture

The suite is structured around three main components: Suites, Benchmarks, and Results.

1. **Suites:**
    * Collections of related benchmarks (e.g., `ComputeBench`, `LlamaCppBench`).
    * Must implement the `Suite` base class (`benches/base.py`).
    * Responsible for shared setup (`setup()`) if needed.
    * Provide a list of `Benchmark` instances via `benchmarks()`.
    * Define a unique `name()`.
    * Can provide additional group-level metadata via `additional_metadata()`.
        * This method should return a dictionary mapping string keys to BenchmarkMetadata objects (with type="group").
        * The keys in this dictionary are used by the dashboard as prefixes to associate group-level metadata with benchmark results (e.g., "Submit" group key will match both "Submit In Order" and "Submit Out Of Order").

2. **Benchmarks:**
    * Represent a single benchmark, usually mapping to a binary execution.
    * Must implement the `Benchmark` base class (`benches/base.py`).
    * **Required Methods:**
        * `setup()`: Initializes the benchmark (e.g., build, download data). Use `self.download()` for data dependencies. **Do not** perform setup in `__init__`.
        * `run(env_vars)`: Executes the benchmark binary (use `self.run_bench()`) and returns a list of `Result` objects. Can be called multiple times, must produce consistent results.
        * `teardown()`: Cleans up resources. Can be empty. No need to remove build artifacts or downloaded datasets.
        * `name()`: Returns a unique identifier string for the benchmark across *all* suites. If a benchmark class is instantiated multiple times with different parameters (e.g., "Submit In Order", "Submit Out Of Order"), the `name()` must reflect this uniqueness.
    * **Optional Methods:**
        * `lower_is_better()`: Returns `True` if lower result values are better (default: `True`).
        * `description()`: Provides a short description about the benchmark.
        * `notes()`: Provides additional commentary about the benchmark results (string).
        * `unstable()`: If it returns a string reason, the benchmark is hidden by default and marked unstable.
        * `get_tags()`: Returns a list of string tags (e.g., "SYCL", "UR", "micro", "application"). See `benches/base.py` for predefined tags.
        * `stddev_threshold()`: Returns a custom standard deviation threshold (float) for stability checks, overriding the global default.
    * **Helper Methods (Base Class):**
        * `run_bench(command, env_vars, ld_library=[], add_sycl=True)`: Executes a command with appropriate environment setup (UR adapter, SYCL paths, extra env vars/libs). Returns stdout.
        * `download(name, url, file, ...)`: Downloads and optionally extracts data dependencies into the working directory.
        * `create_data_path(name, ...)`: Creates a path for benchmark data dependencies.
    * **Metadata:** Benchmarks generate static metadata via `get_metadata()`, which bundles description, notes, tags, etc.

3. **Results:**
    * Store information about a single benchmark execution instance. Defined by the `Result` dataclass (`utils/result.py`).
    * **Fields (set by Benchmark):**
        * `label`: Unique identifier for this *specific result type* within the benchmark instance (e.g., "Submit In Order Time"). Ideally contains `benchmark.name()`.
        * `value`: The measured numerical result (float).
        * `unit`: The unit of the value (string, e.g., "μs", "GB/s", "token/s").
        * `command`: The command list used to run the benchmark (`list[str]`).
        * `env`: Environment variables used (`dict[str, str]`).
        * `stdout`: Full standard output of the benchmark run (string).
        * `passed`: Boolean indicating if verification passed (default: `True`).
        * `explicit_group`: Name for grouping results in visualization (string). Benchmarks in the same group are compared in tables/charts. Ensure consistent units and value ranges within a group.
        * `stddev`: Standard deviation, if calculated by the benchmark itself (float, default: 0.0).
        * `git_url`, `git_hash`: Git info for the benchmark's source code (string).
    * **Fields (set by Framework):**
        * `name`: Set to `label` by the framework.
        * `lower_is_better`: Copied from `benchmark.lower_is_better()`.
        * `suite`: Name of the suite the benchmark belongs to.
        * `stddev`: Calculated by the framework across iterations if not set by the benchmark.

4. **BenchmarkMetadata:**
    * Stores static properties applicable to all results of a benchmark or group. Defined by the `BenchmarkMetadata` dataclass (`utils/result.py`).
    * **Fields:**
        * `type`: "benchmark" (auto-generated) or "group" (manually specified in `Suite.additional_metadata()`).
        * `description`: Displayed description (string).
        * `notes`: Optional additional commentary (string).
        * `unstable`: Reason if unstable, otherwise `None` (string).
        * `tags`: List of associated tags (`list[str]`).
        * `range_min`, `range_max`: Optional minimum/maximum value for the Y-axis range in charts. Defaults to `None`, with range determined automatically.

## Adding New Benchmarks

1. **Create Benchmark Class:** Implement a new class inheriting from `benches.base.Benchmark`. Implement required methods (`setup`, `run`, `teardown`, `name`) and optional ones (`description`, `get_tags`, etc.) as needed.
2. **Add to Suite:**
    * If adding to an existing category, modify the corresponding `Suite` class (e.g., `benches/compute.py`) to instantiate and return your new benchmark in its `benchmarks()` method.
    * If creating a new category, create a new `Suite` class inheriting from `benches.base.Suite`. Implement `name()` and `benchmarks()`. Add necessary `setup()` if the suite requires shared setup. Add group metadata via `additional_metadata()` if needed.
3. **Register Suite:** Import and add your new `Suite` instance to the `suites` list in `main.py`.
4. **Add to Presets:** If adding a new suite, add its `name()` to the relevant lists in `presets.py` (e.g., "Full", "Normal") so it runs with those presets.

## Recommendations

* **Keep benchmarks short:** Ideally under a minute per run. The framework runs benchmarks multiple times (`--iterations`, potentially repeated up to `--iterations-stddev` times).
* **Ensure determinism:** Minimize run-to-run variance. High standard deviation (`> stddev_threshold`) triggers reruns.
* **Handle configuration:** If a benchmark requires specific hardware/software, detect it in `setup()` and potentially skip gracefully if requirements aren't met (e.g., return an empty list from `run` or don't add it in the Suite's `benchmarks()` method).
* **Use unique names:** Ensure `benchmark.name()` and `result.label` are descriptive and unique.
* **Group related results:** Use `result.explicit_group` consistently for results you want to compare directly in outputs. Ensure units match within a group. If defining group-level metadata in the Suite, ensure the chosen explicit_group name starts with the corresponding key defined in additional_metadata.
* **Test locally:** Before submitting changes, test with relevant drivers/backends (e.g., using `--compute-runtime --build-igc` for L0). Check the visualization locally if possible (--output-markdown --output-html, then open the generated files).

## Utilities

* **`utils.utils`:** Provides common helper functions:
    * `run()`: Executes shell commands with environment setup (SYCL paths, LD_LIBRARY_PATH).
    * `git_clone()`: Clones/updates Git repositories.
    * `download()`: Downloads files via HTTP, checks checksums, optionally extracts tar/gz archives.
    * `prepare_workdir()`: Sets up the main working directory.
    * `create_build_path()`: Creates a clean build directory.
* **`utils.oneapi`:** Provides the `OneAPI` singleton class (`get_oneapi()`). Downloads and installs specified oneAPI components (oneDNN, oneMKL) into the working directory if needed, providing access to their paths (libs, includes, CMake configs). Use this if your benchmark depends on these components instead of requiring a system-wide install.
* **`options.py`:** Defines and holds global configuration options, populated by `argparse` in `main.py`. Use options instead of defining your own global variables.
* **`presets.py`:** Defines named sets of suites (`enabled_suites()`) used by the `--preset` argument.
