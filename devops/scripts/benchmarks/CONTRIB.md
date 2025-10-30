# SYCL & UR Benchmark Suite Contribution Guide

## Architecture

The suite is structured around four main components: Suites, Benchmarks, Results, and BenchmarkMetadata.

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
        * `name()`: Returns a unique identifier string for the benchmark across *all* suites. If a benchmark class is instantiated multiple times with different parameters (e.g., "Submit In Order", "Submit Out Of Order"), the `name()` must reflect this uniqueness.
    * **Optional Methods:**
        * `lower_is_better()`: Returns `True` if lower result values are better (default: `True`).
        * `description()`: Provides a short description about the benchmark.
        * `notes()`: Provides additional commentary about the benchmark results (string).
        * `unstable()`: If it returns a string reason, the benchmark is hidden by default and marked unstable.
        * `get_tags()`: Returns a list of string tags (e.g., "SYCL", "UR", "micro", "application"). See `benches/base.py` for predefined tags.
        * `stddev_threshold()`: Returns a custom standard deviation threshold (float) for stability checks, overriding the global default.
        * `display_name()`: Returns a user-friendly name for the benchmark (default: `name()`).
        * `explicit_group()`: Returns an explicit group name for results (string). If not set, results are grouped by the benchmark's `name()`. This is useful for grouping related results in visualizations.
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
        * `command`: The command list used to run the benchmark (`list[str]`).
        * `env`: Environment variables used (`dict[str, str]`).
        * `unit`: The unit of the value (string, e.g., "μs", "GB/s", "token/s").
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
        * `display_name`: Optional user-friendly name for the benchmark (string). Defaults to `name()`.
        * `explicit_group`: Optional explicit group name for results (string). Used to group results in visualizations.

## Dashboard and Visualization

The benchmark suite generates an interactive HTML dashboard that visualizes `Result` objects and their metadata.

### Data Flow from Results to Dashboard

1. **Collection Phase:**
    * Benchmarks generate `Result` objects containing performance measurements.
    * The framework combines these with `BenchmarkMetadata` from benchmarks and suites. The metadata
    is used for defining charts.
    * All data is packaged into a `BenchmarkOutput` object containing runs, metadata, and tags for serialization.

2. **Serialization:**
    * For local viewing (`--output-html local`): Data is written as JavaScript variables in `data.js`.
    These are directly loaded in the HTML dashboard.
    * For remote deployment (`--output-html remote`): Data is written as JSON in `data.json`.
    The `config.js` file contains the URL where the json file is hosted.
    * Historical runs may be separated into archive files for better dashboard load times.

3. **Dashboard Rendering:**
    * JavaScript processes the data to create three chart types:
        * **Historical Results**: Time-series charts showing performance trends over multiple runs. One chart for each unique benchmark scenario.
        * **Historical Layer Comparisons**: Time-series charts for grouped results. Benchmark scenarios that can be directly compared are grouped either by using `explicit_group()` or matching the beginning of their labels with predefined groups.
        * **Comparisons**: Bar charts comparing selected runs side-by-side. Again, based on the `explicit_group()` or labels.

### Chart Types and Result Mapping

**Historical Results (Time-series):**
* One chart per unique `result.label`.
* X-axis: `BenchmarkRun.date` (time).
* Y-axis: `result.value` with `result.unit`.
* Multiple lines for different `BenchmarkRun.name` entries.
* Points include `result.stddev`, `result.git_hash`, and environment info in tooltips.

**Historical Layer Comparisons:**
* Groups related results using `benchmark.explicit_group()` or `result.label` prefixes.
* Useful for comparing different implementations/configurations of the same benchmark.
* Same time-series format but with grouped data.

**Comparisons (Bar charts):**
* Compares selected runs side-by-side.
* X-axis: `BenchmarkRun.name`.
* Y-axis: `result.value` with `result.unit`.
* One bar per selected run.

### Dashboard Features Controlled by Results/Metadata

**Visual Properties:**
* **Chart Title**: `metadata.display_name` or `result.label`.
* **Y-axis Range**: `metadata.range_min` and `range_max` (when custom ranges enabled).
* **Direction Indicator**: `result.lower_is_better` (shows "Lower/Higher is better").
* **Grouping**: `benchmark.explicit_group()` groups related results together.

**Filtering and Organization:**
* **Suite Filters**: Filter by `result.suite`.
* **Tag Filters**: Filter by `metadata.tags`.
* **Regex Search**: Search by `result.label` patterns, `metadata.display_name` patterns are not searchable.
* **Stability**: Hide/show based on `metadata.unstable`.

**Information Display:**
* **Description**: `metadata.description` appears prominently above charts.
* **Notes**: `metadata.notes` provides additional context (toggleable).
* **Tags**: `metadata.tags` displayed as colored badges with descriptions.
* **Command Details**: Shows `result.command` and `result.env` in expandable sections.
* **Git Information**: `result.git_url` and `result.git_hash` for benchmark source tracking.

### Dashboard Interaction

**Run Selection:**
* Users select which `BenchmarkRun.name` entries to compare.
* Default selection uses `BenchmarkOutput.default_compare_names`.
* Changes affect all chart types simultaneously.

**URL State Preservation:**
* All filters, selections, and options are preserved in URL parameters.
* Enables sharing specific dashboard views via URL address copy.

### Best Practices for Dashboard-Friendly Results

**Naming:**
* Use unique `result.label` names that will be most descriptive.
* Consider `metadata.display_name` for prettier chart titles.
* Ensure `benchmark.name()` is unique across all suites.

**Grouping:**
* Use `benchmark.explicit_group()` to group related measurements.
* Ensure grouped results have the same `result.unit`.
* Group metadata keys in `Suite.additional_metadata()` should match group prefixes.

**Metadata:**
* Provide `metadata.description` for user understanding.
* Use `metadata.notes` for implementation details or caveats.
* Tag with relevant `metadata.tags` for filtering.
* Set `metadata.range_min`/`range_max` for consistent comparisons when needed.

**Stability:**
* Mark unstable benchmarks with `metadata.unstable` to hide them by default.

## Code Style Guidelines

### Benchmark Class Structure

When creating benchmark classes, follow this consistent structure pattern:

**1. Constructor (`__init__`):**
* Assign all parameters to protected (prefixed with `_`) or private (prefixed with `__`) instance variables.
* Set `self._iterations_regular` and `self._iterations_trace` BEFORE calling `super().__init__()` (required for subclasses of `ComputeBenchmark`).

**2. Method Order:**
* Align with methods order as in the abstract base class `Benchmark`. Not all of them are required, but follow the order for consistency.
* Public methods first, then protected, then private.

### Naming Conventions

**Method Return Values:**
* `name()`: Unique identifier with underscores, lowercase, includes all distinguishing parameters
  * Example: `"api_overhead_benchmark_sycl SubmitKernel in order with measure completion"`
* `display_name()`: User-friendly, uses proper capitalization, commas for readability, used for charts titles
  * Example: `"SYCL SubmitKernel in order, with measure completion, NumKernels 10"`

**Class method names and variables should follow PEP 8 guidelines.**
* Use lowercase with underscores for method names and variables.
* Use single underscores prefixes for protected variables/methods and double underscores for private variables/methods.

### Description Writing

Descriptions should:
* Clearly state what is being measured
* Include key parameters and their values
* Explain the purpose or what the benchmark tests
* Be 1-3 sentences, clear and concise
* If not needed, can be omitted

### Tag Selection

* Use predefined tags from `benches/base.py` when available
* Tags should be lowercase, descriptive, single words

## Adding New Benchmarks

1. **Create Benchmark Class:** Implement a new class inheriting from `benches.base.Benchmark`. Implement required methods (`run`, `name`) and optional ones (`description`, `get_tags`, etc.) as needed. Follow the code style guidelines above.
2. **Add to Suite:**
    * If adding to an existing category, modify the corresponding `Suite` class (e.g., `benches/compute.py`) to instantiate and return your new benchmark in its `benchmarks()` method.
    * If creating a new category, create a new `Suite` class inheriting from `benches.base.Suite`. Implement `name()` and `benchmarks()`. Add necessary `setup()` if the suite requires shared setup. Add group metadata via `additional_metadata()` if needed.
3. **Register Suite:** Import and add your new `Suite` instance to the `suites` list in `main.py`.
4. **Add to Presets:** If adding a new suite, add its `name()` to the relevant lists in `presets.py` (e.g., "Full", "Normal") so it runs with those presets. Update `README.md` and benchmarking workflow to include the new suite in presets' description/choices.

## Recommendations

* **Keep benchmarks short:** Ideally under a minute per run. The framework runs benchmarks multiple times (`--iterations`, potentially repeated up to `--iterations-stddev` times).
* **Ensure determinism:** Minimize run-to-run variance. High standard deviation (`> stddev_threshold`) triggers reruns.
* **Handle configuration:** If a benchmark requires specific hardware/software, detect it in `setup()` and potentially skip gracefully if requirements aren't met (e.g., return an empty list from `run` or don't add it in the Suite's `benchmarks()` method).
* **Use unique names:** Ensure `benchmark.name()` and `result.label` are descriptive and unique.
* **Group related results:** Use `benchmark.explicit_group()` consistently for results you want to compare directly in outputs. Ensure units match within a group. If defining group-level metadata in the Suite, ensure the chosen explicit_group name starts with the corresponding key defined in additional_metadata.
* **Test locally:** Before submitting changes, test with relevant drivers/backends (e.g., using `--compute-runtime --build-igc` for L0). Check the visualization locally if possible (--output-markdown --output-html, then open the generated files).
* **Test dashboard visualization:** When adding new benchmarks, always generate and review the HTML dashboard to ensure:
    * Chart titles and labels are clear and readable.
    * Results are grouped logically using `explicit_group()`.
    * Metadata (description, notes, tags) displays correctly.
    * Y-axis ranges are appropriate (consider setting `range_min`/`range_max` if needed).
    * Filtering by suite and tags works as expected.
    * Time-series trends make sense for historical data.
    * **Tip**: Use `--dry-run --output-html local` to regenerate the dashboard without re-running benchmarks. This uses existing historical data and is useful for testing metadata changes, new groupings, or dashboard improvements.

## Utilities

* **`git_project.GitProject`:** Manages git repository cloning, building, and installation for benchmark suites:
    * Automatically clones repositories to a specified directory and checks out specific commits/refs.
    * Provides standardized directory structure with `src_dir`, `build_dir`, and `install_dir` properties.
    * Handles incremental updates - only re-clones if the target commit has changed.
    * Supports force rebuilds and custom directory naming via constructor options.
    * Provides `configure()`, `build()`, and `install()` methods for CMake-based projects.
    * Use this for benchmark suites that need to build from external git repositories (e.g., `ComputeBench`, `VelocityBench`).
* **`utils.utils`:** Provides common helper functions:
    * `run()`: Executes shell commands with environment setup (SYCL paths, LD_LIBRARY_PATH).
    * `download()`: Downloads files via HTTP, checks checksums, optionally extracts tar/gz archives.
    * `prepare_workdir()`: Sets up the main working directory.
* **`utils.oneapi`:** Provides the `OneAPI` singleton class (`get_oneapi()`). Downloads and installs specified oneAPI components (oneDNN, oneMKL) into the working directory if needed, providing access to their paths (libs, includes, CMake configs). Use this if your benchmark depends on these components instead of requiring a system-wide install.
* **`options.py`:** Defines and holds global configuration options, populated by `argparse` in `main.py`. Use options instead of defining your own global variables.
* **`presets.py`:** Defines named sets of suites (`enabled_suites()`) used by the `--preset` argument.
