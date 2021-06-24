# Users Manual

This is the list of SYCL specific options supported by compiler and some
examples.

Options marked as [DEPRECATED] are going to be removed in some future updates.
Options marked as [EXPERIMENTAL] are expected to be used only in limitted cases
and not recommended to use in production environment.

## Generic options

**`-fsycl`**

    General enabling option for SYCL compilation and linking mode. List of
    targets can be specified with `-fsycl-targets`. This is fundamental option
    for any SYCL compilation. All other SYCL specific options require it.

**`-fsycl-targets=<T1>[,...,<Tn>]`**

    Enables ahead of time (AOT) compilation for specified device targets. T is
    a compiler target triple string, representing a target device architecture.
    You can specify more than one target, comma separated. Default just in time
    (JIT) compilation target can be added to the list to produce a combination
    of AOT and JIT code in the resulting fat binary.
    The following triples are supported by default:
    * spir64-unknown-unknown-sycldevice - this is the default generic SPIR-V
      target;
    * spir64_x86_64-unknown-unknown-sycldevice - generate code ahead of time
      for x86_64 CPUs;
    * spir64_fpga-unknown-unknown-sycldevice - generate code ahead of time for
      Intel FPGA;
    * spir64_gen-unknown-unknown-sycldevice - generate code ahead of time for
      Intel Processor Graphics;
    Shorter aliases of the above triples can also be used:
    * spir64, spir64_x86_64, spir64_fpga, spir64_gen
    Available in special build configuration:
    * nvptx64-nvidia-cuda-sycldevice - generate code ahead of time for CUDA
      target;

## Language options

**`-sycl-std=<value>`** [EXPERIMENTAL]

    SYCL language standard to compile for. Possible values:
    * 121 - SYCL 1.2.1
    * 2020 - SYCL 2020
    It doesn't guarantee specific standard compliance, but some selected
    compiler features change behavior.
    It is under development and not recommended to use in production
    environment.
    Default value is 2020.

**`-f[no-]sycl-unnamed-lambda`**

    Enables/Disables unnamed SYCL lambda kernels support.
    Disabled by default.

**`-f[no-]sycl-explicit-simd`** [DEPRECATED]

    The option was used to enable/disable SYCL explicit SIMD extension.
    Not used anymore.

## Optimization options

**`-f[no-]sycl-early-optimizations`**

    Enables (or disables) intermediate representation optimization pipeline
    before translation to SPIR-V. Have effect only if optimizations are turned
    on by standard compiler options (-O1 or higher).
    Enabled by default.

**`-f[no-]sycl-dead-args-optimization`**

    Enables (or disables) LLVM IR dead argument elimination pass to remove
    unused arguments for the kernel functions before translation to SPIR-V.
    Currently has effect only on spir64\* targets.
    Disabled by default.

**`-f[no-]sycl-id-queries-fit-in-int`**

    Assume/Do not assume that SYCL ID queries fit within MAX_INT. It assumes
    that these values fit within MAX_INT:
    * id class get() member function and operator[]
    * item class get_id() member function and operator[]
    * nd_item class get_global_id()/get_global_linear_id() member functions
    Enabled by default.

## Target toolchain options

**`-Xsycl-target-backend=<T> "options"`**
**`-Xs "options"`**

    Pass "options" to the backend of target device compiler, specified by
    triple T. The backend of device compiler generates target machine code from
    intermediate representation. This option can be used to tune code
    generation for a specific target. The "options" are used during AOT
    compilation. For JIT compilation "options" are saved in a fat binary and
    used when code is JITed during runtime.
    -Xs is a shortcut to pass "options" to all backends specified via the
    '-fsycl-targets' option (or default one).

**`-Xsycl-target-frontend=<T> "options"`**

    Pass "options" to the frontend of target device compiler, specified by
    triple T. This option can be used to control of intermediate representation
    generation during offline or online compilation.

**`-Xsycl-target-linker=<T> "options"`**

    Pass "options" to the device code linker, when linking multiple device
    object modules. T is specific target device triple.

## Link options

**`-fsycl-link`**

    Link device object modules and wrap those into a host-compatible object
    module that can be linked later by any standard host linker into the final
    fat binary.

**`-fsycl-link-targets=<T1,...,Tn>`** [DEPRECATED]

    Specify comma-separated list of triples SYCL offloading targets to produce
    linked device images. Used in a link step to link device code for given
    targets and output multiple linked device code images, whose names consist
    of the common prefix taken from the -o option and the triple string.
    Does not produce fat binary and must be used together with -fsycl.

**`-fsycl-add-targets=<T1:file1...Tn:filen>`** [DEPRECATED]

    Add arbitrary device images to the fat binary being linked

    Specify comma-separated list of triple and device binary image file name
    pairs to add to the final SYCL binary. Tells clang to include given set of
    device binaries into the fat SYCL binary when linking; the option value is
    a set of pairs triple,filename - filename is treated as the device binary
    image for the target triple it is paired with, and offload bundler is
    invoked to do the actual bundling.

**`-foffload-static-lib=<lib>`** [DEPRECATED]

    Link with fat static library.

    Link with <lib>, which is a fat static archive containing fat objects which
    correspond to the target device. When linking clang will extract the device
    code from the objects contained in the library and link it with other
    device objects coming from the individual fat objects passed on the command
    line.
    NOTE:  Any libraries that are passed on the command line which are not
    specified with `-foffload-static-lib` are treated as host libraries and are
    only used during the final host link.

**`-foffload-whole-static-lib=<lib>`** [DEPRECATED]

    Similar to `-foffload-static-lib` but uses the whole archive when
    performing the device code extraction.  This is helpful when creating
    shared objects from fat static archives.

**`-fsycl-device-code-split=<mode>`**

    Specifies SYCL device code module assembly. Mode is one of the following:
    * per_kernel - creates a separate device code module for each SYCL kernel.
      Each device code module will contain a kernel and all its dependencies,
      such as called functions and used variables.
    * per_source - creates a separate device code module for each source
      (translation unit). Each device code module will contain a bunch of
      kernels grouped on per-source basis and all their dependencies, such as
      all used variables and called functions, including the `SYCL_EXTERNAL`
      macro-marked functions from other translation units.
    * off - creates a single module for all kernels.
    * auto - the compiler will use a heuristic to select the best way of
      splitting device code. This is default mode.

**`-f[no-]sycl-device-lib=<lib1>[,<lib2>,...]`**

    Enables/disables linking of the device libraries. Supported libraries:
    libm-fp32, libm-fp64, libc, all. Use of 'all' will enable/disable all of
    the device libraries.

## Intel FPGA specific options

**`-fintelfpga`**

    Perform ahead of time compilation for Intel FPGA. It sets the target to
    FPGA and turns on the debug options that are needed to generate FPGA
    reports. It is functionally equivalent shortcut to
    `-fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -g -MMD` on Linux
    and `-fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -Zi -MMD` on
    Windows.

**`-fsycl-link=<output>`**

    Controls FPGA target binary output format. Same as -fsycl-link, but
    optional output can be one of the following:
    * early - generate html reports and an intermediate object file that avoids
    a full Quartus compile. Usually takes minutes to generate. Link can later
    be resumed from this point using -fsycl-link=image.
    * image - generate a bitstream which is ready to be linked and used on a
    FPGA board. Usually takes hours to generate.

**`-reuse-exe=<exe>`**

    Speed up FPGA backend compilation if the device code in <binary> is
    unchanged. If it's safe to do so the compiler will re-use the device binary
    embedded within it. This can be used to minimize or avoid long Quartus
    compile times for FPGA targets when the device code is unchanged.

## Other options

**`-fsycl-device-only`**

    Compile only device part of the code and ignore host part.

**`-f[no-]sycl-use-bitcode`** [EXPERIMENTAL]

    Emit SYCL device code in LLVM-IR bitcode format. When disabled, SPIR-V is
    emitted.
    Enabled by default.

**`-fsycl-help[=backend]`**

    Emit help information from device compiler backend. Backend can be one of
    the following: "x86_64", "fpga", "gen", or "all". Specifying "all" is the
    same as specifying -fsycl-help with no argument and emits help for all
    backends.

**`-fsycl-host-compiler=<arg>`**

    Informs the compiler driver that the host compilation step that is performed
    as part of the greater compilation flow will be performed by the compiler
    <arg>.  It is expected that <arg> is the compiler to be called, either by
    name (in which the PATH will be used to discover it) or a fully qualified
    directory with compiler to invoke.  This option is only useful when -fsycl
    is provided on the command line.

**`-fsycl-host-compiler-options="opts"`**

    Passes along the space separated quoted "opts" string as option arguments
    to the compiler specified with the -fsycl-host-compiler=<arg> option.  It is
    expected that the options used here are compatible with the compiler
    specified via -fsycl-host-compiler=<arg>.

    NOTE: Using -fsycl-host-compiler-options to pass any kind of phase limiting
    options (e.g. -c, -E, -S) may interfere with the expected output set during
    the host compilation.  Doing so is considered undefined behavior.

# Example: SYCL device code compilation

To invoke SYCL device compiler set `-fsycl-device-only` flag.

```console
$ clang++ -fsycl-device-only sycl-app.cpp -o sycl-app.bc
```

By default the output format for SYCL device is LLVM bytecode.

`-fno-sycl-use-bitcode` can be used to emit device code in SPIR-V format.

```console
$ clang++ -fsycl-device-only -fno-sycl-use-bitcode sycl-app.cpp -o sycl-app.spv
```
