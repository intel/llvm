# Users Manual

This is the list of SYCL specific options supported by compiler and some
examples.

Options marked as [DEPRECATED] are going to be removed in some future updates.
Options marked as [EXPERIMENTAL] are expected to be used only in limited cases
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

    Normally, '-fsycl-targets' is specified when linking an application, in
    which case the AOT compiled device binaries are embedded within the
    application’s fat executable.  However, this option may also be used in
    combination with '-c' and '-fno-sycl-rdc' when compiling a source file.
    In this case, the AOT compiled device binaries are embedded within the fat
    object file.

    The following triples are supported by default:
    * spir64 - this is the default generic SPIR-V target;
    * spir64_x86_64 - generate code ahead of time for x86_64 CPUs;
    * spir64_fpga - generate code ahead of time for Intel FPGA;
    * spir64_gen - generate code ahead of time for Intel Processor Graphics;
    Full target triples can also be used:
    * spir64-unknown-unknown, spir64_x86_64-unknown-unknown,
      spir64_fpga-unknown-unknown, spir64_gen-unknown-unknown
    Available in special build configuration:
    * nvptx64-nvidia-cuda - generate code ahead of time for CUDA target;
    * native_cpu - allows to run SYCL applications with no need of an 
    additional backend (note that this feature is WIP and experimental, and 
    currently overrides all the other specified SYCL targets when enabled.)

    Special target values specific to Intel, NVIDIA and AMD Processor Graphics
    support are accepted, providing a streamlined interface for AOT. Only one of
    these values at a time is supported.
    * intel_gpu_lnl_m, intel_gpu_20_4_4 - Lunar Lake Intel graphics architecture
    * intel_gpu_bmg_g21, intel_gpu_20_1_4 - Battlemage G21 Intel graphics architecture
    * intel_gpu_arl_h, intel_gpu_12_74_4 - Arrow Lake H Intel graphics architecture
    * intel_gpu_mtl_h, intel_gpu_12_71_4 - Meteor Lake H Intel graphics architecture
    * intel_gpu_mtl_u, intel_gpu_mtl_s, intel_gpu_arl_u, intel_gpu_arl_s, intel_gpu_12_70_4 - Meteor Lake U/S or Arrow Lake U/S Intel graphics architecture
    * intel_gpu_pvc_vg, intel_gpu_12_61_7 - Ponte Vecchio VG Intel graphics architecture
    * intel_gpu_pvc, intel_gpu_12_60_7 - Ponte Vecchio Intel graphics architecture
    * intel_gpu_acm_g12, intel_gpu_dg2_g12, intel_gpu_12_57_0 - Alchemist G12 Intel graphics architecture
    * intel_gpu_acm_g11, intel_gpu_dg2_g11, intel_gpu_12_56_5 - Alchemist G11 Intel graphics architecture
    * intel_gpu_acm_g10, intel_gpu_dg2_g10, intel_gpu_12_55_8 - Alchemist G10 Intel graphics architecture
    * intel_gpu_dg1, intel_gpu_12_10_0 - DG1 Intel graphics architecture
    * intel_gpu_adl_n - Alder Lake N Intel graphics architecture
    * intel_gpu_adl_p - Alder Lake P Intel graphics architecture
    * intel_gpu_rpl_s - Raptor Lake Intel graphics architecture (equal to intel_gpu_adl_s)
    * intel_gpu_adl_s - Alder Lake S Intel graphics architecture
    * intel_gpu_rkl - Rocket Lake Intel graphics architecture
    * intel_gpu_tgllp, intel_gpu_tgl, intel_gpu_12_0_0 - Tiger Lake Intel graphics architecture
    * intel_gpu_jsl - Jasper Lake Intel graphics architecture (equal to intel_gpu_ehl)
    * intel_gpu_ehl - Elkhart Lake Intel graphics architecture
    * intel_gpu_icllp, intel_gpu_icl, intel_gpu_11_0_0 - Ice Lake Intel graphics architecture
    * intel_gpu_cml, intel_gpu_9_7_0 - Comet Lake Intel graphics architecture
    * intel_gpu_aml, intel_gpu_9_6_0 - Amber Lake Intel graphics architecture
    * intel_gpu_whl, intel_gpu_9_5_0 - Whiskey Lake Intel graphics architecture
    * intel_gpu_glk, intel_gpu_9_4_0 - Gemini Lake Intel graphics architecture
    * intel_gpu_bxt - Broxton Intel graphics architecture (equal to intel_gpu_apl)
    * intel_gpu_apl, intel_gpu_9_3_0 - Apollo Lake Intel graphics architecture
    * intel_gpu_cfl, intel_gpu_9_2_9 - Coffee Lake Intel graphics architecture
    * intel_gpu_kbl, intel_gpu_9_1_9 - Kaby Lake Intel graphics architecture
    * intel_gpu_skl, intel_gpu_9_0_9 - Intel(R) microarchitecture code name Skylake Intel graphics architecture
    * intel_gpu_bdw, intel_gpu_8_0_0 - Intel(R) microarchitecture code name Broadwell Intel graphics architecture
    * nvidia_gpu_sm_50 - NVIDIA Maxwell architecture (compute capability 5.0)
    * nvidia_gpu_sm_52 - NVIDIA Maxwell architecture (compute capability 5.2)
    * nvidia_gpu_sm_53 - NVIDIA Maxwell architecture (compute capability 5.3)
    * nvidia_gpu_sm_60 - NVIDIA Pascal architecture (compute capability 6.0)
    * nvidia_gpu_sm_61 - NVIDIA Pascal architecture (compute capability 6.1)
    * nvidia_gpu_sm_62 - NVIDIA Pascal architecture (compute capability 6.2)
    * nvidia_gpu_sm_70 - NVIDIA Volta architecture (compute capability 7.0)
    * nvidia_gpu_sm_72 - NVIDIA Volta architecture (compute capability 7.2)
    * nvidia_gpu_sm_75 - NVIDIA Turing architecture (compute capability 7.5)
    * nvidia_gpu_sm_80 - NVIDIA Ampere architecture (compute capability 8.0)
    * nvidia_gpu_sm_86 - NVIDIA Ampere architecture (compute capability 8.6)
    * nvidia_gpu_sm_87 - NVIDIA Jetson/Drive AGX Orin architecture
    * nvidia_gpu_sm_89 - NVIDIA Ada Lovelace architecture
    * nvidia_gpu_sm_90 - NVIDIA Hopper architecture
    * nvidia_gpu_sm_90a - NVIDIA Hopper architecture (with wgmma and setmaxnreg instructions)
    * amd_gpu_gfx700 - AMD GCN GFX7 (Sea Islands (CI)) architecture
    * amd_gpu_gfx701 - AMD GCN GFX7 (Sea Islands (CI)) architecture
    * amd_gpu_gfx702 - AMD GCN GFX7 (Sea Islands (CI)) architecture
    * amd_gpu_gfx801 - AMD GCN GFX8 (Volcanic Islands (VI)) architecture
    * amd_gpu_gfx802 - AMD GCN GFX8 (Volcanic Islands (VI)) architecture
    * amd_gpu_gfx803 - AMD GCN GFX8 (Volcanic Islands (VI)) architecture
    * amd_gpu_gfx805 - AMD GCN GFX8 (Volcanic Islands (VI)) architecture
    * amd_gpu_gfx810 - AMD GCN GFX8 (Volcanic Islands (VI)) architecture
    * amd_gpu_gfx900 - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx902 - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx904 - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx906 - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx908 - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx909 - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx90a - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx90c - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx940 - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx941 - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx942 - AMD GCN GFX9 (Vega) architecture
    * amd_gpu_gfx1010 - AMD GCN GFX10.1 (RDNA 1) architecture
    * amd_gpu_gfx1011 - AMD GCN GFX10.1 (RDNA 1) architecture
    * amd_gpu_gfx1012 - AMD GCN GFX10.1 (RDNA 1) architecture
    * amd_gpu_gfx1013 - AMD GCN GFX10.1 (RDNA 1) architecture
    * amd_gpu_gfx1030 - AMD GCN GFX10.3 (RDNA 2) architecture
    * amd_gpu_gfx1031 - GCN GFX10.3 (RDNA 2) architecture
    * amd_gpu_gfx1032 - GCN GFX10.3 (RDNA 2) architecture
    * amd_gpu_gfx1033 - GCN GFX10.3 (RDNA 2) architecture
    * amd_gpu_gfx1034 - GCN GFX10.3 (RDNA 2) architecture
    * amd_gpu_gfx1035 - GCN GFX10.3 (RDNA 2) architecture
    * amd_gpu_gfx1036 - GCN GFX10.3 (RDNA 2) architecture
    * amd_gpu_gfx1100 - GCN GFX11 (RDNA 3) architecture
    * amd_gpu_gfx1101 - GCN GFX11 (RDNA 3) architecture
    * amd_gpu_gfx1102 - GCN GFX11 (RDNA 3) architecture
    * amd_gpu_gfx1103 - GCN GFX11 (RDNA 3) architecture
    * amd_gpu_gfx1150 - GCN GFX11 (RDNA 3) architecture
    * amd_gpu_gfx1151 - GCN GFX11 (RDNA 3) architecture
    * amd_gpu_gfx1200 - GCN GFX12 (RDNA 4) architecture
    * amd_gpu_gfx1201 - GCN GFX12 (RDNA 4) architecture

## Language options

**`-sycl-std=<value>`** [EXPERIMENTAL]

    SYCL language standard to compile for. Currently the possible value is:
    * 2020 - for SYCL 2020
    It doesn't guarantee specific standard compliance, but some selected
    compiler features change behavior.
    It is under development and not recommended to use in production
    environment.
    Default value is 2020.

**`-f[no-]sycl-unnamed-lambda`**

    Enables/Disables unnamed SYCL lambda kernels support.
    The default value depends on the SYCL language standard: it is enabled
    by default for SYCL 2020.

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
    Enabled by default.

**`-f[no-]sycl-id-queries-fit-in-int`**

    Assume/Do not assume that SYCL ID queries fit within MAX_INT. It assumes
    that these values fit within MAX_INT:
    * id class get() member function and operator[]
    * item class get_id() member function and operator[]
    * nd_item class get_global_id()/get_global_linear_id() member functions
    Enabled by default.

**`-f[no-]sycl-force-inline-kernel-lambda`**

  Enables/Disables inlining of the kernel lambda operator into the compiler
  generated entry point function. This flag does not apply to ESIMD
  kernels.
  Disabled when optimizations are disabled (-O0 or equivalent). Enabled
  otherwise.

**`-fgpu-inline-threshold=<n>`**

    Sets the inline threshold for device compilation to <n>. Note that this
    option only affects the behaviour of the DPC++ compiler, not target-
    specific compilers (e.g. OpenCL/Level Zero/Nvidia/AMD target compilers)
    which may or may not perform additional inlining.
    Default value is 225.

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
    * off - creates a single module for all kernels. If `-fsycl-no-rdc` is
      specified, the behavior is the same as per_source.
    * auto - the compiler will use a heuristic to select the best way of
      splitting device code. This is default mode.

**`-f[no-]sycl-device-code-split-esimd`** [EXPERIMENTAL]

     Controls SYCL/ESIMD device code splitting. When enabled (this is the
     default), SYCL and ESIMD entry points along with their call graphs are
     put into separate device binary images. Otherwise, SYCL and ESIMD parts
     of the device code are kept in the same device binary image and get
     compiled by the Intel GPU compiler back end as a single module. This
     option has effect only for SPIR-based targets and apps containing ESIMD
     kernels.

**`-fsycl-max-parallel-link-jobs=<N>`**

    Experimental feature. When specified, it informs the compiler
    that it can simultaneously spawn up to `N` processes to perform
    actions required to link the DPC++ application. This option is
    only useful in SYCL mode. It only takes effect if link action
    needs to be executed, i.e. it won't have any effect in presence of
    options like `-c` or `-E`. Default value of `N` is 1.

**`-f[no-]sycl-device-lib=<lib1>[,<lib2>,...]`**

    Enables/disables linking of the device libraries. Supported libraries:
    libm-fp32, libm-fp64, libc, all. Use of 'all' will enable/disable all of
    the device libraries.

**`-f[no-]sycl-device-lib-jit-link`** [EXPERIMENTAL]

    Enables/disables jit link mechanism for SYCL device library in JIT
    compilation. If jit link is enabled, all required device libraries will
    be linked with user's device image by SYCL runtime during execution time,
    otherwise the link will happen in build time, jit link is disabled by
    default currently. This option is ignored in AOT compilation.

**`-f[no-]sycl-instrument-device-code`** [EXPERIMENTAL]

    Enables/disables linking of the Instrumentation and Tracing Technology (ITT)
    device libraries for VTune(R). This provides annotations to intercept
    various events inside JIT generated kernels. These device libraries are
    linked in by default.

**`-f[no-]sycl-link-huge-device-code`** [DEPRECATED]

    Place device code later in the linked binary in order to avoid precluding
    32-bit PC relative relocations between surrounding ELF sections when device
    code is larger than 2GiB. This is disabled by default.

    Deprecated in favor of `-f[no-]link-huge-device-code`.

    NOTE: This option is currently only supported on Linux.

**`-fsycl-force-target=<T>`**

    When used along with '-fsycl-targets', force the device object being
    unbundled to match the target <T> given.  This allows the user to override
    the expected unbundling type even though the target given does not match.
    The forced target applies to all objects, archives and default device
    libraries.

**`-f[no-]sycl-rdc`**

    Enables/disables relocatable device code. If relocatable device code is
    disabled, device code cannot use SYCL_EXTERNAL functions, which allows
    the compiler to link device code on a per-translation-unit basis.
    This may result in compile time and compiler memory usage improvements.
    '-fno-sycl-rdc' used along with '-fsycl-max-parallel-link-jobs' will enable
    additional device linking parallism for fat static archives.
    Relocatable device code is enabled by default.

## Intel FPGA specific options

**`-fintelfpga`**

    Perform ahead of time compilation for Intel FPGA. It sets the target to
    FPGA and turns on the debug options that are needed to generate FPGA
    reports. It is functionally equivalent shortcut to
    `-fsycl-targets=spir64_fpga -g -MMD` on Linux and
    `-fsycl-targets=spir64_fpga -Zi -MMD` on Windows.

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

**`-f[no-]sycl-use-bitcode`** [DEPRECATED]

    Emit SYCL device code in LLVM-IR bitcode format. When disabled, SPIR-V is
    emitted.
    Enabled by default.

**`-fsycl-device-obj=<arg>`** [EXPERIMENTAL]

    Specify format of device code stored in the resulting object. The <arg> can
    be one of the following:  "spirv" - SPIR-V is emitted, "llvmir" - LLVM-IR
    bitcode format is emitted (default).

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

**`-fsycl-fp32-prec-sqrt`**

    Enable use of correctly rounded `sycl::sqrt` function as defined by IEE754.
    Without this flag, the default precision requirement for `sycl::sqrt` is 3
    ULP.

    NOTE: This flag is currently only supported with the CUDA and HIP targets.


**`-f[no-]sycl-esimd-force-stateless-mem`** [EXPERIMENTAL]

    Enforces stateless memory access and enables the automatic conversion of
    "stateful" memory access via SYCL accessors to "stateless" within ESIMD
    (Explicit SIMD) kernels.

    -fsycl-esimd-force-stateless-mem disables the intrinsics and methods
    accepting SYCL accessors or "surface-index" which cannot be automatically
    converted to their "stateless" equivalents.

    -fno-sycl-esimd-force-stateless-mem is used to tell compiler not to
    enforce usage of stateless memory accesses. This is the default behavior.

    NOTE: "Stateful" access is the one that uses SYCL accessor or a pair
    of "surface-index" + 32-bit byte-offset and uses specific memory access
    data port messages to read/write/fetch.
    "Stateless" memory access uses memory location represented with virtual
    memory address pointer such as USM pointer.

    The "stateless" memory may be beneficial as it does not have the limit
    of 4Gb per surface.
    Also, some of Intel GPUs or GPU run-time/drivers may support only
    "stateless" memory accesses.

**`-ftarget-compile-fast`** [EXPERIMENTAL]

    Instructs the target backend to reduce compilation time, potentially
    at the cost of runtime performance. Currently only supported on Intel GPUs.

**`-f[no-]target-export-symbols`**

    Exposes exported symbols in a generated target library to allow for
    visibility to other modules.

    NOTE: This flag is only supported for spir64_gen AOT targets.

**`-ftarget-register-alloc-mode=<arg>`**

    Specify a register allocation mode for specific hardware for use by supported
    target backends. The format of the argument is "Device0:Mode0[,Device1:Mode1...]".
    Currently the only supported Device is "pvc". The supported modes are
    "default","small","large", and "auto".

**`-fpreview-breaking-changes`**

    When specified, it informs the compiler driver and compilation phases
    that it is allowed to break backward compatibility. When this option is
    specified the compiler will also set the macro
    __INTEL_PREVIEW_BREAKING_CHANGES.
    When this option is used in conjunction with -fsycl, the driver will link
    against an alternate form of libsycl, libsycl-preview.

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
