# Overview

The SYCL* Compiler contains many options to generate the desired binaries for
your application.

## SYCL specific command line options

**`-fsycl`**

    General enabling option for SYCL compilation mode.  This option enables
    offload compilation for a given target using the `-fsycl-targets` option.
    When the `-fsycl-targets` option is not provided, the default triple is
    `spir64-unknown-unknown-sycldevice`.

**`-fsycl-targets=<value>`**

    A comma separated list of triples to specify the device target(s) to generate
    code for.  This option is only valid when used with `-fsycl`.

### Target toolchain options.

**`-Xsycl-target-backend=<triple> <arg>`**

    Pass <arg> to the SYCL based backend identified by <triple>.

**`-Xsycl-target-backend <arg>`**

    Pass <arg> to the SYCL based target backend.

**`-Xsycl-target-frontend=<triple> <arg>`**

    Pass <arg> to the SYCL based target frontend identified by <triple>.

**`-Xsycl-target-frontend <arg>`**

    Pass <arg> to the SYCL based target frontend.

**`-Xsycl-target-linker=<triple> <arg>`**

    Pass <arg> to the SYCL based target linker identified by <triple>.

**`-Xsycl-target-linker <arg>`**

    Pass <arg> to the SYCL based target linker.

### Link options

**`-fsycl-link`**

    Generate partially linked device object to be used with the host link.

**`-fsycl-link-targets=<T1,...,Tn>`**

    Specify comma-separated list of triples SYCL offloading targets to produce
    linked device images. Used in a link step to link device code for given
    targets and output multiple linked device code images, whose names consist
    of the common prefix taken from the -o option and the triple string.
    Does not produce fat binary and must be used together with -fsycl.

**`-fsycl-add-targets=<T1:file1...Tn:filen>`**

    Add arbitrary device images to the fat binary being linked

    Specify comma-separated list of triple and device binary image file name
    pairs to add to the final SYCL binary. Tells clang to include given set of
    device binaries into the fat SYCL binary when linking; the option value is
    a set of pairs triple,filename - filename is treated as the device binary
    image for the target triple it is paired with, and offload bundler is
    invoked to do the actual bundling.

**`-foffload-static-lib=<lib>`**

    Link with fat static library.

    Link with <lib>, which is a fat static archive containing fat objects which
    correspond to the target device. When linking clang will extract the device
    code from the objects contained in the library and link it with other
    device objects coming from the individual fat objects passed on the command
    line.
    NOTE:  Any libraries that are passed on the command line which are not
    specified with `-foffload-static-lib` are treated as host libraries and are
    only used during the final host link.

### Intel FPGA specific options

**`-fintelfpga`**

    Perform ahead of time compilation for FPGA.

**`-fsycl-link=<value>`**

    Generate partially linked device and host object to be used at various
    stages of compilation. Takes the device binary(s) generated from a `-fsycl`
    enabled compilation and wrap to create a host linkable object. This option
    is enabled only in ahead of time compilation mode fore FPGA (i.e. when
    `-fintelfpga` is set).

**`-reuse-exe=<exe>`**

    Speed up FPGA aoc compile if the device code in <exe> is unchanged.

### Other options

**`-fsycl-device-only`**

    Compile only SYCL device code.

**`-fsycl-use-bitcode`**

    Emit SYCL device code in LLVM-IR bitcode format. When disabled, SPIR-V is
    emitted. Default is true.

**`-fno-sycl-use-bitcode`**

    Use SPIR-V instead of LLVM bitcode in fat objects.

**`-sycl-std=<value>`**

    SYCL language standard to compile for.

**`-fsycl-help`**

    Emit help information from all of the offline compilation tools.

**`-fsycl-help=<value>`**

    Emit help information from the offline compilation tool associated with the
    given architecture argument. Supported architectures: `x86_64`, `fpga` and
    `gen`.

**`-fsycl-unnamed-lambda`**

    Allow unnamed SYCL lambda kernels.

# SYCL device code compilation

To invoke SYCL device compiler set `-fsycl-device-only` flag.

```console
$ clang++ -fsycl-device-only sycl-app.cpp -o sycl-app.bc
```

By default the output format for SYCL device is LLVM bytecode.

`-fno-sycl-use-bitcode` can be used to emit device code in SPIR-V format.

```console
$ clang++ -fsycl-device-only -fno-sycl-use-bitcode sycl-app.cpp -o sycl-app.spv
```

# Static archives with SYCL device code

The SYCL Compiler contains support to create and use static archives that
contain device enabled fat objects.

## Build your objects

```console
$ clang++ -fsycl sycl-app1.cpp sycl-app2.cpp -c
```

## Create the static archive

Build the static archive in the same manner as you would any other normal
static archive, using the objects that were created using the above step.

```console
$ ar cr libsyclapp.a sycl-app1.o sycl-app2.o
```

## Use the static archive

Once you have created the archive, you can use it when creating your final
application.  The fat archives are treated differently than a regular archive
so the option `-foffload-static-lib` is used to signify the needed behavior.

```console
$ clang++ -fsycl sycl-main.cpp -foffload-static-lib=libsyclapp.a
```

Use of `-foffload-static-lib` is required or the library will be treated as
a normal archive.
