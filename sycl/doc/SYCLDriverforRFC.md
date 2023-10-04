# Adjust the compiler driver to support offloading for SYCL kernels

Supporting the SYCL programming model requires a number of adjustments to the
compiler driver. These changes involve introducing enabling command line
options, adjusting the compilation tool chains to provide device specific
compilations as well as corresponding host compilations. The behaviors are
centered around compilation of source files, consumption of objects and
libraries that contain SYCL kernels will be done by the `clang-linker-wrapper`.

## Command line options for enabling.

A number of command line options will be introduced to enable the offloading
compilation for SYCL.
* -fsycl
  * Enables the device and host compilations against a given source file.
* -fsycl-targets=\<target\>
  * Compiles the device based on the given target value. 
* -fsycl-device-only
  * Enables the ability to create only device code.
* -fsycl-host-compiler
  * Provide the ability to use a 3rd party compiler to perform the host
    compilation step.

## Introduction of SYCL specific offload action builder

An additional offloading target toolchain will be created to support the SYCL
device compilations. This will be unique to the existing OpenMP and other
offloading targets allowing for the creating of unique toolchains specific
for each SYCL based target being compiled for.

## Implementation of two compilation modes

Two different compilation modes will be provided. By default, when enabled with
the `-fsycl` option, two-step compilation will be performed. The ability to only
produce code for the target device will also be available with the
`-fsycl-device-only` option.

Device only compilation will perform the device specific compilation to enable
the ability to create binaries specific to the device target specified.

Two-step compilation will perform the device compilation as specified above,
but also perform a corresponding host compilation. This also involves passing
along the integration header file that is produced during the device
compilation. This header file is used during the host compilation.

## Additional details

Generation of the object in the two-step compilation scenario is considered a
'fat object' that will be consumed by the clang-linker-wrapper (details covered
separately).

Integration will be using the existing clang-linker-wrapper functionality to
perform final device and host link steps.
