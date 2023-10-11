Discourse topic details:
- Category: [Clang Frontend](https://discourse.llvm.org/c/clang/6)
- Title: "RFC: SYCL driver enhancements"
- Tags: sycl

# Adjust the compiler driver to support offloading for SYCL kernels

Support for the SYCL programming model requires a number of adjustments to the
compiler driver. These changes involve introducing enabling command line
options, adjusting the compilation tool chains to provide device specific
compilations for corresponding host compilations. The behaviors are centered
around compilation of source files. Consumption of objects and libraries that
contain SYCL kernels will be done by the existing `clang-linker-wrapper` that is
used for other languages with offloading support like OpenMP.

## Command line options for enabling.

A number of command line options will be introduced to enable the offloading
compilation for SYCL.
* -fsycl-targets=\<target\>
  * Enables device compilation for each listed comma separated target. Targets
    can consist of target triples and other device specific representations.
    When not provided the default target is `spir64`.
* -fsycl-device-only
  * Enables the ability to create only device code.
* -fsycl-host-compiler=\<compiler\>
  * Provide the ability to use a 3rd party compiler to perform the host
    compilation step.
The above options are to be used with the existing `-fsycl` option.

## Introduction of SYCL specific offload action builder

An additional offloading target toolchain will be created to support the SYCL
device compilations. This will be unique to the existing OpenMP and other
offloading targets and will allow for the creation of unique toolchains specific
to each SYCL target.

## Compilation behaviors

Using the `-fsycl` option will enable the multi-pass compilation approach. This
consists of a device specific compilation and a corresponding host compilation.
The device compilation will also produce an integration header and integration
footer that used during the subsequent host compilation. Additional information
detailing the integration header and footer can be found in "RFC: SYCL host
compiler integration header and footer".

The ability to use a 3rd party compiler is available via the
`-fsycl-host-compiler` option. The argument provided points to the 3rd party
compiler that would be used for the host specific portion of the compile. The
driver is responsible for constructing the compilation call to the 3rd party
compiler to contain the generated integration header and footer.

The `-fsycl-device-only` option provides the user the ability to create only
the device binary as opposed to generating a multi-targeted object as stated
above. Although the generated object file cannot be executed, it is useful to
allow developers to examine the device code for the offload regions.

## Additional details

Generation of the object in the two-step compilation scenario is considered a
multi-targeted object that will be consumed by the clang-linker-wrapper. See
"RFC: Offloading design for SYCL offload kind and SPIR targets" for details.

Integration will be using the existing clang-linker-wrapper functionality to
perform final device and host link steps.
