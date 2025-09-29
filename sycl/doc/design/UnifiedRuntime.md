# Unified Runtime

## Overview
The Unified Runtime project serves as an interface layer between the DPC++
runtime and the device-specific runtime layers which control execution on
devices. The parts of it primarily utilized by DPC++ are its C API, loader
library, and the adapter libraries that implement the API for various backends.

The DPC++ runtime accesses the UR api via the [Plugin](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/plugin.hpp)
object. Each Plugin object owns a `ur_adapter_handle_t`, which represents a UR
backend (e.g. OpenCL, Level Zero, etc).

The picture below illustrates the placement of UR within the overall DPC++
runtime stack. Dotted lines show components or paths which are not yet available
in the runtime, but are likely to be developed.
![UR in DPC++ runtime architecture](images/RuntimeArchitecture.svg)

For detailed information about the UR project including the API specification
see the
[Unified Runtime Documentation](https://oneapi-src.github.io/unified-runtime/core/INTRO.html).
You can find the Unified Runtime repo [here](https://github.com/oneapi-src/unified-runtime).
