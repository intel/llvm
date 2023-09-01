# Unified Runtime

This directory contains the implementation of the PI plugin for Unified Runtime,
including the pi2ur translation layer, as well as the sources for the individual
Unified Runtime adapters.

## Making changes to Unified Runtime
If you introduce changes to PI (e.g. new entry points, new enum values) you
should introduce matching changes to the Unified Runtime spec.

To do this, open a Pull Request adding the changes to the
[Unified Runtime](https://github.com/oneapi-src/unified-runtime)
repository, making sure to follow the
[Contribution Guide](https://oneapi-src.github.io/unified-runtime/core/CONTRIB.html).

When your changes to Unified Runtime are merged, you should:
* Update the UR commit used by changing the `UNIFIED_RUNTIME_TAG` value in
  [`CMakeLists.txt`](CMakeLists.txt)
* Make changes to [`pi2ur.hpp`](pi2ur.hpp) to ensure correct mapping from PI to
  UR
* Make changes to the affected adapter implementations in the
[`ur/adapters`](ur/adapters) folder
