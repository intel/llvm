# XPTI Framework Library

Implementation of the instrumentation framework library to support
instrumentation of arbitrary regions of code. This implementation requires the
specification header files used by the proxy library in `xpti/`. This
library is not necessary for building the SYCL runtime library and only required
to build tools that extract the traces from instrumented code.

To see the implementation of the basic collector and how it can be attached to
an application that has been instrumented with XPTI, see [samples/basic_collector/README.md](samples/basic_collector/README.md).

To see how to determine the cost of the APIs, see the tests under [basic_test/](basic_test/README.md).

Unit tests are available under [unit_test](unit_test/README.md).

To see the complete documentation on XPTI framework API, please see [XPTI Framework library documentation](doc/XPTI_Framework.md)