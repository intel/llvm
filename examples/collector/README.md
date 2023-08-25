# Unified Runtime XPTI collector example

This example demonstrates how to use the tracing layer integrated into
Unified Runtime. It leverages XPTI to intercept all UR calls in the traced
application and prints information about each function to standard output.

## Running the example

To run the example, you will need to build Unified Runtime with
tracing enabled (`-DUR_ENABLE_TRACING=ON`), link it with your application,
and then set appropriate environment variables for the collector (`XPTI_SUBSCRIBERS`) and the
XPTI dispatcher (`XPTI_FRAMEWORK_DISPATCHER`) to be loaded and enabled (`XPTI_TRACE_ENABLE`).
You will also need to setup your chosen adapter (`UR_ADAPTERS_FORCE_LOAD`).

For example, to run the `hello_world` example with the collector, use:

```
$ mkdir build
$ cd build
$ cmake .. -DUR_ENABLE_TRACING=ON
$ make
$ UR_ADAPTERS_FORCE_LOAD=./lib/libur_adapter_null.so UR_ENABLE_LAYERS=UR_LAYER_TRACING XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=./lib/libxptifw.so XPTI_SUBSCRIBERS=./lib/libcollector.so ./bin/hello_world
```

See [XPTI framework documentation](https://github.com/intel/llvm/blob/sycl/xptifw/doc/XPTI_Framework.md) for more information.
