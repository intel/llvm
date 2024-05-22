# Example SYCL Generic Collector

The SYCL runtime exports many different streams and this sample collector will
monitor events from many of the available streams and save them as JSON files
for viewing in Perfetto or chrome://tracing

1. Set the environment variable that indicates that tracing has been enabled.

   This is defined by the variable `XPTI_TRACE_ENABLE`. The possible
   values taken by this environment variable are:

   To enable: `XPTI_TRACE_ENABLE=1` or `XPTI_TRACE_ENABLE=true`

   To disable: `XPTI_TRACE_ENABLE=0` or `XPTI_TRACE_ENABLE=false`

2. Set the environment variable that points to the XPTI framework dispatcher so
   the stub library can dynamically load it and dispatch the calls to the
   dispatcher.
   `XPTI_FRAMEWORK_DISPATCHER=/path/to/libxptifw.[so,dll,dylib]`

3. Set the environment variable that points to the subscriber, which in this
  case is `libsycl_perf__collector.[so,dll,dylib]`.

     `XPTI_SUBSCRIBERS=/path/to/libsycl_perf_collector.[so,dll,dylib]`

In order to simplify this set up process, a helper script (sycl-perf.sh) is also available that allows you to have 
this automated for the most part. It however requires you to set the XPTI_PER_DIR=/path/to/lib where the dispatcher 
and subscriber shared objects are present.

For more detail on the framework, the tests that are provided and their usage,
please consult the [XPTI Framework library documentation](doc/XPTI_Framework.md).
