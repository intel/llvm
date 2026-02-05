// Enable time tracing capability in SYCL applications.

// This test verifies that Clang driver correctly propagates time-trace related options
// during a compile-only invocation and enables JSON time-trace output.
// RUN: mkdir d e f && cp %s d/a.cpp && touch d/b.c
// RUN: %clang -### -fsycl --offload-new-driver -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -o e/a.o 2>&1 | FileCheck %s --check-prefixes=SYCL-DEVICE-COMPILE,SYCL-HOST-COMPILE
// SYCL-DEVICE-COMPILE: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace={{.*}}.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// SYCL-HOST-COMPILE: -cc1{{.*}} "-fsycl-is-host"{{.*}} "-ftime-trace=e/a.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// Verify that the Clang driver generates JSON time-trace output for compile-only
// invocation and propagates the time-trace options, respecting the specified dump directory.
// RUN: %clang -### -fsycl --offload-new-driver -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -dumpdir f/ 2>&1 | FileCheck %s --check-prefixes=SYCL-DEVICE-DUMPDIR,SYCL-HOST-DUMPDIR
// SYCL-DEVICE-DUMPDIR: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace={{.*}}.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// SYCL-HOST-DUMPDIR: -cc1{{.*}} "-fsycl-is-host"{{.*}} "-ftime-trace=f/a.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// This test verifies that Clang driver correctly propagates time-trace related options
// during a compile-and-link invocation and enables JSON time-trace output.
// RUN: %clang -### -fsycl --offload-new-driver -ftime-trace=e -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -o f/x -dumpdir f/ 2>&1 | FileCheck %s --check-prefixes=LINK-DEVICE,LINK-CLW
// LINK-DEVICE: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=e{{/|\\\\}}a-{{[^.]*}}.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// LINK-CLW: clang-linker-wrapper{{.*}} "--device-compiler=spir64-unknown-unknown=-ftime-trace=e"{{.*}}

