// Enable time tracing capability in SYCL applications.

// This test verifies that Clang driver correctly propagates time-trace related options
// during a compile-only invocation and enables JSON time-trace output.

// REQUIRES: system-linux

// RUN: mkdir d e f && cp %s d/a.cpp && touch d/b.c
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -o e/a.o 2>&1 | FileCheck %s --check-prefixes=SYCL-DEVICE-COMPILE,SYCL-HOST-COMPILE
// SYCL-DEVICE-COMPILE: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=e{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// SYCL-HOST-COMPILE: -cc1{{.*}} "-fsycl-is-host"{{.*}} "-ftime-trace=e{{/|\\\\}}a-host-x86_64-unknown-linux-gnu.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// Verify that the Clang driver generates JSON time-trace output for compile-only
// invocation and propagates the time-trace options, respecting the specified dump directory.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -dumpdir f/ 2>&1 | FileCheck %s --check-prefixes=SYCL-DEVICE-DUMPDIR,SYCL-HOST-DUMPDIR
// SYCL-DEVICE-DUMPDIR: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=f{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// SYCL-HOST-DUMPDIR: -cc1{{.*}} "-fsycl-is-host"{{.*}} "-ftime-trace=f{{/|\\\\}}a-host-x86_64-unknown-linux-gnu.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// This test verifies that Clang driver correctly propagates time-trace related options
// during a compile-and-link invocation and enables JSON time-trace output.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -ftime-trace=e -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -o f/x -dumpdir f/ 2>&1 | FileCheck %s --check-prefixes=LINK-DEVICE,LINK-HOST,LINK-CLW
// LINK-DEVICE: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=e{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// LINK-HOST: -cc1{{.*}} "-fsycl-is-host"{{.*}} "-ftime-trace=e{{/|\\\\}}a-host-x86_64-unknown-linux-gnu.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// LINK-CLW: clang-linker-wrapper{{.*}} "--device-compiler=spir64-unknown-unknown=-ftime-trace=e"{{.*}}

// Verify time tracing works for SYCL offload in the old model
// This test verifies that Clang driver correctly propagates time-trace related options
// during a compile-only invocation and enables JSON time-trace output.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -o e/a.o 2>&1 | FileCheck %s --check-prefixes=SYCL-DEVICE-COMPILE-OLD-MODEL,SYCL-HOST-COMPILE-OLD-MODEL
// SYCL-DEVICE-COMPILE-OLD-MODEL: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=e{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// SYCL-HOST-COMPILE-OLD-MODEL: -cc1{{.*}} "-fsycl-is-host"{{.*}} "-ftime-trace=e{{/|\\\\}}a-host-x86_64-unknown-linux-gnu.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// Verify that the Clang driver generates JSON time-trace output for compile-only
// invocation and propagates the time-trace options, respecting the specified dump directory.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -dumpdir f/ 2>&1 | FileCheck %s --check-prefixes=SYCL-DEVICE-DUMPDIR-OLD-MODEL,SYCL-HOST-DUMPDIR-OLD-MODEL
// SYCL-DEVICE-DUMPDIR-OLD-MODEL: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=f{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// SYCL-HOST-DUMPDIR-OLD-MODEL: -cc1{{.*}} "-fsycl-is-host"{{.*}} "-ftime-trace=f{{/|\\\\}}a-host-x86_64-unknown-linux-gnu.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// This test verifies that Clang driver correctly propagates time-trace related options
// during a compile-and-link invocation and enables JSON time-trace output.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -ftime-trace=e -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -o f/x -dumpdir f/ 2>&1 | FileCheck %s --check-prefixes=LINK-DEVICE-OLD-MODEL,LINK-HOST-OLD-MODEL
// LINK-DEVICE-OLD-MODEL: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=e{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// LINK-HOST-OLD-MODEL: -cc1{{.*}} "-fsycl-is-host"{{.*}} "-ftime-trace=e{{/|\\\\}}a-host-x86_64-unknown-linux-gnu.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"

// Verify time tracing works with -fsycl-device-only (new driver model)
// This test verifies device-only compilation generates the correct time-trace file
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-device-only -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -o e/a.o 2>&1 | FileCheck %s --check-prefixes=DEVICE-ONLY-COMPILE
// DEVICE-ONLY-COMPILE: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=e{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// DEVICE-ONLY-COMPILE-NOT: "-fsycl-is-host"

// Verify device-only compilation with -dumpdir (new driver model)
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-device-only -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -dumpdir f/ 2>&1 | FileCheck %s --check-prefixes=DEVICE-ONLY-DUMPDIR
// DEVICE-ONLY-DUMPDIR: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=f{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// DEVICE-ONLY-DUMPDIR-NOT: "-fsycl-is-host"

// Verify time tracing works with -fsycl-device-only (old driver model)
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-device-only -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -o e/a.o 2>&1 | FileCheck %s --check-prefixes=DEVICE-ONLY-COMPILE-OLD
// DEVICE-ONLY-COMPILE-OLD: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=e{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// DEVICE-ONLY-COMPILE-OLD-NOT: "-fsycl-is-host"

// Verify device-only compilation with -dumpdir (old driver model)
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-device-only -c -ftime-trace -ftime-trace-granularity=0 -ftime-trace-verbose d/a.cpp -dumpdir f/ 2>&1 | FileCheck %s --check-prefixes=DEVICE-ONLY-DUMPDIR-OLD
// DEVICE-ONLY-DUMPDIR-OLD: -cc1{{.*}} "-fsycl-is-device"{{.*}} "-ftime-trace=f{{/|\\\\}}a-sycl-spir64-unknown-unknown.json" "-ftime-trace-granularity=0" "-ftime-trace-verbose"
// DEVICE-ONLY-DUMPDIR-OLD-NOT: "-fsycl-is-host"

