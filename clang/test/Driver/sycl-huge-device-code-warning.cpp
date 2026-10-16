// The warning is emitted when the real linker Command actually executes
// (its inputs don't exist yet at job-construction time, so this can't be
// tested with -### like sycl-flink-huge-device-code.cpp). Dummy "object"
// files whose combined size crosses the 1GiB threshold are enough: the
// real `ld` invocation that follows is expected to then fail on them
// (unrecognized file format), which is fine -- the warning is diagnosed
// before that invocation happens.
//
// Output is redirected to a log file rather than piped directly into
// FileCheck: piping loses output from this slow/large-input invocation
// under lit's internal shell.
// UNSUPPORTED: system-windows

// RUN: %python -c "open(r'%t-big.o', 'wb').truncate(1100000000)"
// RUN: %python -c "open(r'%t-small.o', 'wb').truncate(1000)"

// RUN: %clangxx -fsycl %t-big.o %t-small.o -o %t.out > %t-warn.log 2>&1 || true
// RUN: FileCheck --check-prefix=WARN %s < %t-warn.log

// -flink-huge-device-code suppresses the warning.
// RUN: %clangxx -fsycl -flink-huge-device-code %t-big.o %t-small.o -o %t.out > %t-nowarn.log 2>&1 || true
// RUN: FileCheck --check-prefix=NO-WARN --allow-empty %s < %t-nowarn.log

// -Wno-large-device-kernel suppresses the warning too.
// RUN: %clangxx -fsycl -Wno-large-device-kernel %t-big.o %t-small.o -o %t.out > %t-wflag-nowarn.log 2>&1 || true
// RUN: FileCheck --check-prefix=NO-WARN --allow-empty %s < %t-wflag-nowarn.log

// Below the size threshold, no warning either.
// RUN: %clangxx -fsycl %t-small.o -o %t.out > %t-small.log 2>&1 || true
// RUN: FileCheck --check-prefix=NO-WARN --allow-empty %s < %t-small.log

// WARN: warning: total SYCL device code size is {{[0-9.]+}} GB ({{[0-9]+}} bytes); the host link may fail with 'relocation truncated to fit' or 'PC-relative offset overflow' because 32-bit PC-relative relocations cannot reach past 2 GB; add '-flink-huge-device-code' to the link command line (not only the compile line) to place device code sections via a custom linker script

// NO-WARN-NOT: total SYCL device code size
