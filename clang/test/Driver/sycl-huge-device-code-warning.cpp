// Checks that linking with huge device code input warns about possible
// 32-bit relocation overflow, and that the warning can be suppressed with
// -flink-huge-device-code or -Wno-large-device-kernel.
//
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

// RUN: rm -f %t-big.o

// WARN: warning: total link input size of {{[0-9.]+}} GB ({{[0-9]+}} bytes) may overflow 32-bit relocations in the host link; consider linking with '-flink-huge-device-code'

// NO-WARN-NOT: total link input size
