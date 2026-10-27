// Regression test: a huge link input that is NOT wrapped SYCL device code
// (e.g. a plain host object or static archive) must not trigger the
// large-device-kernel warning, with or without the flags that suppress it.
// The positive case (an actual oversized wrapped device-code object
// triggering the warning, and that -flink-huge-device-code /
// -Wno-large-device-kernel suppress it) is covered by
// ToolChainTest.HugeSYCLDeviceCodeWarning and its suppression variants,
// since reproducing a >1GiB real device binary here would require an
// impractically large/slow compile.
//
// Dummy "object" files whose combined size crosses the 1GiB threshold are
// enough. -fuse-ld=/bin/true stands in for the real linker so the overall
// command succeeds even though these aren't real object files -- we only
// care that no warning is emitted.
//
// Output is redirected to a log file rather than piped directly into
// FileCheck: piping loses output from this slow/large-input invocation
// under lit's internal shell.
// UNSUPPORTED: system-windows

// RUN: %python -c "open(r'%t-big.o', 'wb').truncate(1100000000)"
// RUN: %python -c "open(r'%t-small.o', 'wb').truncate(1000)"

// RUN: %clangxx -fsycl -fuse-ld=/bin/true %t-big.o %t-small.o -o %t.out > %t-warn.log 2>&1
// RUN: FileCheck --check-prefix=NO-WARN --allow-empty %s < %t-warn.log

// RUN: %clangxx -fsycl -fuse-ld=/bin/true -flink-huge-device-code %t-big.o %t-small.o -o %t.out > %t-flink.log 2>&1
// RUN: FileCheck --check-prefix=NO-WARN --allow-empty %s < %t-flink.log

// RUN: %clangxx -fsycl -fuse-ld=/bin/true -Wno-large-device-kernel %t-big.o %t-small.o -o %t.out > %t-wno.log 2>&1
// RUN: FileCheck --check-prefix=NO-WARN --allow-empty %s < %t-wno.log

// RUN: rm -f %t-big.o

// NO-WARN-NOT: total device code size
