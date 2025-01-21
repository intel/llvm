// REQUIRES: system-linux
// Test that the return value from the called tool is retained.
// Runs a script within a script so we can retain the return code without
// the testing infrastructure getting in the way.

// RUN: echo 'Content of first file' > %t1.tgt
// RUN: echo 'Content of second file' > %t2.tgt
// RUN: echo "%t1.tgt" > %t.list
// RUN: echo "%t2.tgt" >> %t.list

// RUN: echo "#!/bin/sh" > %t.sh
// RUN: echo "cat \$1" >> %t.sh
// RUN: echo "exit 21" >> %t.sh
// RUN: chmod 777 %t.sh
// RUN: echo "#!/bin/sh" > %t2.sh
// RUN: echo "llvm-foreach --in-replace=\"{}\" --in-file-list=%t.list -- %t.sh \"{}\" > %t.res" >> %t2.sh
// RUN: echo "echo \$? >> %t.res" >> %t2.sh
// RUN: chmod 777 %t2.sh
// RUN: %t2.sh
// RUN: FileCheck < %t.res %s
// CHECK: Content of first file
// CHECK: Content of second file
// CHECK: 21
