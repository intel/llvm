// Ensure that debug info for typedefs in system headers is still generated
// even if -fno-system-debug is used.  This is justified because debug size 
// savings is small, but debugging is commonly done with types that are
// typedef-ed in system headers.  Thus, the increased debuggability
// is worth the small extra cost.

// Windows does not have <unistd.h>
// UNSUPPORTED: system-windows

// RUN: %clang -fno-system-debug -emit-llvm -S -g %s -o %t.ll

// RUN: FileCheck %s < %t.ll

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "gid_t",
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__gid_t",
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "uid_t",
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__uid_t",
  
#include <unistd.h>

uid_t xuid;
gid_t xgid;

int
main (void)
{
  return 0;
}
