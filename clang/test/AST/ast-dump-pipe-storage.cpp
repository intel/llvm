// RUN: %clang %s -fsyntax-only -Xclang -ast-dump -fsycl-device-only | FileCheck %s

typedef __attribute__((pipe("storage"))) int StoragePipe;
// CHECK:      PipeType {{.*}} 'storage pipe int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'

typedef __attribute__((pipe("read_only"))) int ReadPipe;
// CHECK:      PipeType {{.*}} 'read_only pipe int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'

typedef __attribute__((pipe("write_only"))) int WritePipe;
// CHECK:      PipeType {{.*}} 'write_only pipe int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'
