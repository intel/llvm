// RUN: cgeist %s --function=* -emit-llvm -S | FileCheck %s

// TODO: This test case does not yet work with opaque pointers,
// as it requires the Polygeist transformations to work correctly
// with opaque pointers.

struct X{
 double* a;
 double* b;
 int c;
};

void perm(struct X* v) {
    v->a = v->b;
}

// CHECK: define void @perm
