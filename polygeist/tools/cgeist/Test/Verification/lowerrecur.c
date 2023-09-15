// RUN: cgeist %s --function=* -emit-llvm -S | FileCheck %s

struct X{
 double* a;
 double* b;
 int c;
};

void perm(struct X* v) {
    v->a = v->b;
}

// CHECK: define void @perm
