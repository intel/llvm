// INTEL RUN: %clang -Xclang -opaque-pointers -w -target x86_64-linux-gnu -S -emit-llvm -o - -fno-exceptions -O0 \
// INTEL RUN:     -Xclang -disable-llvm-passes %s | FileCheck %s -check-prefixes=CHECK \
// INTEL RUN:      --implicit-check-not=llvm.lifetime
// INTEL RUN: %clang -Xclang -opaque-pointers -w -target x86_64-linux-gnu -S -emit-llvm -o - -fno-exceptions -O0 \
// INTEL RUN:     -fsanitize=address -fsanitize-address-use-after-scope \
// INTEL RUN:     -Xclang -disable-llvm-passes %s | FileCheck %s -check-prefixes=CHECK,LIFETIME
// INTEL RUN: %clang -Xclang -opaque-pointers -w -target x86_64-linux-gnu -S -emit-llvm -o - -fno-exceptions -O0 \
// INTEL RUN:     -fsanitize=memory -Xclang -disable-llvm-passes %s | \
// INTEL RUN:     FileCheck %s -check-prefixes=CHECK,LIFETIME
// INTEL RUN: %clang -Xclang -opaque-pointers -w -target aarch64-linux-gnu -S -emit-llvm -o - -fno-exceptions -O0 \
// INTEL RUN:     -fsanitize=hwaddress -Xclang -disable-llvm-passes %s | \
// INTEL RUN:     FileCheck %s -check-prefixes=CHECK,LIFETIME

extern int bar(char *A, int n);

struct X {
  X();
  ~X();
  int *p;
};
struct Y {
  Y();
  int *p;
};

extern "C" void a(), b(), c(), d();

// CHECK: define dso_local void @_Z3fooi(i32 noundef %[[N:[^)]+]])
void foo(int n) {
  // CHECK: store i32 %[[N]], ptr %[[NADDR:[^,]+]]
  // CHECK-LABEL: call void @a()
  a();

  // CHECK-LABEL: call void @b()
  // CHECK: [[NARG:%[^ ]+]] = load i32, ptr %[[NADDR]]
  // CHECK: [[BOOL:%[^ ]+]] = icmp ne i32 [[NARG]], 0
  // CHECK: store i1 false
  // CHECK: br i1 [[BOOL]], label %[[ONTRUE:[^,]+]], label %[[ONFALSE:[^,]+]]
  //
  // CHECK: [[ONTRUE]]:
  // LIFETIME: @llvm.lifetime.start
  // LIFETIME: store i1 true
  // LIFETIME: call void @_ZN1XC
  // CHECK: br label %[[END:[^,]+]]
  //
  // CHECK: [[ONFALSE]]:
  // LIFETIME: @llvm.lifetime.start
  // LIFETIME: store i1 true
  // LIFETIME: call void @_ZN1YC
  // CHECK: br label %[[END]]
  //
  // CHECK: [[END]]:
  // CHECK: call void @c()
  // LIFETIME: @llvm.lifetime.end
  // LIFETIME: @llvm.lifetime.end
  b(), (n ? X().p : Y().p), c();

  // CHECK: call void @d()
  d();
}
