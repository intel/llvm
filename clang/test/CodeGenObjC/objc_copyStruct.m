// Added -Xclang -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: %clang -Xclang -opaque-pointers -target x86_64-unknown-windows-msvc -fobjc-runtime=ios -Wno-objc-root-class -S -o - -emit-llvm %s | FileCheck %s
// RUN: %clang -Xclang -opaque-pointers -target x86_64-apple-ios -fobjc-runtime=ios -Wno-objc-root-class -S -o - -emit-llvm %s | FileCheck %s

struct S {
  float f, g;
};

@interface I
@property struct S s;
@end

@implementation I
@end

// CHECK: declare {{.*}}void @objc_copyStruct(ptr, ptr, i64, i1, i1)

