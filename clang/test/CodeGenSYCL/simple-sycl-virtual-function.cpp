// Test verifying that RTTI information is not emitted during SYCL device compilation.

// RUN: %clang_cc1 -triple spir64 -fsycl-allow-virtual-functions -fsycl-is-device -emit-llvm %s -o - | FileCheck %s --implicit-check-not _ZTI4Base --implicit-check-not _ZTI8Derived1 -check-prefix VTABLE
// RUN: %clang_cc1 -triple spir64 -fsycl-allow-virtual-functions -fsycl-is-device -fexperimental-relative-c++-abi-vtables -emit-llvm %s -o - | FileCheck %s --implicit-check-not _ZTI4Base --implicit-check-not _ZTI8Derived1

// VTABLE: @_ZTV8Derived1 = linkonce_odr unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN8Derived17displayEv] }, comdat, align 8
// VTABLE: @_ZTV4Base = linkonce_odr unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN4Base7displayEv] }, comdat, align 8

SYCL_EXTERNAL bool rand();

class Base {
   public:
    virtual void display() {}
};

class Derived1 : public Base {
   public:
    void display() {}
};

SYCL_EXTERNAL void test() {
  Derived1 d1;
  Base *b = nullptr;
  if (rand())
    b = &d1;
  b->display();
}
