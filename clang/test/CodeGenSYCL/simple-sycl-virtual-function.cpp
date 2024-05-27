// Test verifies the following
//  1. RTTI information is not emitted during SYCL device compilation.
//  2. Virtual table elements are generated in AS4.
//  3. Runtime Global Variables are generated in AS1.

// RUN: %clang_cc1 -triple spir64 -fsycl-allow-virtual-functions -fsycl-is-device -emit-llvm %s -o - | FileCheck %s --implicit-check-not _ZTI4Base --implicit-check-not _ZTI8Derived1 -check-prefix VTABLE
// RUNx: %clang_cc1 -triple spir64 -fsycl-allow-virtual-functions -fsycl-is-device -fexperimental-relative-c++-abi-vtables -emit-llvm %s -o - | FileCheck %s --implicit-check-not _ZTI4Base --implicit-check-not _ZTI8Derived1

// Since experimental-relative-c++-abi-vtables is some experimental option, temporary disabling the check for now
// until we emit proper address spaces (and casts) everywhere.

// VTABLE: @_ZTV8Derived1 = linkonce_odr unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @_ZN8Derived17displayEv to ptr addrspace(4))] }, comdat, align 8
// VTABLE: @_ZTV4Base = linkonce_odr unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @_ZN4Base7displayEv to ptr addrspace(4))] }, comdat, align 8

SYCL_EXTERNAL bool rand();

class Base {
   public:
    [[intel::device_indirectly_callable]] virtual void display() {}
};

class Derived1 : public Base {
   public:
    [[intel::device_indirectly_callable]] void display() override {}
};

SYCL_EXTERNAL void test() {
  Derived1 d1;
  Base *b = nullptr;
  if (rand())
    b = &d1;
  b->display();
}
