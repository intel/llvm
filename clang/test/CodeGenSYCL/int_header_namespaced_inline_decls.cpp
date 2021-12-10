// RUN: %clang_cc1 -fsycl-is-device -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// This test validates the behavior of inline-kernel-names to try to put them in
// the 'closest' possible namespace.

#include "Inputs/sycl.hpp"

using namespace sycl;

// Forward declarations of templated kernel function types:

namespace TopLevel {
void use() {
  kernel_single_task<class DirectTopLevel>([]() {});
  // CHECK: namespace TopLevel {
  // CHECK-NEXT: class DirectTopLevel;
  // CHECK-NEXT: }
}

struct TypeName {
  void member_func() {
    kernel_single_task<class DirectTopLevelMemFunc>([]() {});
    // CHECK: namespace TopLevel {
    // CHECK-NEXT: class DirectTopLevelMemFunc;
    // CHECK-NEXT: }
  }
};

extern "C" {
void use1() {
  kernel_single_task<class DirectTopLevelLinkage>([]() {});
  // CHECK: namespace TopLevel {
  // CHECK-NEXT: class DirectTopLevelLinkage;
  // CHECK-NEXT: }
}
struct LinkageTypeName {
  void member_func() {
    kernel_single_task<class DirectTopLevelLinkageMemFunc>([]() {});
  // CHECK: namespace TopLevel {
  // CHECK-NEXT: class DirectTopLevelLinkageMemFunc;
  // CHECK-NEXT: }
  }
};
}
} // namespace TopLevel

namespace {
void use2() {
  kernel_single_task<class TopLevelAnonNS>([]() {});
  // CHECK: namespace  {
  // CHECK-NEXT: class TopLevelAnonNS;
  // CHECK-NEXT: }
}

struct LinkageTypeName {
  void member_func() {
    kernel_single_task<class AnonNSMemFunc>([]() {});
  // CHECK: namespace  {
  // CHECK-NEXT: class AnonNSMemFunc;
  // CHECK-NEXT: }
  }
};
} // namespace

inline namespace InlineTopLevel {
void use3() {
  kernel_single_task<class InlineDirectTopLevel>([]() {});
  // CHECK: inline namespace InlineTopLevel {
  // CHECK-NEXT: class InlineDirectTopLevel;
  // CHECK-NEXT: }
}
struct LinkageTypeName {
  void member_func() {
    kernel_single_task<class InlineNSMemFunc>([]() {});
  // CHECK: inline namespace InlineTopLevel {
  // CHECK-NEXT: class InlineNSMemFunc;
  // CHECK-NEXT: }
  }
};

inline namespace {
void use4() {
  kernel_single_task<class AnonNS>([]() {});
  // CHECK: inline namespace  {
  // CHECK-NEXT: class AnonNS;
  // CHECK-NEXT: }
}

extern "C" {
void use5() {
  kernel_single_task<class AnonNSLinkage>([]() {});
  // CHECK: inline namespace  {
  // CHECK-NEXT: class AnonNSLinkage;
  // CHECK-NEXT: }
}
}
struct LinkageTypeName {
  void member_func() {
    kernel_single_task<class InlineAnonNSMemFunc>([]() {});
    // CHECK: inline namespace  {
    // CHECK-NEXT: class InlineAnonNSMemFunc;
    // CHECK-NEXT: }
  }
};
} // namespace
} // namespace TopLevel

namespace A {
namespace B {
namespace {
namespace C::D {
struct DeepStruct {
  void member_func() {
    kernel_single_task<class WoahDeep>([]() {});
    // CHECK: namespace A { namespace B { namespace { namespace C { namespace D {
    // CHECK-NEXT: class WoahDeep;
    // CHECK-NEXT: }}}}}
  }
};
} // namespace C::D
} // namespace
} // namespace B
} // namespace A
