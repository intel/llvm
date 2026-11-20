// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-work-group-dtor -verify=quiet %s
// RUN: %clang_cc1 -fsycl-is-host -internal-isystem %S/Inputs -verify=quiet %s
// quiet-no-diagnostics
#include "sycl.hpp"

struct Trivial { int I; };
struct NonTrivial { ~NonTrivial() {} };
struct Inherited : NonTrivial { ~Inherited() = default; };
struct Composed { ~Composed() = default; int i; NonTrivial N; };

NonTrivial MakeNonTrivial() { return NonTrivial{}; }

struct NonTrivialRange {
  ~NonTrivialRange() {}
  int *begin() const { return nullptr; }
  int *end() const { return nullptr; }
};

void wg_func(sycl::group<1> G) {
    Trivial T;    // no warning
    NonTrivial N; // expected-warning {{variable 'N' declared at work group scope has a non-trivial destructor}}
    volatile NonTrivial vN; // expected-warning {{variable 'vN' declared at work group scope has a non-trivial destructor}}
    const NonTrivial cN; // expected-warning {{variable 'cN' declared at work group scope has a non-trivial destructor}}
    NonTrivial aN[2]; // expected-warning {{variable 'aN' declared at work group scope has a non-trivial destructor}}
    Inherited I;  // expected-warning {{variable 'I' declared at work group scope has a non-trivial destructor}}
    Composed C;   // expected-warning {{variable 'C' declared at work group scope has a non-trivial destructor}}
    auto [i, sbN] = Composed{}; // expected-warning {{variable '[i, sbN]' declared at work group scope has a non-trivial destructor}}
    
    sycl::private_memory<NonTrivial> PM(G); // no warning: work-item scope
    const NonTrivial& rN = MakeNonTrivial(); // no warning: temporary is a scoped local, destructor runs
    for (int i : NonTrivialRange{})
    (void)i;
      
    G.parallel_for_work_item([&]() {});
}

// Functions that reach parallel_for_work_item indirectly also run at work group
// scope, so their locals are diagnosed too.
void indirect_mid(sycl::group<1> G) {
  NonTrivial MidN; // expected-warning {{variable 'MidN' declared at work group scope has a non-trivial destructor}}
  wg_func(G);
}

void indirect_top(sycl::group<1> G) {
  NonTrivial TopN; // expected-warning {{variable 'TopN' declared at work group scope has a non-trivial destructor}}
  indirect_mid(G);
}

// Does not reach parallel_for_work_item, so it is not at work group scope.
void no_pfwi() {
  NonTrivial N; // no warning
}

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    sycl::kernel_handler KH;
    NonTrivial Implicit; // expected-warning {{captured variable 'Implicit' copied into the kernel object at work group scope has a non-trivial destructor}}
    NonTrivial Explicit;
    CGH.parallel_for_work_group([=, Explicit = Explicit](sycl::group<1> G, sycl::kernel_handler) { // #PFWGCall expected-warning {{captured variable 'Explicit' copied into the kernel object at work group scope has a non-trivial destructor}}
      Trivial T;                 // no warning
      static const NonTrivial S; // no warning: not automatic storage
      NonTrivial N; // expected-warning {{variable 'N' declared at work group scope has a non-trivial destructor}}
                    // expected-note@#PFWGCall {{in instantiation of function template specialization}}
      volatile NonTrivial vN; // expected-warning {{variable 'vN' declared at work group scope has a non-trivial destructor}}
      const NonTrivial cN; // expected-warning {{variable 'cN' declared at work group scope has a non-trivial destructor}}
      NonTrivial aN[2]; // expected-warning {{variable 'aN' declared at work group scope has a non-trivial destructor}}
      Inherited I;  // expected-warning {{variable 'I' declared at work group scope has a non-trivial destructor}}
      Composed C;   // expected-warning {{variable 'C' declared at work group scope has a non-trivial destructor}}
      auto [i, sbN] = Composed{}; // expected-warning {{variable '[i, sbN]' declared at work group scope has a non-trivial destructor}}
      (void)Implicit; // expected-note {{variable 'Implicit' is captured here}}
      
      sycl::private_memory<NonTrivial> PM(G); // no warning: work-item scope
      const NonTrivial& rN = MakeNonTrivial(); // no warning: temporary is a scoped local, destructor runs
      for (int i : NonTrivialRange{})
        (void)i;
      
      wg_func(G);
      indirect_top(G);
      no_pfwi();
    }, KH);
  });
}