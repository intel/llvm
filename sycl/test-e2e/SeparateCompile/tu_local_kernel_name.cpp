// Two TUs launching kernels whose names are only guaranteed to be unique within
// their own TU must not cause either TU's kernels to be silently dropped.

// DEFINE: %{compile} = %clangxx -fsycl %{sycl_target_opts} -c %s
// DEFINE: %{link} = %clangxx -fsycl %{sycl_target_opts} -Wno-unused-command-line-argument
//
// RUN: %{compile} -DTU=0 -o %t.main.o
// RUN: %{compile} -DTU=1 -o %t.tu1.o
// RUN: %{compile} -DTU=2 -o %t.tu2.o
// RUN: %{link} -o %t.out %t.main.o %t.tu1.o %t.tu2.o
// RUN: %{run} %t.out
//
// Without RDC, each TU's device code becomes its own device image, so a
// collision would surface in the runtime's kernel name lookup instead.
// RUN: %{link} -fno-sycl-rdc -o %t.no_rdc.out %t.main.o %t.tu1.o %t.tu2.o
// RUN: %{run} %t.no_rdc.out
//
// Range rounding results in range-rounded kernels launching from a wrapper
// kernel whose name is derived from the TU-local kernel lambda. This should
// not result in name collisions if multiple TUs do the same thing.
// RUN: %{compile} -fsycl-range-rounding=force -DTU=0 -o %t.rr.main.o
// RUN: %{compile} -fsycl-range-rounding=force -DTU=1 -o %t.rr.tu1.o
// RUN: %{compile} -fsycl-range-rounding=force -DTU=2 -o %t.rr.tu2.o
// RUN: %{link} -o %t.rr.out %t.rr.main.o %t.rr.tu1.o %t.rr.tu2.o
// RUN: %{run} %t.rr.out
//
// Each TU in its own shared library, with both loaded into one process. The
// libraries are never device-linked together, but the runtime looks kernels up
// by name across all loaded images, so the names must still differ.
// RUN: %if linux %{ rm -rf %t.dir; mkdir -p %t.dir %}
// RUN: %if linux %{ %{build} -DTU=1 %fPIC %shared_lib -o %t.dir/lib%basename_t-1.so %}
// RUN: %if linux %{ %{build} -DTU=2 %fPIC %shared_lib -o %t.dir/lib%basename_t-2.so %}
// RUN: %if linux %{ %{run-aux} %{link} -o %t.so.out %t.main.o -L%t.dir -l%basename_t-1 -l%basename_t-2 -Wl,-rpath=%t.dir %}
// RUN: %if linux %{ %{run} %t.so.out %}

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <algorithm>
#include <iostream>

constexpr int NumTUs = 2;
constexpr int NumTests = 4;

void runTU1(sycl::queue &q, int *Res);
void runTU2(sycl::queue &q, int *Res);

// Large and not a multiple of common work-group sizes in order to trigger
// range rounding. Range-rounded kernels launch from detail::RoundedRangeKernel
// wrapper, which should still be impervious to TU-local name collisions.
constexpr size_t N = 1031;

// This kernel's name is TU-local because it gets template instantiated with
// TU-local types.
template <typename T> void rangeRounding(sycl::queue &q, int *Out) {
  q.parallel_for(sycl::range<1>(N), [Out](sycl::id<1> idx) {
     if (idx == 0)
       *Out = T::Value;
   }).wait();
}

#if TU != 0

static void internalLinkage(sycl::queue &q, int *Out) {
  q.single_task([Out] { *Out = TU; }).wait();
}

namespace {
class KAnon;
struct Anonymous {
  static constexpr int Value = TU;
};
struct AnonFunctor {
  int *Out;
  void operator()() const { *Out = TU; }
};
} // namespace

#if TU == 1
void runTU1(sycl::queue &q, int *Res) {
#elif TU == 2
void runTU2(sycl::queue &q, int *Res) {
#endif
  // Each kernel writes its own TU as a constant and captures nothing else, so a
  // kernel that replaced the other TU's one writes the wrong value.
  q.single_task<KAnon>([Res] { Res[0] = TU; }).wait();
  q.single_task(AnonFunctor{Res + 1}).wait();
  internalLinkage(q, Res + 2);
  rangeRounding<Anonymous>(q, Res + 3);
}

#else // TU == 0

int main() {
  sycl::queue q;
  int *Res = sycl::malloc_shared<int>(NumTUs * NumTests, q);
  std::fill(Res, Res + NumTUs * NumTests, 0);
  runTU1(q, Res);
  runTU2(q, Res + NumTests);

  int Failures = 0;
  for (int T = 1; T <= NumTUs; T++) {
    for (int Test = 0; Test < NumTests; Test++) {
      int Got = Res[(T - 1) * NumTests + Test];
      if (Got != T) {
        std::cerr << "TU " << T << ", test " << Test << ": got " << Got << "\n";
        Failures++;
      }
    }
  }
  sycl::free(Res, q);
  return Failures;
}

#endif
