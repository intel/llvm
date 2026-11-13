// RUN: %{build} -Wno-error=deprecated-declarations -Wno-sycl-work-group-dtor -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -Wno-error=deprecated-declarations -Wno-sycl-work-group-dtor %O0 -o %t_O0.out
// RUN: %{run} %t_O0.out

// Checks that destructors of work group scope variables are called once per
// work group at the end of their scope.

#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/h_item.hpp>
#include <sycl/usm.hpp>

#include <algorithm>
#include <iostream>

using namespace sycl;

constexpr size_t NumGroups = 4;
constexpr size_t GroupSize = 16;
constexpr int NumIters = 3;

// The increment is atomic so that a destructor wrongly called by every work
// item is counted instead of being lost in a data race.
struct Counter {
  int *Ptr;
  ~Counter() {
#ifdef __SYCL_DEVICE_ONLY__
    atomic_ref<int, memory_order::relaxed, memory_scope::device> Ref(*Ptr);
    Ref.fetch_add(1);
#endif
  }
};

static int check(const char *What, const int *Values, size_t Count,
                 int Expected) {
  int Errors = 0;
  for (size_t I = 0; I < Count; ++I) {
    if (Values[I] != Expected) {
      std::cout << What << "[" << I << "] = " << Values[I] << ", expected "
                << Expected << "\n";
      ++Errors;
    }
  }
  return Errors;
}

int main() {
  queue Q;

  int *ScopeCount = malloc_shared<int>(NumGroups, Q);
  int *ScopeCountAtExit = malloc_shared<int>(NumGroups, Q);
  int *LoopCount = malloc_shared<int>(NumGroups, Q);
  std::fill(ScopeCount, ScopeCount + NumGroups, 0);
  std::fill(ScopeCountAtExit, ScopeCountAtExit + NumGroups, 0);
  std::fill(LoopCount, LoopCount + NumGroups, 0);

  Q.submit([&](handler &CGH) {
     CGH.parallel_for_work_group(
         range<1>{NumGroups}, range<1>{GroupSize}, [=](group<1> G) {
           size_t Id = G.get_group_id(0);
           {
             Counter InScope{&ScopeCount[Id]};
             G.parallel_for_work_item([&](h_item<1>) { (void)InScope; });
           }
           ScopeCountAtExit[Id] = ScopeCount[Id];
           for (int I = 0; I < NumIters; ++I) {
             Counter InLoop{&LoopCount[Id]};
             G.parallel_for_work_item([&](h_item<1>) { (void)InLoop; });
           }
           // FIXME: Captures with a non-trivial destructor are not tested:
           // the kernel function is not lowered by LowerWGScope, so the
           // destructor of the kernel object is called by every work item.
           // See https://github.com/intel/llvm/issues/23700
         });
   }).wait();

  int Errors = 0;
  Errors += check("ScopeCount", ScopeCount, NumGroups, 1);
  Errors += check("ScopeCountAtExit", ScopeCountAtExit, NumGroups, 1);
  Errors += check("LoopCount", LoopCount, NumGroups, NumIters);

  free(ScopeCount, Q);
  free(ScopeCountAtExit, Q);
  free(LoopCount, Q);
  return Errors;
}
