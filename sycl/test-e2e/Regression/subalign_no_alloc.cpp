// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that a type with a different alignment from its size does not cause
// the runtime to reallocate memory.

#include <sycl/detail/core.hpp>

#include <vector>

using namespace sycl;

constexpr size_t N = 100;

struct alignas(long) SubalignedStruct {
  long x, y, z;
};

struct PanicAllocator {
  using value_type = SubalignedStruct;
  using size_type = std::size_t;

  PanicAllocator() = default;
  PanicAllocator(const PanicAllocator &) {}

  SubalignedStruct *allocate(size_type) {
    assert(false && "Allocation should not have happened! Panic!");
    return nullptr;
  }
  void deallocate(SubalignedStruct *, size_type) {}
  bool operator==(const PanicAllocator &) const { return true; }
  bool operator!=(const PanicAllocator &) const { return false; }
};

int main() {
  queue Q;

  std::vector<SubalignedStruct> Data{N};
  buffer<SubalignedStruct, 1, PanicAllocator> Buf{Data.data(), N};

  Q.submit([&](handler &CGH) {
    accessor Acc{Buf, CGH, write_only};
    CGH.parallel_for(N, [=](item<1> I) {
      Acc[I].x = I.get_linear_id() + 1;
      Acc[I].y = I.get_linear_id() + 2;
      Acc[I].z = I.get_linear_id() + 3;
    });
  });

  return 0;
}
