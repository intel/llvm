// RUN: not %{build} -fsyntax-only -o %t.out

#include <sycl/detail/core.hpp>

using namespace sycl;

int main(int argc, char *argv[]) {

  // SYCL 2020 reductions cannot be created from spans with dynamic extents
  auto Span = span<int, dynamic_extent>(nullptr, 1);
  auto Redu = reduction(Span, plus<>());
}
