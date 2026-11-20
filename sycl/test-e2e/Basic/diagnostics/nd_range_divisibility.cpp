// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that the SYCL runtime throws errc::nd_range when the global work size
// is not evenly divisible by the local work-group size, and that the error
// message names the actual dimensions used.

#include <cassert>
#include <sycl/detail/core.hpp>

#include <string>

using namespace sycl;

static auto makeQueue() {
  return queue{[](exception_list EL) {
    for (auto E : EL)
      std::rethrow_exception(E);
  }};
}

template <int Dims>
static void checkThrows(nd_range<Dims> NDR, std::string_view GlobalStr,
                        std::string_view LocalStr) {
  queue Q = makeQueue();
  bool caught = false;
  try {
    Q.submit([&](handler &H) {
      H.parallel_for(NDR, [=](nd_item<Dims>) {});
    });
    Q.wait_and_throw();
  } catch (const sycl::exception &E) {
    if (E.code() != errc::nd_range)
      throw;
    std::string msg = E.what();
    assert(msg.find(GlobalStr) != std::string::npos &&
           "Global range not in error message");
    assert(msg.find(LocalStr) != std::string::npos &&
           "Local range not in error message");
    caught = true;
  }
  assert(caught && "Expected errc::nd_range was not thrown");
}

template <int Dims>
static void checkNoThrow(nd_range<Dims> NDR) {
  queue Q = makeQueue();
  Q.submit([&](handler &H) {
    H.parallel_for(NDR, [=](nd_item<Dims>) {});
  });
  Q.wait_and_throw();
}

int main() {
  // --- error cases ---
  // 1D: global=1, local=2
  checkThrows(nd_range<1>{{1}, {2}}, "{1}", "{2}");

  // 1D: global=3, local=2
  checkThrows(nd_range<1>{{3}, {2}}, "{3}", "{2}");

  // 2D: second dimension not divisible
  checkThrows(nd_range<2>{{4, 3}, {2, 2}}, "{4, 3}", "{2, 2}");

  // 3D: third dimension not divisible
  checkThrows(nd_range<3>{{4, 4, 5}, {2, 2, 2}}, "{4, 4, 5}", "{2, 2, 2}");

  // --- valid cases (must not throw) ---
  // 1D evenly divisible
  checkNoThrow(nd_range<1>{{4}, {2}});

  // 2D both dimensions evenly divisible
  checkNoThrow(nd_range<2>{{4, 6}, {2, 3}});

  return 0;
}
