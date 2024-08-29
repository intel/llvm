// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <cstring>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// This test performs a swap of two scalars/arrays inside a kernel using a
// work_group_memory object as a temporary buffer. The test is done for scalar
// types and bounded arrays. After the kernel finishes, it is verified on the
// host side that the swap worked.

// One important note is that for unbounded arrays, the feature is unstable.
// Specifically, the code may or may not compile when kernels reference work
// group memory objects that have been constructed with the unbounded array
// type. This is due to a limitation of SPIRV where it does not allow arrays of
// length zero. For example, an unbounded array may be translated to an array of
// length zero in LLVM IR and during the LLVM IR -> SPIRV translation phase, the
// translator rejects all arrays of length zero because they are invalid
// constructs in SPIRV. As a result of this, unbounded arrays do not appear in
// this test. They do appear in the sanity test though in this directory because
// there the unbounded arrays are used with concrete subscript indices which
// seems to work, for now at least.

template <typename T> void swap_scalar(T &a, T &b) {
  sycl::queue q;
  const T old_a = a;
  const T old_b = b;
  {
    sycl::buffer<T, 1> buf_a{&a, 1};
    sycl::buffer<T, 1> buf_b{&b, 1};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T> temp{cgh};
      cgh.single_task([=]() {
        temp = acc_a[0];
        acc_a[0] = acc_b[0];
        acc_b[0] = temp;
      });
    });
  }
  assert(a == old_b && b == old_a && "Incorrect swap!");

  // swap again but this time using two temporaries. The first temporary will be
  // used to save the value of a and the second temporay will be
  // default-constructed and then copy-assigned from the first temporary to be
  // then used to write that value to b.
  {
    sycl::buffer<T, 1> buf_a{&a, 1};
    sycl::buffer<T, 1> buf_b{&b, 1};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T> temp{cgh};
      syclexp::work_group_memory<T> temp2;
      cgh.single_task([=]() {
        temp2 = temp; // temp and temp2 have the same underlying data
        temp = acc_a[0];
        acc_a[0] = acc_b[0];
        acc_b[0] = temp2; // safe to use temp2
      });
    });
  }
  // Two swaps same as no swaps
  assert(a == old_a && b == old_b && "Incorrect swap!");

  // Initialize a second temporary and instead of assigning the first temporary
  // to it, assign only the value of the data of the first temporary so that
  // unlike above, the two temporaries will not be aliasing the same memory
  // location but they will have equal values.
  {
    sycl::buffer<T, 1> buf_a{&a, 1};
    sycl::buffer<T, 1> buf_b{&b, 1};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T> temp{cgh};
      syclexp::work_group_memory<T> temp2{cgh};
      cgh.single_task([=]() {
        temp = acc_a[0];
        acc_a[0] = acc_b[0];
        temp2 = *(temp.get_multi_ptr()); // temp2 now has the same value as temp
                                         // but not the same memory location
        acc_b[0] = temp2;
      });
    });
  }
  // Three swaps same as one swap
  assert(a == old_b && b == old_a && "Incorrect swap!");

  // Same as above but instead of using multi_ptr, use address-of operator.
  {
    sycl::buffer<T, 1> buf_a{&a, 1};
    sycl::buffer<T, 1> buf_b{&b, 1};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T> temp{cgh};
      syclexp::work_group_memory<T> temp2{cgh};
      cgh.single_task([=]() {
        temp = acc_a[0];
        acc_a[0] = acc_b[0];
        temp2 = *(&temp);
        acc_b[0] = temp2;
      });
    });
  }
  // Four swaps same as no swap
  assert(a == old_a && b == old_b && "Incorrect swap!");
}

template <typename T, size_t N> void swap_array_1d(T (&a)[N], T (&b)[N]) {
  sycl::queue q;
  T old_a[N];
  std::memcpy(old_a, a, sizeof(a));
  T old_b[N];
  std::memcpy(old_b, b, sizeof(b));
  {
    sycl::buffer<T, 1> buf_a{a, N};
    sycl::buffer<T, 1> buf_b{b, N};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N]> temp{cgh};
      cgh.single_task([=]() {
        for (int i = 0; i < N; ++i) {
          temp[i] = acc_a[i];
          acc_a[i] = acc_b[i];
          acc_b[i] = temp[i];
        }
      });
    });
  }
  for (int i = 0; i < N; ++i) {
    assert(a[i] == old_b[i] && b[i] == old_a[i] && "Incorrect swap!");
  }

  // Instead of working with the temporary work group memory object, we retrieve
  // its corresponding multi-pointer and work with it instead.
  {
    sycl::buffer<T, 1> buf_a{a, N};
    sycl::buffer<T, 1> buf_b{b, N};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N]> temp{cgh};
      cgh.single_task([=]() {
        auto ptr = temp.get_multi_ptr();
        for (int i = 0; i < N; ++i) {
          ptr[i] = acc_a[i];
          acc_a[i] = acc_b[i];
          acc_b[i] = ptr[i];
        }
      });
    });
  }
  // Two swaps same as no swap
  for (int i = 0; i < N; ++i) {
    assert(a[i] == old_a[i] && b[i] == old_b[i] && "Incorrect swap!");
  }

  // Same as above but use a pointer returned by the address-of operator
  // instead.
  {
    sycl::buffer<T, 1> buf_a{a, N};
    sycl::buffer<T, 1> buf_b{b, N};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N]> temp{cgh};
      cgh.single_task([=]() {
        auto ptr = &temp;
        for (int i = 0; i < N; ++i) {
          (*ptr)[i] = acc_a[i];
          acc_a[i] = acc_b[i];
          acc_b[i] = (*ptr)[i];
        }
      });
    });
  }
  // Three swaps same as one swap
  for (int i = 0; i < N; ++i) {
    assert(a[i] == old_b[i] && b[i] == old_a[i] && "Incorrect swap!");
  }
}

template <typename T, size_t N, size_t M>
void swap_array_2d(T (&a)[N][M], T (&b)[N][M]) {
  sycl::queue q;
  T old_a[N][M];
  for (int i = 0; i < N; ++i) {
    std::memcpy(old_a[i], a[i], sizeof(a[0]));
  }
  T old_b[N][M];
  for (int i = 0; i < N; ++i) {

    std::memcpy(old_b[i], b[i], sizeof(b[0]));
  }
  {
    sycl::buffer<T, 2> buf_a{a[0], sycl::range{N, M}};
    sycl::buffer<T, 2> buf_b{b[0], sycl::range{N, M}};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N][M]> temp{cgh};
      cgh.single_task([=]() {
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < M; ++j) {
            temp[i][j] = acc_a[i][j];
            acc_a[i][j] = acc_b[i][j];
            acc_b[i][j] = temp[i][j];
          }
        }
      });
    });
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      assert(a[i][j] == old_b[i][j] && b[i][j] == old_a[i][j] &&
             "Incorrect swap!");
    }
  }

  // Perform the swap but this time use two temporary work group memory objects.
  // One will save the value of acc_a and the other will be copy-assigned from
  // it and will be used to write the values back to acc_b.
  {
    sycl::buffer<T, 2> buf_a{a[0], sycl::range{N, M}};
    sycl::buffer<T, 2> buf_b{b[0], sycl::range{N, M}};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N][M]> temp{cgh};
      syclexp::work_group_memory<T[N][M]> temp2{cgh};
      cgh.single_task([=]() {
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < M; ++j) {
            temp[i][j] = acc_a[i][j];
            acc_a[i][j] = acc_b[i][j];
          }
        }
        syclexp::work_group_memory<T[N][M]> temp2;
        temp2 = temp;
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < M; ++j) {
            acc_b[i][j] = temp2[i][j];
          }
        }
      });
    });
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      // Two swaps are the same as no swap
      assert(a[i][j] == old_a[i][j] && b[i][j] == old_b[i][j] &&
             "Incorrect swap!");
    }
  }

  // Same as above but construct the second temporary inside the kernel and
  // copy-construct it from the first temporary.
  {
    sycl::buffer<T, 2> buf_a{a[0], sycl::range{N, M}};
    sycl::buffer<T, 2> buf_b{b[0], sycl::range{N, M}};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N][M]> temp{cgh};
      syclexp::work_group_memory<T[N][M]> temp2{cgh};
      cgh.single_task([=]() {
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < M; ++j) {
            temp[i][j] = acc_a[i][j];
            acc_a[i][j] = acc_b[i][j];
          }
        }
        syclexp::work_group_memory<T[N][M]> temp2{temp};
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < M; ++j) {
            acc_b[i][j] = temp2[i][j];
          }
        }
      });
    });
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      // Three swaps are the same as one swap
      assert(a[i][j] == old_b[i][j] && b[i][j] == old_a[i][j] &&
             "Incorrect swap!");
    }
  }
}

constexpr size_t N = 100;
constexpr size_t M = 100;
int main() {
  int intarr1[N][M];
  int intarr2[N][M];
  float floatarr1[N][M];
  float floatarr2[N][M];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      intarr1[i][j] = i + j;
      intarr2[i][j] = i * j;
      floatarr1[i][j] = (i + 1) / (j + 1);
      floatarr2[i][j] = (j + 1) / (i + 1);
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      swap_scalar(intarr1[i][j], intarr2[i][j]);
      swap_scalar(floatarr1[i][j], floatarr2[i][j]);
    }
    swap_array_1d(intarr1[i], intarr2[i]);
    swap_array_1d(floatarr1[i], floatarr2[i]);
  }
  swap_array_2d(intarr1, intarr2);
  swap_array_2d(floatarr1, floatarr2);
  return 0;
}
