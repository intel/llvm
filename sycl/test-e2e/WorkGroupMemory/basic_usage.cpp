// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <cassert>
#include <cstring>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/half_type.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

sycl::queue q;

// This test performs a swap of two scalars/arrays inside a kernel using a
// work_group_memory object as a temporary buffer. The test is done for scalar
// types and bounded arrays. After the kernel finishes, it is verified on the
// host side that the swap worked.

template <typename T> void swap_scalar(T &a, T &b) {
  const T old_a = a;
  const T old_b = b;
  const size_t size = 1;
  const size_t wgsize = 1;
  {
    sycl::buffer<T, 1> buf_a{&a, 1};
    sycl::buffer<T, 1> buf_b{&b, 1};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T> temp{cgh};
      sycl::nd_range<1> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<1> it) {
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
      sycl::nd_range<1> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<1> it) {
        syclexp::work_group_memory<T> temp2{syclexp::indeterminate};
        temp2 = temp;            // temp and temp2 have the same underlying data
        assert(&temp2 == &temp); // check that both objects return same
                                 // underlying address after assignment
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
      sycl::nd_range<1> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<> it) {
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
  // Also verify that get_multi_ptr() returns the same address as address-of
  // operator.
  {
    sycl::buffer<T, 1> buf_a{&a, 1};
    sycl::buffer<T, 1> buf_b{&b, 1};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T> temp{cgh};
      syclexp::work_group_memory<T> temp2{cgh};
      sycl::nd_range<1> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<> it) {
        assert(&temp == temp.get_multi_ptr().get());
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

// Swap two 1d arrays in batches of size batch_size where each batch will be
// swapped by items in the same work group.
template <typename T, size_t N>
void swap_array_1d(T (&a)[N], T (&b)[N], size_t batch_size) {
  sycl::queue q;
  T old_a[N];
  std::memcpy(old_a, a, sizeof(a));
  T old_b[N];
  std::memcpy(old_b, b, sizeof(b));
  const size_t size = N;
  const size_t wgsize = batch_size;
  {
    sycl::buffer<T, 1> buf_a{a, N};
    sycl::buffer<T, 1> buf_b{b, N};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N]> temp{cgh};
      sycl::nd_range<1> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<> it) {
        const auto i = it.get_global_id();
        temp[i] = acc_a[i];
        acc_a[i] = acc_b[i];
        acc_b[i] = temp[i];
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
      sycl::nd_range<1> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<> it) {
        auto ptr = temp.get_multi_ptr();
        const auto i = it.get_global_id();
        ptr[i] = acc_a[i];
        acc_a[i] = acc_b[i];
        acc_b[i] = ptr[i];
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
      sycl::nd_range<1> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<> it) {
        const auto i = it.get_global_id();
        auto ptr = &temp;
        (*ptr)[i] = acc_a[i];
        acc_a[i] = acc_b[i];
        acc_b[i] = (*ptr)[i];
      });
    });
  }
  // Three swaps same as one swap
  for (int i = 0; i < N; ++i) {
    assert(a[i] == old_b[i] && b[i] == old_a[i] && "Incorrect swap!");
  }

  // Same as above but use an unbounded array as temporary storage
  {
    sycl::buffer<T, 1> buf_a{a, N};
    sycl::buffer<T, 1> buf_b{b, N};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[]> temp{N, cgh};
      sycl::nd_range<1> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<> it) {
        const auto i = it.get_global_id();
        auto ptr = &temp;
        (*ptr)[i] = acc_a[i];
        acc_a[i] = acc_b[i];
        acc_b[i] = (*ptr)[i];
      });
    });
  }
  // Four swaps same as no swap
  for (int i = 0; i < N; ++i) {
    assert(a[i] == old_a[i] && b[i] == old_b[i] && "Incorrect swap!");
  }
}

template <typename T, size_t N>
void swap_array_2d(T (&a)[N][N], T (&b)[N][N], size_t batch_size) {
  sycl::queue q;
  T old_a[N][N];
  for (int i = 0; i < N; ++i) {
    std::memcpy(old_a[i], a[i], sizeof(a[0]));
  }
  T old_b[N][N];
  for (int i = 0; i < N; ++i) {

    std::memcpy(old_b[i], b[i], sizeof(b[0]));
  }
  const auto size = sycl::range{N, N};
  const auto wgsize = sycl::range{batch_size, batch_size};
  {
    sycl::buffer<T, 2> buf_a{a[0], sycl::range{N, N}};
    sycl::buffer<T, 2> buf_b{b[0], sycl::range{N, N}};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N][N]> temp{cgh};
      sycl::nd_range<2> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<2> it) {
        const auto i = it.get_global_id()[0];
        const auto j = it.get_global_id()[1];
        temp[i][j] = acc_a[i][j];
        acc_a[i][j] = acc_b[i][j];
        acc_b[i][j] = temp[i][j];
      });
    });
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      assert(a[i][j] == old_b[i][j] && b[i][j] == old_a[i][j] &&
             "Incorrect swap!");
    }
  }

  // Perform the swap but this time use two temporary work group memory objects.
  // One will save the value of acc_a and the other will be copy-assigned from
  // it and will be used to write the values back to acc_b.
  {
    sycl::buffer<T, 2> buf_a{a[0], sycl::range{N, N}};
    sycl::buffer<T, 2> buf_b{b[0], sycl::range{N, N}};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N][N]> temp{cgh};
      sycl::nd_range<2> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<2> it) {
        const auto i = it.get_global_id()[0];
        const auto j = it.get_global_id()[1];
        temp[i][j] = acc_a[i][j];
        acc_a[i][j] = acc_b[i][j];
        syclexp::work_group_memory<T[N][N]> temp2{syclexp::indeterminate};
        temp2 = temp;
        acc_b[i][j] = temp2[i][j];
      });
    });
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      // Two swaps are the same as no swap
      assert(a[i][j] == old_a[i][j] && b[i][j] == old_b[i][j] &&
             "Incorrect swap!");
    }
  }

  // Same as above but construct the second temporary inside the kernel and
  // copy-construct it from the first temporary.
  {
    sycl::buffer<T, 2> buf_a{a[0], sycl::range{N, N}};
    sycl::buffer<T, 2> buf_b{b[0], sycl::range{N, N}};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[N][N]> temp{cgh};
      sycl::nd_range<2> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<2> it) {
        const auto i = it.get_global_id()[0];
        const auto j = it.get_global_id()[1];
        temp[i][j] = acc_a[i][j];
        acc_a[i][j] = acc_b[i][j];
        syclexp::work_group_memory<T[N][N]> temp2{temp};
        assert(&temp2 == &temp); // check both objects return same underlying
                                 // address after copy construction.
        acc_b[i][j] = temp2[i][j];
      });
    });
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      // Three swaps are the same as one swap
      assert(a[i][j] == old_b[i][j] && b[i][j] == old_a[i][j] &&
             "Incorrect swap!");
    }
  }

  // Same as above but use an unbounded array as temporary storage
  {
    sycl::buffer<T, 2> buf_a{a[0], sycl::range{N, N}};
    sycl::buffer<T, 2> buf_b{b[0], sycl::range{N, N}};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh};
      sycl::accessor acc_b{buf_b, cgh};
      syclexp::work_group_memory<T[][N]> temp{N, cgh};
      sycl::nd_range<2> ndr{size, wgsize};
      cgh.parallel_for(ndr, [=](sycl::nd_item<2> it) {
        const auto i = it.get_global_id()[0];
        const auto j = it.get_global_id()[1];
        temp[i][j] = acc_a[i][j];
        acc_a[i][j] = acc_b[i][j];
        syclexp::work_group_memory<T[][N]> temp2{temp};
        acc_b[i][j] = temp2[i][j];
      });
    });
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      // Four swaps are the same as no swap
      assert(a[i][j] == old_a[i][j] && b[i][j] == old_b[i][j] &&
             "Incorrect swap!");
    }
  }
}

// Coherency test that checks that work group memory is truly shared by
// work-items in a work group. The test allocates an integer in
// work group memory and each leader of the work groups will assign
// its work group index to this integer. The computation that the
// leader does yields the same value for all work-items in the work-group
// so we can verify that each work-item sees the value written by its leader.
// The test also is a sanity check that different work groups get different
// work group memory locations as otherwise we'd have data races.
template <typename T> void coherency(size_t size, size_t wgsize) {
  q.submit([&](sycl::handler &cgh) {
    syclexp::work_group_memory<T> data{cgh};
    sycl::nd_range<1> ndr{size, wgsize};
    cgh.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      if (it.get_group().leader()) {
        data = T(it.get_global_id() / wgsize);
      }
      sycl::group_barrier(it.get_group());
      assert(data == T(it.get_global_id() / wgsize));
    });
  });
}

constexpr size_t N = 32;
template <typename T> void test() {
  T intarr1[N][N];
  T intarr2[N][N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      intarr1[i][j] = T(i) + T(j);
      intarr2[i][j] = T(i) * T(j);
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      swap_scalar(intarr1[i][j], intarr2[i][j]);
    }
    swap_array_1d(intarr1[i], intarr2[i], 8);
  }
  swap_array_2d(intarr1, intarr2, 8);
  coherency<T>(N, N / 2);
  coherency<T>(N, N / 4);
  coherency<T>(N, N / 8);
  coherency<T>(N, N / 16);
  coherency<T>(N, N / 32);
}

template <typename T> void test_ptr() {
  T arr1[N][N];
  T arr2[N][N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      swap_scalar(arr1[i][j], arr2[i][j]);
    }
    swap_array_1d(arr1[i], arr2[i], 8);
  }
  swap_array_2d(arr1, arr2, 8);
}

int main() {
  test<int>();
  test<char>();
  test<uint16_t>();
  if (q.get_device().has(sycl::aspect::fp16))
    test<sycl::half>();
  test_ptr<float *>();
  test_ptr<int *>();
  test_ptr<char *>();
  test_ptr<uint16_t *>();
  if (q.get_device().has(sycl::aspect::fp16))
    test_ptr<sycl::half *>();
  test_ptr<float *>();
  return 0;
}
