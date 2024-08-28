// RUN: %{build} -o %{t.out}
// RUN: %{run} %{t.out} 

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <cassert>
#include <cstring>

namespace syclexp = sycl::ext::oneapi::experimental;

// This test performs a swap of two scalars/arrays inside a kernel using a work_group_memory object as a temporary buffer.
// The test is done for scalars types, bounded and unbounded arrays. After the kernel finishes, it is verified on the host side 
// that the swap worked.

template< typename T>
void swap_scalar(T& a, T& b) {
	sycl::queue q;
	const T old_a = a;
	const T old_b = b;
	{
	sycl::buffer<T, 1> buf_a{ &a, 1};
	sycl::buffer<T, 1> buf_b{ &b, 1};
	q.submit([&](sycl::handler &cgh) {
	sycl::accessor acc_a{ buf_a, cgh };
	sycl::accessor acc_b { buf_b, cgh };
	syclexp::work_group_memory<T> temp{ cgh };
	cgh.single_task([=]() {
		temp = acc_a[0];
		acc_a[0] = acc_b[0];
		acc_b[0] = temp;
	});});
	}
	assert(a == old_b && b == old_a	&& "Swap assertion failed");
}

template<typename T, size_t N>
void swap_bounded_array_1d(T (&a)[N], T (&b)[N]) {
sycl::queue q;
        T old_a[N];
  	std::memcpy(old_a, a, sizeof(a));	
        T old_b[N];
	std::memcpy(old_b, b, sizeof(b));
        { 
        sycl::buffer<T, 1> buf_a{ a, N};
        sycl::buffer<T, 1> buf_b{ b, N};
        q.submit([&](sycl::handler &cgh) {
        sycl::accessor acc_a{ buf_a, cgh };
        sycl::accessor acc_b { buf_b, cgh };
        syclexp::work_group_memory<T[N]> temp{ cgh };
        cgh.single_task([=]() {
for (int i= 0; i < N; ++i) {
                temp[i] = acc_a[i];
                acc_a[i] = acc_b[i];
                acc_b[i] = temp[i];
}
        });});
        }
for (int i = 0; i < N; ++i) {
        assert(a[i] == old_b[i] && b[i] == old_a[i] && "Swap assertion failed");
}
	
}
int main() {
	int a = 25;
	int b = 42;
	int arr1[5] = {0, 1, 2, 3, 4};
	int arr2[5] = {5, 6, 7, 8, 9};
	swap_scalar(a, b);
	swap_bounded_array_1d(arr1, arr2);
return 0;
}
	
	
	
	
	
	
	
	
	
