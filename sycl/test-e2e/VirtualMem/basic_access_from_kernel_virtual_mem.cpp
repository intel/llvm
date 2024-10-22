// REQUIRES: aspect-ext_oneapi_virtual_mem

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

namespace syclext = sycl::ext::oneapi::experimental;

int main() {
    sycl::queue Q;
    sycl::context Context = Q.get_context();

    size_t UsedGranularityInBytes = syclext::get_mem_granularity(Context, syclext::granularity_mode::recommended);
       
    syclext::physical_mem NewPhysicalMem{Q.get_device(), Context, UsedGranularityInBytes};
    uintptr_t VirtualMemoryPtr = syclext::reserve_virtual_mem(0, UsedGranularityInBytes, Context);

    void *MappedPtr = NewPhysicalMem.map(VirtualMemoryPtr, UsedGranularityInBytes, syclext::address_access_mode::read_write);

    int* DataPtr = reinterpret_cast<int*>(MappedPtr);

    sycl::range NumItems{UsedGranularityInBytes/sizeof(int)};
    
    std::vector<int> ResultHostData(NumItems.size());
    
    constexpr int ExpectedValueAfterFill = 1;

    Q.fill(DataPtr,ExpectedValueAfterFill , NumItems.size()).wait_and_throw();
    {
        sycl::buffer<int> CheckBuffer(ResultHostData);
        Q.submit([&](auto &handle) {
            sycl::accessor A(CheckBuffer, handle, sycl::write_only);
            handle.parallel_for(NumItems, [=](auto i) { A[i] =  DataPtr[i]; });
        });
    }
    
    for (size_t i = 0; i < ResultHostData.size(); i++) {
        if (ResultHostData[i] != ExpectedValueAfterFill) {
            std::cout << "Comparison failed after fill operation at index " << i << ": " << ResultHostData[i]
                  << " != " <<ExpectedValueAfterFill << std::endl;
            return 1;
        }
    }
        
    Q.submit([&](auto &handle) {
        handle.parallel_for(NumItems, [=](auto i) {
           DataPtr[i] = i; 
        });
    }).wait_and_throw();
        
    syclext::set_access_mode(DataPtr,UsedGranularityInBytes, syclext::address_access_mode::read, Context);

    
    {
    sycl::buffer<int> ResultBuffer(ResultHostData);
    
    Q.submit([&](auto &handle) {
        sycl::accessor A(ResultBuffer, handle, sycl::write_only);
        handle.parallel_for(NumItems, [=](auto i) { A[i] =  DataPtr[i]; });
    });
    }
    
    for (size_t i = 0; i < NumItems.size(); i++) {
        const int ExpectedValue = static_cast<int>(i); 
        if (ResultHostData[i] != ExpectedValue) {
            std::cout << "Comparison failed at index " << i << ": " << ResultHostData[i]
                  << " != " << ExpectedValue<< std::endl;
            return 1;
        }
    }
    
    syclext::unmap(MappedPtr, UsedGranularityInBytes, Context);
    syclext::free_virtual_mem(VirtualMemoryPtr, UsedGranularityInBytes, Context);

    return 0;
}