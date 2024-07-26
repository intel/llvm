// REQUIRES: hip_be

// RUN: %clangxx -w  -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -fsycl-unnamed-lambda \
// RUN:   --offload-arch=gfx90a -S -emit-llvm %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

// Defaults for create the atomic_ref object.
static constexpr auto scope = sycl::memory_scope::device;
static constexpr auto order = sycl::memory_order::relaxed;
static constexpr auto address_space = sycl::access::address_space::global_space;
static constexpr auto decorated = sycl::access::decorated::legacy;

void test_syncscope(sycl::queue &q, sycl::buffer<int> &buf) {
    using sycl::memory_order;
    using sycl::memory_scope;

    // CHECK: target triple = "amdgcn-amd-amdhsa"

    // test store-release atomic ops
    q.submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::write>(h);
        // CHECK-LABEL: @_ZTSZZ14test_syncscopeRN4sycl3_V15queueERNS0_6bufferIiLi1ENS0_6detail17aligned_allocatorIiEEvEEENKUlRNS0_7handlerEE_clESA_E20atomic_store_release
        h.single_task<class atomic_store_release>([=] {
            sycl::multi_ptr<int, address_space, decorated> mptr{acc};
            sycl::atomic_ref<int, order, scope, address_space> a(*mptr);

            // CHECK: store atomic volatile i32 1, ptr addrspace(1) {{.*}} syncscope("wavefront") release, align 4
            // CHECK-NOT: store atomic volatile i32 1, ptr addrspace(1) {{.*}} syncscope("wavefront-one-as") release, align 4
            a.store(1, memory_order::release, memory_scope::sub_group);

            // CHECK: store atomic volatile i32 1, ptr addrspace(1) {{.*}} syncscope("workgroup") release, align 4
            // CHECK-NOT: store atomic volatile i32 1, ptr addrspace(1) {{.*}} syncscope("workgroup-one-as") release, align 4
            a.store(1, memory_order::release, memory_scope::work_group);

            // CHECK: store atomic volatile i32 1, ptr addrspace(1) {{.*}} syncscope("agent") release, align 4
            // CHECK-NOT: store atomic volatile i32 1, ptr addrspace(1) {{.*}} syncscope("agent-one-as") release, align 4
            a.store(1, memory_order::release, memory_scope::device);

            // CHECK: store atomic volatile i32 1, ptr addrspace(1) {{.*}} release, align 4
            // CHECK-NOT: store atomic volatile i32 1, ptr addrspace(1) {{.*}} syncscope("one-as") release, align 4
            a.store(1, memory_order::release, memory_scope::system);
        });
    }).wait();

    // test load-acquire atomic ops
    q.submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::read>(h);
        h.single_task<class atomic_load_acquire>([=] {
            int x = 0;

            sycl::multi_ptr<int, address_space, decorated> mptr{acc};
	    sycl::atomic_ref<int, order, scope, address_space> a(*mptr);

            // CHECK: load atomic volatile i32, ptr addrspace(1) {{.*}} syncscope("wavefront") acquire, align 4
            // CHECK-NOT: load atomic volatile i32, ptr addrspace(1) {{.*}} syncscope("wavefront-one-as") acquire, align 4
            x = a.load(memory_order::acquire, memory_scope::sub_group);

            // CHECK: load atomic volatile i32, ptr addrspace(1) {{.*}} syncscope("workgroup") acquire, align 4
            // CHECK-NOT: load atomic volatile i32, ptr addrspace(1) {{.*}} syncscope("workgroup-one-as") acquire, align 4
            x = a.load(memory_order::acquire, memory_scope::work_group);

            // CHECK: load atomic volatile i32, ptr addrspace(1) {{.*}} syncscope("agent") acquire, align 4
            // CHECK-NOT: load atomic volatile i32, ptr addrspace(1) {{.*}} syncscope("agent-one-as") acquire, align 4
            x = a.load(memory_order::acquire, memory_scope::device);

            // CHECK: load atomic volatile i32, ptr addrspace(1) {{.*}} acquire, align 4
            // CHECK-NOT: load atomic volatile i32, ptr addrspace(1) {{.*}} syncscope("one-as") acquire, align 4
            x = a.load(memory_order::acquire, memory_scope::system);
        });
    }).wait();

    // test rmw (read-modify-write) atomic ops
    q.submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(h);
        h.single_task<class atomic_rmw_acq_rel>([=] {
            int x = 0;

            sycl::multi_ptr<int, address_space, decorated> mptr{acc};
            sycl::atomic_ref<int, order, scope, address_space> a(*mptr);

            // CHECK: atomicrmw volatile add ptr addrspace(1) {{.*}}, i32 1 syncscope("wavefront") acq_rel, align 4
            // CHECK-NOT: atomicrmw volatile add ptr addrspace(1) {{.*}}, i32 1 syncscope("wavefront-one-as") acq_rel, align 4
            x = a.fetch_add(1, memory_order::acq_rel, memory_scope::sub_group);

            // CHECK: atomicrmw volatile add ptr addrspace(1) {{.*}}, i32 1 syncscope("workgroup") acq_rel, align 4
            // CHECK-NOT: atomicrmw volatile add ptr addrspace(1) {{.*}}, i32 1 syncscope("workgroup-one-as") acq_rel, align 4
            x = a.fetch_add(1, memory_order::acq_rel, memory_scope::work_group);

            // CHECK: atomicrmw volatile add ptr addrspace(1) {{.*}}, i32 1 syncscope("agent") acq_rel, align 4
            // CHECK-NOT: atomicrmw volatile add ptr addrspace(1) {{.*}}, i32 1 syncscope("agent-one-as") acq_rel, align 4
            x = a.fetch_add(1, memory_order::acq_rel, memory_scope::device);

            // CHECK: atomicrmw volatile add ptr addrspace(1) {{.*}}, i32 1 acq_rel, align 4
            // CHECK-NOT: atomicrmw volatile add ptr addrspace(1) {{.*}}, i32 1 syncscope("one-as") acq_rel, align 4
            x = a.fetch_add(1, memory_order::acq_rel, memory_scope::system);
        });
    }).wait();
}

int main() {
  sycl::queue q{};
  sycl::buffer<int> buf(1);
  test_syncscope(q, buf);
  return 0;
}

