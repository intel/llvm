// Intel OpenCL CPU Runtime supports device partition on all (multi-core)
// platforms. Other devices may not support this.
// REQUIRES: cpu
//
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out separate equally | FileCheck %s --check-prefix CHECK-SEPARATE
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out shared equally | FileCheck %s --check-prefix CHECK-SHARED --implicit-check-not piContextCreate --implicit-check-not piMemBufferCreate
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out fused  equally | FileCheck %s --check-prefix CHECK-FUSED --implicit-check-not piContextCreate --implicit-check-not piMemBufferCreate

#include <iostream>
#include <string>
#include <sycl/detail/core.hpp>
#include <vector>

using namespace sycl;

// Log to the same stream as SYCL_PI_TRACE
static void log_pi(const char *msg) { std::cout << msg << std::endl; }

static void use_mem(buffer<int, 1> buf, queue q) {
  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class sum1>(range<1>(buf.size()),
                                 [=](item<1> itemID) { acc[itemID] += 1; });
  });
  q.wait();
}

typedef std::vector<device> (*partition_fn)(device dev);

// FIXME: `partition_by_affinity_domain' is currently not tested: OpenCL CPU
// device only supports `partition_equally'.
static std::vector<device> partition_affinity(device dev) {
  std::vector<device> subdevices = dev.create_sub_devices<
      info::partition_property::partition_by_affinity_domain>(
      info::partition_affinity_domain::next_partitionable);

  return subdevices;
}

static std::vector<device> partition_equally(device dev) {
  std::vector<device> subdevices =
      dev.create_sub_devices<info::partition_property::partition_equally>(1);

  return subdevices;
}

static bool check_separate(device dev, buffer<int, 1> buf,
                           partition_fn partition) {
  log_pi("Create sub devices");
  std::vector<device> subdevices = partition(dev);
  assert(subdevices.size() > 1);
  // CHECK-SEPARATE: Create sub devices
  // CHECK-SEPARATE: ---> piDevicePartition

  log_pi("Test sub device 0");
  {
    queue q0(subdevices[0]);
    use_mem(buf, q0);
  }
  // CHECK-SEPARATE: Test sub device 0
  // CHECK-SEPARATE: ---> piContextCreate
  // CHECK-SEPARATE: ---> piextQueueCreate
  // CHECK-SEPARATE: ---> piMemBufferCreate
  // CHECK-SEPARATE: ---> piEnqueueKernelLaunch
  // CHECK-SEPARATE: ---> piQueueFinish

  log_pi("Test sub device 1");
  {
    queue q1(subdevices[1]);
    use_mem(buf, q1);
  }
  // CHECK-SEPARATE: Test sub device 1
  // CHECK-SEPARATE: ---> piContextCreate
  // CHECK-SEPARATE: ---> piextQueueCreate
  // CHECK-SEPARATE: ---> piMemBufferCreate
  //
  // Verify that we have a memcpy between subdevices in this case
  // CHECK-SEPARATE: ---> piEnqueueMemBuffer{{Map|Read}}
  // CHECK-SEPARATE: ---> piEnqueueMemBufferWrite
  //
  // CHECK-SEPARATE: ---> piEnqueueKernelLaunch
  // CHECK-SEPARATE: ---> piQueueFinish

  return true;
}

static bool check_shared_context(device dev, buffer<int, 1> buf,
                                 partition_fn partition) {
  log_pi("Create sub devices");
  std::vector<device> subdevices = partition(dev);
  assert(subdevices.size() > 1);
  // CHECK-SHARED: Create sub devices
  // CHECK-SHARED: ---> piDevicePartition

  // Shared context: queues are bound to specific subdevices, but
  // memory does not migrate
  log_pi("Create shared context");
  context shared_context(subdevices);
  // CHECK-SHARED: Create shared context
  // CHECK-SHARED: ---> piContextCreate
  //
  // Make sure that a single context is created: see --implicit-check-not above.

  log_pi("Test sub device 0");
  {
    queue q0(shared_context, subdevices[0]);
    use_mem(buf, q0);
  }
  // CHECK-SHARED: Test sub device 0
  // CHECK-SHARED: ---> piextQueueCreate
  // CHECK-SHARED: ---> piMemBufferCreate
  //
  // Make sure that a single buffer is created (and shared between subdevices):
  // see --implicit-check-not above.
  //
  // CHECK-SHARED: ---> piEnqueueKernelLaunch
  // CHECK-SHARED: ---> piQueueFinish

  log_pi("Test sub device 1");
  {
    queue q1(shared_context, subdevices[1]);
    use_mem(buf, q1);
  }
  // CHECK-SHARED: Test sub device 1
  // CHECK-SHARED: ---> piextQueueCreate
  // CHECK-SHARED: ---> piEnqueueKernelLaunch
  // CHECK-SHARED: ---> piQueueFinish
  // CHECK-SHARED: ---> piEnqueueMemBufferRead

  return true;
}

static bool check_fused_context(device dev, buffer<int, 1> buf,
                                partition_fn partition) {
  log_pi("Create sub devices");
  std::vector<device> subdevices = partition(dev);
  assert(subdevices.size() > 1);
  // CHECK-FUSED: Create sub devices
  // CHECK-FUSED: ---> piDevicePartition

  // Fused context: same as shared context, but also includes the root device
  log_pi("Create fused context");
  std::vector<device> devices;
  devices.push_back(dev);
  devices.push_back(subdevices[0]);
  devices.push_back(subdevices[1]);
  context fused_context(devices);
  // CHECK-FUSED: Create fused context
  // CHECK-FUSED: ---> piContextCreate
  //
  // Make sure that a single context is created: see --implicit-check-not above.

  log_pi("Test root device");
  {
    queue q(fused_context, dev);
    use_mem(buf, q);
  }
  // CHECK-FUSED: Test root device
  // CHECK-FUSED: ---> piextQueueCreate
  // CHECK-FUSED: ---> piMemBufferCreate
  //
  // Make sure that a single buffer is created (and shared between subdevices
  // *and* the root device): see --implicit-check-not above.
  //
  // CHECK-FUSED: ---> piEnqueueKernelLaunch
  // CHECK-FUSED: ---> piQueueFinish

  log_pi("Test sub device 0");
  {
    queue q0(fused_context, subdevices[0]);
    use_mem(buf, q0);
  }
  // CHECK-FUSED: Test sub device 0
  // CHECK-FUSED: ---> piextQueueCreate
  // CHECK-FUSED: ---> piEnqueueKernelLaunch
  // CHECK-FUSED: ---> piQueueFinish

  log_pi("Test sub device 1");
  {
    queue q1(fused_context, subdevices[1]);
    use_mem(buf, q1);
  }
  // CHECK-FUSED: Test sub device 1
  // CHECK-FUSED: ---> piextQueueCreate
  // CHECK-FUSED: ---> piEnqueueKernelLaunch
  // CHECK-FUSED: ---> piQueueFinish
  // CHECK-FUSED: ---> piEnqueueMemBufferRead

  return true;
}

int main(int argc, const char **argv) {
  assert(argc == 3 && "Invalid number of arguments");
  std::string test(argv[1]);
  std::string partition_type(argv[2]);

  device dev(default_selector_v);

  std::vector<int> host_mem(1024, 1);
  buffer<int, 1> buf(&host_mem[0], host_mem.size());

  partition_fn partition;
  if (partition_type == "equally") {
    partition = partition_equally;
  } else if (partition_type == "affinity") {
    partition = partition_affinity;
  } else {
    assert(0 && "Unsupported partition type");
  }

  bool result = false;
  if (test == "separate") {
    result = check_separate(dev, buf, partition);
  } else if (test == "shared") {
    result = check_shared_context(dev, buf, partition);
  } else if (test == "fused") {
    result = check_fused_context(dev, buf, partition);
  } else {
    assert(0 && "Unknown test");
  }

  if (!result) {
    fprintf(stderr, "FAILED\n");
    return EXIT_FAILURE;
  }
}
