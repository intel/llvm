// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %level_zero_options %threads_lib
// RUN: %BE_RUN_PLACEHOLDER %t.out
//
// CHECK: Running iteration 0
// CHECK: Running iteration 1
// CHECK: Running iteration 2
// CHECK: Running iteration 3
// CHECK: Running iteration 4
// CHECK: Running iteration 5
// CHECK: Running iteration 6
// CHECK: Running iteration 7
// CHECK: Running iteration 8
// CHECK: Running iteration 9
// CHECK: Running iteration 10
// CHECK: Running iteration 11
// CHECK: Running iteration 12
// CHECK: Running iteration 13
// CHECK: Running iteration 14
// CHECK: Running iteration 15
// CHECK: Running iteration 16
// CHECK: Running iteration 17
// CHECK: Running iteration 18
// CHECK: Running iteration 19

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <level_zero/ze_api.h>
#include <mutex>
#include <sycl/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

using namespace std;

struct operation {
  std::vector<sycl::event> deps;
  ze_event_handle_t sync_event;
  // keep sycl events to prevent early destruction
  sycl::event sycl_event_sync;
  sycl::event return_event;
};

std::mutex mt;
std::condition_variable cv;
std::deque<operation> ops;
std::atomic<bool> stop_worker = false;

ze_driver_handle_t driver = {};
ze_device_handle_t device = {};
ze_context_handle_t context = {};
ze_event_pool_handle_t event_pool = {};

// save completed operations so they won't destroyed earlier
std::vector<operation> old_ops;

void init() {
  zeInit(0);

  uint32_t driverCount = 0;
  assert(zeDriverGet(&driverCount, nullptr) == 0);

  std::vector<ze_driver_handle_t> drivers(driverCount);
  assert(zeDriverGet(&driverCount, drivers.data()) == 0);
  assert(drivers.size() > 0);

  std::vector<ze_device_handle_t> devices;

  for (uint32_t i = 0; i < driverCount; ++i) {
    uint32_t deviceCount = 0;
    assert(zeDeviceGet(drivers[i], &deviceCount, nullptr) == 0);

    devices.resize(deviceCount);
    assert(zeDeviceGet(drivers[i], &deviceCount, devices.data()) == 0);
  }

  assert(devices.size() > 0);

  driver = drivers[0];
  device = devices[0];

  // Create context
  ze_context_desc_t ctxtDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};

  assert(zeContextCreate(driver, &ctxtDesc, &context) == 0);

  ze_event_pool_desc_t desc = {};
  desc.count = 100;
  desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;

  assert(zeEventPoolCreate(context, &desc, 1, &device, &event_pool) == 0);
}

ze_event_handle_t getEvent() {
  static uint32_t index = 0;
  ze_event_desc_t eventDesc = {};
  eventDesc.index = index++;
  eventDesc.signal = 0;
  eventDesc.wait = 0;

  ze_event_handle_t event = {};

  assert(zeEventCreate(event_pool, &eventDesc, &event) == 0);
  assert(zeEventHostReset(event) == 0);

  return event;
}

void worker() {
  std::unique_lock<std::mutex> lk(mt);

  while (true) {
    cv.wait(lk, []() { return ops.size() > 0 || stop_worker; });
    if (stop_worker)
      return;
    auto op = ops.front();
    ops.pop_front();

    for (auto dep : op.deps) {
      // Wait for dependencies to complete
      while (dep.get_info<sycl::info::event::command_execution_status>() !=
             sycl::info::event_command_status::complete) {
      }
    }

    // Do some work and signal event
    assert(zeEventHostSignal(op.sync_event) == 0);

    // Wait more to ensure event competion before we return
    while (true) {
      auto status1 = op.sycl_event_sync.template get_info<
                         sycl::info::event::command_execution_status>() ==
                     sycl::info::event_command_status::complete;
      auto status2 = op.return_event.template get_info<
                         sycl::info::event::command_execution_status>() ==
                     sycl::info::event_command_status::complete;

      if (getenv("QUERY_STATUS") != nullptr) {
        auto ev1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            op.sycl_event_sync);
        auto ev2 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            op.return_event);
        status1 = (zeEventQueryStatus(ev1) == ZE_RESULT_SUCCESS);
        status2 = (zeEventQueryStatus(ev2) == ZE_RESULT_SUCCESS);
      }

      if (status1 && status2)
        break;
    }

    old_ops.push_back(std::move(op));
  }
}

sycl::event operation(sycl::queue q) {
  std::vector<sycl::event> deps;
  deps.push_back(q.ext_oneapi_submit_barrier());

  ze_event_handle_t l0_event = getEvent();
  auto sycl_event = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
      {l0_event, sycl::ext::oneapi::level_zero::ownership::keep},
      q.get_context());

  auto return_event = q.ext_oneapi_submit_barrier({sycl_event});
  struct operation new_op = {std::move(deps), l0_event, sycl_event,
                             return_event};

  {
    std::lock_guard<std::mutex> lg(mt);
    ops.push_back(std::move(new_op));
  }
  cv.notify_all();

  return return_event;
}

int main(int argc, char *argv[]) {
  size_t count = 100;

  int size = 0;
  int rank = 0;

  size_t num_iters = 20;
  size_t kernel_num = 3;

  if (argc > 1)
    kernel_num = atoi(argv[1]);
  if (argc > 2)
    count = atoi(argv[2]);
  if (argc > 3)
    num_iters = atoi(argv[3]);

  size_t byte_count = count * 4;

  sycl::property_list props{sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}};
  sycl::queue q{props};

  init();

  // Store allocated mem ptrs to free them later
  std::vector<std::pair<float *, float *>> ptrs(kernel_num);
  // allocate all the buffers
  for (size_t i = 0; i < kernel_num; i++) {
    float *weight_buf = (float *)sycl::malloc_device(byte_count, q);
    float *weight_allreduce_buf = (float *)sycl::malloc_device(byte_count, q);
    ptrs[i] = {weight_buf, weight_allreduce_buf};
  }

  std::vector<std::tuple<sycl::event, sycl::event>> kernel_events(num_iters *
                                                                  kernel_num);

  std::vector<sycl::event> barrier_events;

  std::thread worker_thread(worker);

  for (size_t i = 0; i < num_iters; ++i) {
    std::cout << "Running iteration " << i << std::endl;

    for (size_t j = 0; j < kernel_num; j++) {
      size_t num = i * kernel_num + j;
      float *weight_buf = ptrs[j].first;
      float *weight_allreduce_buf = ptrs[j].second;

      // Step1: FWK kernel submission
      sycl::event submit_event;
      if (i == 0) {
        submit_event = q.submit([&](auto &h) {
          h.parallel_for(count, [=](auto id) {
            // Initial weight in first iteration
            weight_buf[id] = j * (rank + 1);
          });
        });
      } else {
        submit_event = q.submit([&](auto &h) {
          h.parallel_for(count, [=](auto id) {
            // Make weight differ in each iteration
            weight_buf[id] = weight_buf[id] + (j * (rank + 1));
          });
        });
      }

      barrier_events.push_back(operation(q));

      // Step3: Weight update
      auto update_event = q.submit([&](auto &h) {
        h.parallel_for(count, [=](auto id) {
          // Update weight in each iteration
          weight_buf[id] = weight_allreduce_buf[id] * 0.5;
        });
      });

      kernel_events[num] = {submit_event, update_event};
    }
    q.wait();
  }

  // Make sure there is no exceptions in the queue
  q.wait_and_throw();

  for (auto p : ptrs) {
    sycl::free(p.first, q);
    sycl::free(p.second, q);
  }

  stop_worker = true;
  cv.notify_all();
  worker_thread.join();
  return 0;
}
