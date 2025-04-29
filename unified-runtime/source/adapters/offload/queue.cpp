#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "context.hpp"
#include "device.hpp"
#include "queue.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreate(
    [[maybe_unused]] ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *, ur_queue_handle_t *phQueue) {

  assert(hContext->Device == hDevice);

  ur_queue_handle_t Queue = new ur_queue_handle_t_();
  auto Res = olCreateQueue(hDevice->OffloadDevice, &Queue->OffloadQueue);
  if (Res != OL_SUCCESS) {
    delete Queue;
    return offloadResultToUR(Res);
  }

  Queue->OffloadDevice = hDevice->OffloadDevice;

  *phQueue = Queue;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  hQueue->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  if (--hQueue->RefCount == 0) {
    auto Res = olDestroyQueue(hQueue->OffloadQueue);
    if (Res) {
      return offloadResultToUR(Res);
    }
    delete hQueue;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  return offloadResultToUR(olWaitQueue(hQueue->OffloadQueue));
}
