#include <OffloadAPI.h>
#include <ur_api.h>

#include "event.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(uint32_t numEvents, const ur_event_handle_t *phEventWaitList) {
  for (uint32_t i = 0; i < numEvents; i++) {
    auto Res = olWaitEvent(phEventWaitList[i]->OffloadEvent);
    if (Res) {
      return offloadResultToUR(Res);
    }
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRetain(ur_event_handle_t hEvent) {
  hEvent->RefCount++;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRelease(ur_event_handle_t hEvent) {
  if (--hEvent->RefCount == 0) {
    // There's a small bug in olDestroyEvent that will crash. Leak the event
    // in the meantime.
    // auto Res = olDestroyEvent(hEvent->OffloadEvent);
    // if (Res) {
    //   return offloadResultToUR(Res);
    // }
  }

  delete hEvent;
  return UR_RESULT_SUCCESS;
}
