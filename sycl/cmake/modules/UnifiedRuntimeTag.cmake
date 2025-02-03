# commit 04882ba3abd976df81455ea5142b6b0d7c306efc
# Author: Harald van Dijk <harald.vandijk@codeplay.com>
# Date:   Fri Jan 31 17:26:43 2025 +0000
# 
#     [NativeCPU] Handle null phEvent.
# 
#     The documentation for urEnqueueKernelLaunch marks phEvent as optional,
#     so make sure we only set it if provided.
set(UNIFIED_RUNTIME_TAG 04882ba3abd976df81455ea5142b6b0d7c306efc)
