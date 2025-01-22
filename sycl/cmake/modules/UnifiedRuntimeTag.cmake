set(UNIFIED_RUNTIME_REPO "https://github.com/RossBrunton/unified-runtime.git")
# commit 431e7dc702d0b69cc143b176c9d7daebe9e6fc46
# Author: Ross Brunton <ross@codeplay.com>
# Date:   Wed Jan 8 16:08:16 2025 +0000
#     Make profiling info optional and update tests
#
#     This patch turns all of the values returned by urEventGetProfilingInfo
#     to be optional and updates adapters to handle this by returning the
#     appropriate enum when it is not supported.
#
#     The tests have also been updated, to ensure that returning a counter of
#     "0" or values equal to the previous profiling event is no longer
#     considered a failure.
set(UNIFIED_RUNTIME_TAG 431e7dc702d0b69cc143b176c9d7daebe9e6fc46)
