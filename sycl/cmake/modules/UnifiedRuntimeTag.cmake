# commit 6c54509c921c5f788ff35456b0fa8c882d9b4213 (HEAD -> main)
# Author: Artur Gainullin <artur.gainullin@intel.com>
# Date:   Wed Nov 6 16:05:50 2024 -0800
#
#     [L0] Fix binary sizes and binaries returned by urProgramGetInfo
#
#     Currently urProgramGetInfo will return UR_INVALID_PROGRAM is program is
#     compiled only for a subset of associated devices, i.e. not all devices
#     have level zero module and binaries. This PR fixes this behaviour.
set(UNIFIED_RUNTIME_TAG 6c54509c921c5f788ff35456b0fa8c882d9b4213)
