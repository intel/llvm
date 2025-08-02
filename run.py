import re

with open("buildbot/dependency.conf") as file_:
    text = file_.read()

ocl_fpga_emu_ver = re.findall("(ocl_fpga_emu_ver)=(.*)", text)[0][1]
fpga_ver         = re.findall("(fpga_ver)=(.*)", text)[0][1]
ocl_gpu_rt_ver   = re.findall("(ocl_gpu_rt_ver)=(.*)", text)[0][1]
ocl_cpu_rt_ver   = re.findall("(ocl_cpu_rt_ver)=(.*)", text)[0][1]
tbb_ver          = re.findall("(tbb_ver)=(.*)", text)[0][1]

ocl_fpga_emu_root = re.findall("(ocl_fpga_emu_root)=(.*)", text)[0][1]
fpga_root         = re.findall("(fpga_root)=(.*)", text)[0][1]
ocl_gpu_rt_root   = re.findall("(ocl_gpu_root)=(.*)", text)[0][1]
ocl_cpu_rt_root   = re.findall("(ocl_cpu_root)=(.*)", text)[0][1]
tbb_root          = re.findall("(tbb_root)=(.*)", text)[0][1]

ARCHIVE_ROOT="/nfs/site/proj/icl/xarch/archive"
# Replace {DEPS_ROOT} -> $DEPS
ocl_fpga_emu_root = "$DIST" + ocl_fpga_emu_root[11:]
fpga_root         = ARCHIVE_ROOT + fpga_root[14:]
ocl_gpu_rt_root   = "$DIST" + ocl_gpu_rt_root[11:]
ocl_cpu_rt_root   = "$DIST" + ocl_cpu_rt_root[11:]
tbb_root          = "$DIST" + tbb_root[11:]

# Full
ocl_fpga_emu_full = f"{ocl_fpga_emu_root}/{ocl_fpga_emu_ver}"
fpga_full         = f"{fpga_root}/{fpga_ver}"
ocl_gpu_rt_full   = f"{ocl_gpu_rt_root}/{ocl_gpu_rt_ver}"
ocl_cpu_rt_full   = f"{ocl_cpu_rt_root}/{ocl_cpu_rt_ver}"
tbb_full          = f"{tbb_root}/{tbb_ver}"

print("export DIST=/rdrive/ref")
print("export DPCPP_HOME=$(cd ..; pwd)")

print(f"export PATH=$DPCPP_HOME/llvm/build/bin:{ocl_fpga_emu_full}/bin:{fpga_full}/build/bin:{ocl_gpu_rt_full}:$PATH")

print(f"export OCL_ICD_FILENAMES={ocl_fpga_emu_full}/libintelocl_emu.so:{ocl_cpu_rt_full}/libintelocl_emu.so:{ocl_gpu_rt_full}/libigdrcl.so")

print(f"export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:{ocl_fpga_emu_full}/intel64_lin:{ocl_fpga_emu_full}:{ocl_cpu_rt_full}/intel64_lin:{tbb_full}/lib/intel64/gcc4.8:{ocl_gpu_rt_full}")
