import os
import re
import json
import urllib
import tempfile
import subprocess
from urllib import request
from pathlib import Path

class DetectVersion:
	_instance = None

	def __init__(self):
		raise RuntimeError("Use init() to init and instance() to get instead.")

	@classmethod
	def init(cls, detect_ver_path: Path, dpcpp_exec: str = "clang++"):
		"""
		Constructs the singleton instance for DetectVersion, and initializes by
		building and run detect _version.cpp, which outputs:
		  - L0 driver version via ZE_intel_get_driver_version_string extension,
		  - DPC++ version via `__clang_version__` builtin.

		Remind: DO NOT allow user input in args.

		Parameters:
		detect_ver_path (Path): Path to detect_version.cpp
		dpcpp_exec (str): Name of DPC++ executable
		"""
		if cls._instance is not None:
			return cls._instance

		detect_ver_exe = tempfile.mktemp()
		result = subprocess.run(
			[dpcpp_exec, "-lze_loader", detect_ver_path, "-o", detect_ver_exe],
			check=True,
			env=os.environ,
		)
		result = subprocess.run(
			[detect_ver_exe],
			check=True,
			text=True,
			capture_output=True,
			env=os.environ,
		)
		# Variables are printed to stdout, each var is on its own line
		result_vars = result.stdout.strip().split('\n')

		def get_var(var_name: str) -> str:
			var_str = next(
				filter(lambda v: re.match(f"^{var_name}='.*'", v), result_vars)
			)
			return var_str[len(f"{var_name}='"):-len("'")]

		cls._instance = cls.__new__(cls)
		cls._instance.l0_ver = get_var("L0_VER")
		cls._instance.dpcpp_ver = get_var("DPCPP_VER")
		return cls._instance

	@classmethod
	def instance(cls):
		"""
		Returns singleton instance of DetectVersion if it has been initialized
		via init(), otherwise return None.
		"""
		return cls._instance

	def get_l0_ver(self) -> str:
		"""
		Returns the full L0 version string.
		"""
		return self.l0_ver

	def get_dpcpp_ver(self) -> str:
		"""
		Returns the full DPC++ version / clang version string of DPC++ used.
		"""
		return self.dpcpp_ver

	def get_dpcpp_commit(self) -> str:
		# clang++ formats are in <clang ver> (<git url> <commit>): if this
		# regex does not match, it is likely this is not upstream clang.
		git_info_match = re.search(r'(http.+ [0-9a-f]+)', self.dpcpp_ver)
		if git_info_match is None:
			return None
		git_info = git_info_match.group(0)
		return git_info[1:-1].split(' ')[1]

	def get_compute_runtime_ver(self) -> str:
		"""
		Returns the compute-runtime version by deriving from l0 version.
		"""
		# L0 version strings follows semver: major.minor.patch+optional
		# compute-runtime version tags follow year.WW.patch.optional instead,
		# but patch, pptional is still the same across both.
		#
		# We use patch+optional from L0 to figure out compute-runtime version:
		patch = re.sub(r'^\d+\.\d+\.', '', self.l0_ver)
		patch = re.sub(r'\+', '.', patch, count=1)

		# TODO unauthenticated users only get 60 API calls per hour: this will
		# not work if we enable benchmark CI in precommit.
		url = "https://api.github.com/repos/intel/compute-runtime/tags"
		MAX_PAGINATION_CALLS = 2

		try:
			for _ in range(MAX_PAGINATION_CALLS):
				res = request.urlopen(url)
				tags = [ tag["name"] for tag in json.loads(res.read()) ]

				for tag in tags:
					tag_patch = re.sub(r'^\d+\.\d+\.', '', tag)
					# compute-runtime's cmake files produces "optional" fields
					# padded with 0's: this means e.g. L0 version string
					# 1.6.32961.200000 could be either compute-runtime ver.
					# 25.09.32961.2, 25.09.32961.20, or even 25.09.32961.200.
					#
					# Thus, we take the longest match. Since the github api
					# provides tags from newer -> older, we take the first tag
					# that matches as it would be the "longest" ver. to match.
					if tag_patch == patch[:len(tag_patch)]:
						return tag

				def get_link_name(link: str) -> str:
					rel_str = re.search(r'rel="\w+"', link).group(0)
					return rel_str[len('rel="'):-len('"')]

				def get_link_url(link: str) -> str:
					return link[link.index('<')+1:link.index('>')]

				links = {
					get_link_name(link): get_link_url(link)
					for link in res.getheader("Link").split(", ")
				}

				if "next" in links:
					url = links["next"]
				else:
					break

		except urllib.error.HTTPError as e:
			print(f"HTTP error {e.code}: {e.read().decode('utf-8')}")

		except urllib.error.URLError as e:
			print(f"URL error: {e.reason}")

		return "Unknown"


def main():
	detect_res = DetectVersion.init(f"{os.path.dirname(__file__)}/detect_versions.cpp")
	# print(query_res.get_compute_runtime_ver())
	print(detect_res.get_dpcpp_commit())
	# print(query_res.get_l0_ver(), query_res.get_dpcpp_ver())

if __name__ == "__main__":
	main()
