import re

class Validate:
    """Static class containing methods for validating various fields"""

    @staticmethod
    def runner_name(runner_name: str) -> bool:
        """
        Returns True if runner_name is clean (no illegal characters).
        """
        runner_name_re = re.compile(r"[a-zA-Z0-9_]+")
        return runner_name_re.match(runner_name) is not None

    @staticmethod
    def timestamp(t: str) -> bool:
        """
        Returns True if t is in form YYYYMMDD_HHMMSS, otherwise False.
        """
        timestamp_re = re.compile(
            r"^\d{4}(0[1-9]|1[0-2])([0-2][0-9]|3[01])_([01][0-9]|2[0-3])[0-5][0-9][0-5][0-9]$"
        )
        return timestamp_re.match(t) is not None