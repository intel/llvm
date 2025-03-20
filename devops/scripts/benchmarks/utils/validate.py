import re

class Validate:
    """Static class containing methods for validating various fields"""

    @staticmethod
    def filepath(path: str) -> bool:
        """
        Returns True if path is clean (no illegal characters), otherwise False.
        """
        filepath_re = re.compile(r"[a-zA-Z0-9\/\._\-]+")
        return filepath_re.match(path) is not None

    @staticmethod
    def timestamp(t: str) -> bool:
        """
        Returns True if t is in form YYYYMMDD_HHMMSS, otherwise False.
        """
        timestamp_re = re.compile(
            r"^\d{4}(0[1-9]|1[0-2])([0-2][0-9]|3[01])_([01][0-9]|2[0-3])[0-5][0-9][0-5][0-9]$"
        )
        return timestamp_re.match(t) is not None