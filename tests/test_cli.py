import io
import unittest
from contextlib import redirect_stdout

from qeanalyzer import __version__
from qeanalyzer.cli import main


class TestCLI(unittest.TestCase):
    def test_version_is_defined(self):
        self.assertTrue(__version__)

    def test_empty_cli_prints_help(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([])
        self.assertEqual(code, 0)
        self.assertIn("qeanalyzer", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
