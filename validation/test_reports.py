"""Synthetic negative controls for the numerical validation report parser."""
import unittest
from run import errors, compare, check_rates


class Reports(unittest.TestCase):
    row = 'DIVU_ERROR time=0.5 level=0 field=u linf=0.1 l1=0.05 l2=0.07\n'

    def test_missing_duplicate_nonfinite(self):
        for text in ('', self.row*2, self.row.replace('linf=0.1', 'linf=nan')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                errors(text, 0.5, ['u'])

    def test_offset_mismatch(self):
        a = errors(self.row, 0.5, ['u'])
        b = errors(self.row.replace('0.1 ', '0.2 '), 0.5, ['u'])
        with self.assertRaises(ValueError):
            compare(a,b)

    def test_rate_failure(self):
        a = errors(self.row, 0.5, ['u'])
        with self.assertRaises(ValueError):
            check_rates('ballistic', a,a)

    def test_initial_report_is_not_final(self):
        with self.assertRaises(ValueError):
            errors(self.row.replace('time=0.5', 'time=0'), 0.5, ['u'])


if __name__ == '__main__':
    unittest.main()
