import unittest

from normalizer.data_utils import ml_normalizer
from normalizer.hindi import cardinal


def dev(*code_points):
    return "".join(map(chr, code_points))


PAANCH = dev(0x092A, 0x093E, 0x0901, 0x091A)  # 5
PAANCH_ALT = dev(0x092A, 0x093E, 0x0902, 0x091A)
CHHAH_ALT = dev(0x091B, 0x0939)  # 6
ADTALEES = dev(0x0905, 0x0921, 0x093C, 0x0924, 0x093E, 0x0932, 0x0940, 0x0938)  # 48
PRATISHAT = dev(0x092A, 0x094D, 0x0930, 0x0924, 0x093F, 0x0936, 0x0924)
AUR = dev(0x0914, 0x0930)


def hi(text):
    return ml_normalizer(text, lang="hi")


class HindiNumbersTest(unittest.TestCase):
    def test_cardinal(self):
        self.assertEqual(cardinal(48), ADTALEES)
        self.assertEqual(cardinal(1500), " ".join([cardinal(1), dev(0x0939, 0x091C, 0x093C, 0x093E, 0x0930), PAANCH, dev(0x0938, 0x094C)]))

    def test_digits(self):
        self.assertEqual(hi("48"), ADTALEES)
        self.assertEqual(hi(dev(0x096A, 0x096E)), ADTALEES)
        self.assertEqual(hi("5"), hi(PAANCH_ALT))
        self.assertEqual(hi("6"), hi(CHHAH_ALT))
        self.assertEqual(hi("20%"), hi("20 " + PRATISHAT))

    def test_separators(self):
        self.assertEqual(hi("1,23,456"), hi("123456"))
        self.assertEqual(hi("1,234,567"), hi("1234567"))
        self.assertEqual(hi("3.5"), hi("3 5"))

    def test_other_languages_unchanged(self):
        self.assertEqual(ml_normalizer("3,5% und 10.000", lang="de"), "drei fünf und zehntausend")
        self.assertEqual(ml_normalizer("5 " + AUR + " 10"), "5 " + AUR + " 10")


if __name__ == "__main__":
    unittest.main()
