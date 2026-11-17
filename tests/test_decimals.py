"""Decimal numbers are read with their separator word on both sides (MultilingualNormalizer._normalize_decimals).

Run from the repository root: python -m unittest discover tests
"""

import unittest

from normalizer.data_utils import ml_normalizer


def norm(lang, text):
    # As score_results compares them: normalized, with hyphens folded to spaces.
    return ml_normalizer(text, lang=lang).replace("-", " ")


class DecimalsTest(unittest.TestCase):
    def assertSame(self, lang, reference, *readings):
        expected = norm(lang, reference)
        for reading in readings:
            with self.subTest(lang=lang, reference=reference, reading=reading):
                self.assertEqual(norm(lang, reading), expected)

    def assertDiffer(self, lang, reference, *readings):
        expected = norm(lang, reference)
        for reading in readings:
            with self.subTest(lang=lang, reference=reference, reading=reading):
                self.assertNotEqual(norm(lang, reading), expected)

    def test_separator_word(self):
        self.assertSame("de", "von etwa 3,7 Millionen", "von etwa drei Komma sieben Millionen")
        self.assertSame("fr", "(14,7 milliards de dollars US)", "quatorze virgule sept milliards de dollars US")
        self.assertSame("it", "risalenti a 4,2-3,9 milioni", "risalenti a quattro virgola due tre virgola nove milioni")
        self.assertSame("pt", "juros em 5,25% para", "juros em cinco vírgula vinte e cinco por cento para")
        self.assertSame("pt", "tem apenas 0,01 milímetros", "tem apenas zero vírgula zero um milímetros")
        self.assertSame("nl", "een kracht van 5,8 heeft", "een kracht van vijf komma acht heeft")
        self.assertSame("pl", "o sile 5,8 stopnia", "o sile pięć przecinek osiem stopnia")
        self.assertSame("cs", "o síle 5,8 stupně", "o síle pět celá osm stupně")

    def test_spanish_point_and_comma(self):
        # The Spanish references write a decimal point; the ASR writes either, or "punto"/"coma".
        self.assertSame("es", "hasta 5.8 metros de altura", "hasta 5,8 metros de altura",
                        "hasta cinco punto ocho metros de altura", "hasta cinco coma ocho metros de altura")
        self.assertEqual(ml_normalizer("un punto turístico", lang="es"), "un punto turístico")

    def test_and_a_half(self):
        self.assertSame("nl", "de rente op 3,5% te houden", "de rente op drie en een half procent te houden",
                        "de rente op drie en half procent te houden")
        self.assertSame("pl", "na poziomie 3,5% w celu", "na poziomie trzy i pół procent w celu")
        self.assertSame("cs", "sazby na 3,5 % pro", "sazby na tři a půl procent pro")
        self.assertSame("pt", "de magnitude 6,5 atingiu", "de magnitude seis e meio atingiu")

    def test_separator_left_out_is_an_error(self):
        self.assertDiffer("de", "von etwa 3,7 Millionen", "von etwa drei sieben Millionen", "von etwa 37 Millionen")
        self.assertDiffer("es", "hasta 5.8 metros de altura", "hasta cinco ocho metros de altura")

    def test_thousands_and_times_left_alone(self):
        self.assertEqual(ml_normalizer("mehr als 1.000 Pfund", lang="de"), ml_normalizer("mehr als 1,000 Pfund", lang="de"))
        self.assertEqual(ml_normalizer("Con 17.000 islas", lang="es"), "con diecisiete mil islas")
        self.assertNotIn("virgola", ml_normalizer("alle 20.30 locali", lang="it"))
        self.assertNotIn("virgule", ml_normalizer("un temps combiné de 4:41.30", lang="fr"))
        # Languages without a decimal entry keep the old behaviour.
        self.assertEqual(ml_normalizer("5,8", lang="ro"), "cinci opt")


if __name__ == "__main__":
    unittest.main()
