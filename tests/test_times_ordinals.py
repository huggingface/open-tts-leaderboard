"""Times, ordinals and Roman numerals normalize to the way they are read (normalizer/times_ordinals.py).

Run from the repository root: python -m unittest discover tests
"""

import unittest

from normalizer.data_utils import ml_normalizer, normalizer


def norm(lang, text):
    # As score_results compares them: normalized, with hyphens folded to spaces.
    return (normalizer(text) if lang == "en" else ml_normalizer(text, lang=lang)).replace("-", " ")


class TimesOrdinalsTest(unittest.TestCase):
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

    def test_english_times(self):
        self.assertSame("en", "At 1:15 a.m. Saturday", "At one fifteen a.m. Saturday", "At 1:15 AM Saturday")
        self.assertSame("en", "at about 5am (eastern time)", "at about 5:00 a.m. eastern time", "at about five a.m. eastern time")
        self.assertDiffer("en", "At 1:15 a.m. Saturday", "At one a.m. Saturday", "At eight a.m. Saturday")

    def test_full_hours(self):
        self.assertSame("de", "Die Spiele starteten um 10:00 Uhr bei", "Die Spiele starteten um zehn Uhr bei",
                        "Die Spiele starteten um 10 Uhr bei")
        self.assertSame("es", "hora local (15:00 UTC).", "hora local, quince UTC.", "hora local (15 UTC).")
        self.assertSame("it", "Oggi alle 10:00 il tempo", "Oggi alle dieci il tempo")
        self.assertDiffer("de", "Die Spiele starteten um 10:00 Uhr bei", "Die Spiele starteten um zehn dreißig bei")

    def test_hour_and_minutes_joined(self):
        self.assertSame("es", "A las 11:20, la policía", "A las once y veinte, la policía", "A las once veinte, la policía")
        self.assertSame("it", "alle 20:30 locali", "alle venti e trenta locali", "alle venti trenta locali")

    def test_french_times(self):
        self.assertSame("fr", "à 7 h 19 heure locale (21 h 19 GMT vendredi)",
                        "à 7h19 heure locale, 21h19 GMT vendredi",
                        "à sept heures dix-neuf heure locale, vingt et un heures dix-neuf GMT vendredi",
                        "à 7:19 heure locale, 21:19 GMT vendredi")
        self.assertDiffer("fr", "à 7 h 19 heure locale", "à sept dix-neuf heure locale")
        # A race time is not a time of day.
        self.assertSame("fr", "un temps combiné de 4:41.30", "un temps combiné de 4:41.30")
        self.assertNotIn("heures", ml_normalizer("un temps combiné de 4:41.30", lang="fr"))

    def test_ordinals(self):
        self.assertSame("fr", "le 1er avril", "le premier avril")
        self.assertSame("fr", "au XVe siècle", "au XVème siècle", "au quinzième siècle", "au 15e siècle")
        self.assertSame("fr", "au Xe siècle", "au dixième siècle")
        self.assertSame("it", "il 1° e il 3° reggimento", "il primo e il terzo reggimento")
        self.assertDiffer("it", "il 1° e il 3° reggimento", "il uno e il tre reggimento")
        # Not ordinals: degrees with a unit, a longitude, and words that look like Roman numerals.
        self.assertIn("gradi celsius", ml_normalizer("oltre i 30°C sono", lang="it"))
        for degrees in ("a est del 35°W.", "a est del 35° W.", "a est del 35° di latitudine nord.", "a est del 35° gradi W."):
            self.assertNotIn("trentacinquesimo", ml_normalizer(degrees, lang="it"))
        self.assertEqual(ml_normalizer("Le chat", lang="fr"), "le chat")

    def test_roman_numerals(self):
        self.assertSame("en", "in World War I and World War II.", "in World War One and World War Two.",
                        "in World War 1 and World War 2.")
        self.assertSame("es", "a fines del siglo IX", "a fines del siglo noveno", "a fines del siglo nueve")
        self.assertSame("es", "del siglo XX", "del siglo veinte")
        self.assertSame("it", "nel XV secolo", "nel quindicesimo secolo")
        self.assertDiffer("es", "a fines del siglo IX", "a fines del siglo XIX")
        self.assertEqual(ml_normalizer("un siglo vil", lang="es"), "un siglo vil")
        self.assertSame("fr", "sous François Ier", "sous François premier")

    def test_a_lone_i_is_not_a_numeral(self):
        # Roman numerals are read only in the contexts above, never "I" on its own.
        self.assertEqual(normalizer("Now I know, and since I left, World War I ended."),
                         "now i know and since i left world war one ended")
        self.assertEqual(ml_normalizer("I bambini e I secoli passati", lang="it"), "i bambini e i secoli passati")
        self.assertEqual(ml_normalizer("Il re Carlo I e il I secolo", lang="it"), "il re carlo i e il primo secolo")


if __name__ == "__main__":
    unittest.main()
