"""Units, symbols and abbreviations normalize to one spoken form on both sides (normalizer/spoken_forms.py).

Run from the repository root: python -m unittest discover tests
"""

import unittest

from normalizer.data_utils import ml_normalizer, normalizer


def norm(lang, text):
    # As score_results compares them: normalized, with hyphens folded to spaces.
    return (normalizer(text) if lang == "en" else ml_normalizer(text, lang=lang)).replace("-", " ")


class SpokenFormsTest(unittest.TestCase):
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

    def test_percent(self):
        self.assertSame("de", "um etwa 12% gestiegen", "um etwa 12 Prozent gestiegen", "um etwa zwölf Prozent gestiegen")
        self.assertSame("es", "el 99 % de nuestra historia", "el 99% de nuestra historia", "el noventa y nueve por ciento de nuestra historia")
        self.assertSame("fr", "environ 10% au cours", "environ dix pour cent au cours")
        self.assertSame("it", "circa il 12% negli", "circa il dodici per cento negli")
        self.assertSame("pt", "Quase 9% dos portugueses", "Quase nove por cento dos portugueses")
        self.assertSame("nl", "ongeveer 12% gestegen", "ongeveer twaalf procent gestegen")
        self.assertSame("pl", "o około 12% w ciągu", "o około dwanaście procent w ciągu")
        self.assertSame("cs", "přibližně o 12 %, což", "přibližně o dvanáct procent, což", "přibližně o 12 procenta, což")
        self.assertSame("ro", "cu aproximativ 12% în", "cu aproximativ 12 la sută în")
        self.assertSame("el", "περίπου 12% τους", "περίπου 12 τοις εκατό τους")
        self.assertSame("ru", "примерно на 12% за", "примерно на 12 процентов за", "примерно на 12 процента за")

    def test_units_after_a_number(self):
        self.assertSame("de", "bis zu 480 km/h (133 m/s; 300 MPH)",
                        "bis zu 480 Kilometer pro Stunde, 133 Meter pro Sekunde, 300 Meilen pro Stunde",
                        "bis zu 480 km/h, 133 m/s, 300 mph")
        self.assertSame("de", "200 Pfund (90 kg) wiegt", "200 Pfund, 90 Kilogramm wiegt", "200 Pfund, 90 Kilo wiegt")
        self.assertSame("de", "etwa 100 Fuß (30 m) vor dem Zaun", "etwa 100 Fuß, 30 Meter vor dem Zaun")
        self.assertSame("es", "un camino de 1600 km desde", "un camino de 1600 kilómetros desde")
        self.assertSame("es", "de 40 mph (64 km/h).", "de cuarenta millas por hora, sesenta y cuatro kilómetros por hora.",
                        "de 40 mph, 64 kilómetros/hora.", "de 40 mph, 64 kilómetros/h.")
        self.assertSame("es", "de 220 km (140 mi) de Pekín", "de 220 kilómetros, 140 millas de Pekín")
        self.assertSame("fr", "des vents de 165 km/h (105 miles par heure)", "des vents de 165 kilomètres par heure, 105 mph",
                        "des vents de 165 kilomètres-heure, 105 miles par heure")
        self.assertSame("fr", "Le film standard 35 mm", "Le film standard trente-cinq millimètres")
        self.assertSame("it", "verso la Cina a 11 km/h.", "verso la Cina a 11 chilometri orari.",
                        "verso la Cina a 11 chilometri per ora.", "verso la Cina a 11 chilometri/ora.")
        self.assertSame("it", "del formato 35 mm è di 24 mm", "del formato trentacinque mm è di ventiquattro millimetri")
        self.assertSame("ru", "скоростью более 160 км/ч семь раз", "скоростью более 160 километров в час семь раз",
                        "скоростью более 160 км ч семь раз")
        self.assertSame("ru", "площадь в 3850 кв. км, из", "площадь в 3850 квадратных километров, из")
        self.assertSame("en", "winds of 40 mph (64 kph).", "winds of 40 miles per hour, 64 kilometers per hour.",
                        "winds of forty mph, sixty-four kph.", "winds of 40 miles an hour, 64 kph.")
        self.assertSame("en", "about 645 miles (1040 km) west", "about 645 miles, 1040 kilometers west")

    def test_degrees(self):
        self.assertSame("de", "Temperaturen über +30 °C üblich", "Temperaturen über plus 30 Grad Celsius üblich",
                        "Temperaturen über 30 Grad Celsius üblich")
        self.assertSame("es", "superiores a +30° C durante", "superiores a más treinta grados Celsius durante")
        self.assertSame("fr", "dépassant 90 °F.", "dépassant 90 degrés Fahrenheit.")
        self.assertSame("it", "oltre i 30°C sono", "oltre i trenta gradi Celsius sono")
        # "degrees C" is an accepted English reading; elsewhere a letter reading of the unit is an error.
        self.assertSame("en", "temperatures above +30°C are common", "temperatures above plus thirty degrees Celsius are common",
                        "temperatures above plus thirty degrees C are common", "temperatures above 30 °C are common")
        self.assertDiffer("de", "Temperaturen über +30 °C üblich", "Temperaturen über 30 Grad C üblich")

    def test_abbreviations(self):
        self.assertSame("de", "für Ihr Reiseziel (z.B. Visa)", "für Ihr Reiseziel, zum Beispiel Visa")
        self.assertSame("de", "des Landwirts bzw. der Landwirtin", "des Landwirts beziehungsweise der Landwirtin")
        self.assertSame("de", "Six Flags St. Louis", "Six Flags Sankt Louis", "Six Flags Saint Louis", "Six Flags St Louis")
        self.assertSame("es", "en los EE. UU. con", "en los Estados Unidos con", "en los EE.UU. con", "en los EEUU con")
        self.assertSame("es", "en 323 a. C.", "en 323 antes de Cristo", "en 323 a.C.")
        self.assertSame("es", "el «cosmonauta n.º 11»", "el cosmonauta número once", "el cosmonauta No. 11")
        self.assertSame("fr", "le Dr. Damadian", "le docteur Damadian", "le Dr Damadian")
        self.assertSame("fr", "Selon M. Pittman", "Selon monsieur Pittman")
        self.assertSame("fr", "5 000 ans av. J.-C.", "5000 ans avant Jésus-Christ", "5 000 ans av. J.C.", "5 000 ans avant J-C.")
        self.assertSame("it", "singolare (ad es. nei Paesi Bassi", "singolare. Ad esempio nei Paesi Bassi", "singolare. Ad es, nei Paesi Bassi")
        self.assertSame("it", "il 21 luglio del 356 a.C. il tempio", "il 21 luglio del 356 avanti Cristo il tempio")
        self.assertSame("ru", "с помощью фагов, т. е. вирусов.", "с помощью фагов, то есть вирусов.")

    def test_wrong_readings_still_differ(self):
        self.assertDiffer("es", "un camino de 1600 km desde", "un camino de 1600 ka eme desde", "un camino de 1600 desde")
        self.assertDiffer("de", "um etwa 12% gestiegen", "um etwa 12 gestiegen")
        self.assertDiffer("en", "winds of 40 mph (64 kph).", "winds of 40 m p h, 64 k p h.")
        self.assertDiffer("it", "singolare (ad es. nei Paesi Bassi", "singolare. Adesso nei Paesi Bassi")
        self.assertDiffer("ru", "скоростью более 160 км/ч семь раз", "скоростью более 160 семь раз")

    def test_left_alone(self):
        # Not a unit: an ordinal, a longitude, a short word after a number written in words, a word, a sentence end.
        self.assertNotIn("gradi", ml_normalizer("il 1° e il 3° reggimento", lang="it"))
        self.assertNotIn("gradi", ml_normalizer("a est del 35°W.", lang="it"))
        self.assertEqual(ml_normalizer("con cien mi amigo", lang="es"), "con cien mi amigo")
        self.assertEqual(ml_normalizer("die Stadt Stuttgart", lang="de"), "die stadt stuttgart")
        self.assertEqual(ml_normalizer("dijo que no. Luego", lang="es"), "dijo que no luego")
        # English has no one-letter units: "50m", "2h", "30s" (often minutes, hours, seconds) stay as written.
        self.assertEqual(normalizer("the job took 50m, then 2h, then 30s"), "the job took 50 m then 2h then 30s")
        # Languages without entries are unchanged: "%" is still stripped there.
        self.assertEqual(ml_normalizer("12%", lang="fi"), "kaksitoista")


if __name__ == "__main__":
    unittest.main()
