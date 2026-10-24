import sys
import unittest
from pathlib import Path

MODULES_DIR = Path(__file__).resolve().parents[1] / "Modules"
sys.path.insert(0, str(MODULES_DIR))

from serial_reader import SerialReader


class SerialReaderParseLineTests(unittest.TestCase):
    def test_json_signal_aliases_preserve_adc_value(self):
        for key in ("v", "ecg", "val", "value", "adc", "raw", "signal", "ADC"):
            with self.subTest(key=key):
                self.assertEqual(
                    SerialReader._parse_line(f'{{"{key}":2048,"lo":0}}'),
                    (2048.0, False),
                )

    def test_json_string_lead_off_flags_are_parsed_as_booleans(self):
        self.assertEqual(
            SerialReader._parse_line('{"value":2048,"lead_off":"0"}'),
            (2048.0, False),
        )
        self.assertEqual(
            SerialReader._parse_line('{"value":2048,"lead_off":"1"}'),
            (2048.0, True),
        )

    def test_existing_csv_and_plain_formats_remain_supported(self):
        self.assertEqual(SerialReader._parse_line("1534,0"), (1534.0, False))
        self.assertEqual(SerialReader._parse_line("1534"), (1534.0, False))

    def test_json_without_signal_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no recognized ECG value field"):
            SerialReader._parse_line('{"status":"ready"}')

    def test_non_finite_json_signal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite ECG value"):
            SerialReader._parse_line('{"adc":"NaN"}')

    def test_each_subscriber_receives_the_full_sample_stream(self):
        reader = SerialReader()
        first = reader.subscribe()
        second = reader.subscribe()

        reader._push(1200.0)
        reader._push(1300.0)

        self.assertEqual(reader.pop_new_samples(first), [1200.0, 1300.0])
        self.assertEqual(reader.pop_new_samples(second), [1200.0, 1300.0])
        reader.unsubscribe(first)
        reader.unsubscribe(second)


if __name__ == "__main__":
    unittest.main()
