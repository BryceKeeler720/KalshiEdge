"""Tests for forecast parsing."""

from kalshiedge.prompts import parse_forecast


class TestParseForecast:
    def test_full_parse(self):
        text = (
            "PROBABILITY: 72%\n"
            "CONFIDENCE_RANGE: [65%, 80%]\n"
            "KEY_FACTORS:\n"
            "1. Strong economic indicators\n"
            "2. Historical precedent\n"
            "3. Recent Fed commentary\n"
            "REASONING: Based on current economic data and Fed signaling, "
            "a rate cut is likely but not certain."
        )
        result = parse_forecast(text)
        assert result["probability"] == 0.72
        assert result["confidence_low"] == 0.65
        assert result["confidence_high"] == 0.80
        assert "economic data" in result["reasoning"]

    def test_decimal_probability(self):
        text = "PROBABILITY: 55.5%\nCONFIDENCE_RANGE: [50%, 60%]\nREASONING: Close call."
        result = parse_forecast(text)
        assert result["probability"] == 0.555

    def test_missing_fields(self):
        text = "Some random text without the expected format."
        result = parse_forecast(text)
        assert result["probability"] is None
        assert result["confidence_low"] is None
        assert result["reasoning"] is None

    def test_partial_parse(self):
        text = "PROBABILITY: 80%\nSome other stuff."
        result = parse_forecast(text)
        assert result["probability"] == 0.80
        assert result["confidence_low"] is None
