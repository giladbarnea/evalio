from evalio.evaluation import ANSWER_COMPARISON_PROMPT_TEMPLATE, are_texts


class TestAreTexts:
    def test_are_texts_logically_consistent_returns_false_for_earth_sun(self):
        text_1 = "The Earth revolves around the Sun."
        text_2 = "The Sun revolves around the Earth."
        assert not are_texts(text_1, text_2, criterion="logically consistent")

    def test_are_texts_logically_consistent_returns_false_for_birds_penguins(self):
        text_1 = "All birds can fly."
        text_2 = "Penguins are birds."
        assert not are_texts(text_1, text_2, criterion="logically consistent")

    def test_are_texts_contradictory_returns_true(self):
        text_1 = "Cats are mammals."
        text_2 = "Cats are reptiles."
        assert are_texts(text_1, text_2, criterion="contradictory")

    def test_are_texts_complementary(self):
        text_1 = "Water is a liquid at room temperature."
        text_2 = "Ice is a solid form of water."
        assert not are_texts(text_1, text_2, criterion="complementary")

    def test_are_texts_overlapping(self):
        text_1 = "The Great Wall of China is one of the Seven Wonders of the World."
        text_2 = "The Seven Wonders of the World include the Great Wall of China."
        assert are_texts(text_1, text_2, criterion="overlapping")

    def test_are_texts_mutually_exclusive(self):
        text_1 = "The Earth revolves around the Sun."
        text_2 = "The Sun revolves around the Earth."
        assert are_texts(text_1, text_2, criterion="mutually_exclusive")
