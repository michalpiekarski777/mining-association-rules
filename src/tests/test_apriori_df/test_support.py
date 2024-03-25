from src.mining_association_rules.apriori_df.apriori.apriori import DataFrameRuleGenerator


def test_support(example_dataset):
    rule_gen = DataFrameRuleGenerator(source="book_example.csv")

    assert rule_gen.support({"cola"}, example_dataset) == 0.6
    assert rule_gen.support({"cola", "orzeszki"}, example_dataset) == 0.4
    assert rule_gen.support({"orzeszki", "pieluszki", "piwo"}, example_dataset) == 0.4
    assert rule_gen.support({"cola", "orzeszki", "pieluszki", "piwo"}, example_dataset) == 0.0
