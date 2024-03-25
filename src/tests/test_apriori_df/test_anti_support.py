from src.mining_association_rules.apriori_df.apriori.apriori import DataFrameRuleGenerator


def test_anti_support(example_dataset):
    rule_gen = DataFrameRuleGenerator(source="book_example.csv")

    assert rule_gen.anti_support({"cola"}, {"orzeszki"}, example_dataset) == 0.2
    assert rule_gen.anti_support({"orzeszki", "pieluszki"}, {"piwo"}, example_dataset) == 0.0
    assert (
        rule_gen.anti_support({"cola", "orzeszki"}, {"pieluszki", "piwo"}, example_dataset) == 0.4
    )
