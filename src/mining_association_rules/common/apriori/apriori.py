import datetime
import json
import time
from abc import ABCMeta
from abc import abstractmethod
from itertools import chain
from itertools import combinations
from pathlib import Path

from config import ROOT_DIR
from src.mining_association_rules.apriori_df.interest_measures import *  # noqa: F403
from src.mining_association_rules.apriori_df.interest_measures.base import Measure
from src.mining_association_rules.common.utils.encoders import JSONEncoder
from src.mining_association_rules.common.utils.loggers import Logger
from src.mining_association_rules.common.utils.typed_dicts import AssociationRule


class RuleGenerator(metaclass=ABCMeta):
    total_duration: float = 0.0

    def __init__(
        self,
        runner: str,
        itemset_measures: dict[type[Measure], float],
        rule_measures: dict[type[Measure], float],
        logger_class: type[Logger] = Logger,
        *,
        verbose: bool = False,
    ):
        self.start = time.perf_counter()
        self.itemset_measures = {measure(): threshold for measure, threshold in itemset_measures.items()}
        self.rule_measures = {measure(): threshold for measure, threshold in rule_measures.items()}
        self.verbose = verbose
        self._runner = runner
        self._rules: list[AssociationRule] = []
        self._logger = logger_class(name="Rules")

    @abstractmethod
    def generate_strong_association_rules(self, *args, **kwargs) -> list[AssociationRule]:
        raise NotImplementedError

    def _apriori_gen(self, itemsets: list[frozenset[str]]) -> list[frozenset[str]]:
        """
        :param itemsets: list of k-element frequent itemsets
        :return: list of k+1-element candidates for the frequent itemsets
        """
        set_length = len(itemsets[0])
        sorted_itemsets = [sorted(itemset) for itemset in itemsets]

        return [
            itemset.union(itemsets[j])
            for index, (itemset, sorted_itemset) in enumerate(zip(itemsets, sorted_itemsets, strict=False))
            for j in range(index + 1, len(itemsets))
            if sorted_itemset[: set_length - 1] == sorted_itemsets[j][: set_length - 1]
        ]

    def _generate_subset_combinations(self, elements: frozenset[str]) -> list[tuple]:
        """
        :param elements: set of elements
        :return: list of not empty subsets of set elements excluding set of length len(elements)
        """
        return list(chain.from_iterable(combinations(elements, size) for size in range(1, len(elements))))

    def _dump_rules_to_json(self):
        path = Path(ROOT_DIR) / "outputs"
        path.mkdir(exist_ok=True)
        filename = datetime.datetime.now(tz=datetime.UTC).astimezone().strftime("%Y%m%d_%H%M_%s") + ".json"

        with Path.open(Path(path) / filename, "w") as f:
            json.dump(self._rules, f, cls=JSONEncoder, indent=4)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log_info = {"runner": self._runner, "duration": self.total_duration}
        self._logger.info("Rules generated using %(runner)s database in %(duration)s seconds", log_info)
        if not self.verbose:
            return
        for itemset_measure in self.itemset_measures:
            itemset_measure_name = type(itemset_measure).__name__
            itemset_measure_count = itemset_measure.calculations_count
            itemset_measure_time = itemset_measure.calculations_time
            log = f"Calculating {itemset_measure_name} was done {itemset_measure_count} and took {itemset_measure_time}"
            self._logger.info(log)

        for rule_measure in self.rule_measures:
            rule_measure_name = type(rule_measure).__name__
            rule_measure_count = rule_measure.calculations_count
            rule_measure_time = rule_measure.calculations_time
            log = f"Calculating {rule_measure_name} was done {rule_measure_count} and took {rule_measure_time}"
            self._logger.info(log)
        self._logger.info("Found %(total_rules)s rules", {"total_rules": len(self._rules)})

        for rule in self._rules:
            rule_members = f"Association rule {set(rule['antecedent'])} -> {set(rule['consequent'])},"
            itemset_measure_log = f"{rule['itemset_measure']['name']}: {rule['itemset_measure']['value']}"
            rule_measure_log = ""
            for rule_measure in rule["rule_measures"]:
                rule_measure_log += f"{rule_measure['name']}: {round(rule_measure['value'], 3)}, "

            msg = rule_members + itemset_measure_log + "," + rule_measure_log
            self._logger.info(msg)

        self._dump_rules_to_json()
