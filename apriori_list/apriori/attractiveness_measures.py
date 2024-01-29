from common.utils.exceptions import EmptyTransactionBaseException


def support(itemset: set, transactions: list[set]) -> float:
    if not transactions:
        raise EmptyTransactionBaseException
    supported = [
        itemset for transaction in transactions if itemset.issubset(transaction)
    ]
    return len(supported) / len(transactions)
