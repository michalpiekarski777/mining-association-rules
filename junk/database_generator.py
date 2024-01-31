import random
import string


def generate_random_name(length) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


def generate_elements_and_transactions(
    no_of_elements: int = 100,
    no_of_transactions: int = 2000,
) -> tuple[set[str], list[set], list[dict]]:
    elements = set()

    for _ in range(no_of_elements):
        elements.add(generate_random_name(random.randint(4, 8)))

    transactions: list[set] = []
    to_df_transactions: list[dict] = []

    for _ in range(no_of_transactions):
        sample_elements = random.sample(sorted(elements), 40)
        transactions.append(set(random.sample(sorted(elements), 40)))
        to_df_transactions.append({element: 1 for element in sample_elements})

    return elements, transactions, to_df_transactions
