# mining-association-rules: Package to generate association rules from your data source

## Installation from sources
### Step 1: Clone the repository
```
git clone git@github.com:michalpiekarski777/mining-association-rules.git
```
### Step 2: Install the library
In the root directory execute:
```
pip install -e .
```

## Usage Examples

### CLI Usage
See [examples/cli_usage.py](examples/cli_usage.py) for running the algorithm from the command line with a config file or CLI arguments.

### Programmatic Usage
See [examples/programmatic_usage.py](examples/programmatic_usage.py) for an example of how to use the package as a library with native Python data structures.

## Available Interest Measures

Measures are specified by their key with a threshold value, either in a config JSON or as CLI arguments (e.g., `confidence=0.5`).

### Itemset Measures (used to filter frequent itemsets)
| Key | Name | Description |
|-----|------|-------------|
| `support` | Support | Proportion of transactions containing the itemset |

### Rule Measures (used to filter association rules)
| Key | Name | Description |
|-----|------|-------------|
| `anti_support` | Anti-Support | Proportion of transactions with antecedent but without consequent |
| `confidence` | Confidence | Conditional probability of consequent given antecedent |
| `conviction` | Conviction | Implication strength: (1 - support(C)) / (1 - confidence) |
| `dependency_factor` | Dependency Factor | Normalized dependency: (confidence - support(C)) / (confidence + support(C)) |
| `gain_function` | Gain Function | Rule gain relative to baseline (gain=0.8) |
| `hyperconfidence` | Hyperconfidence | Statistical confidence using hypergeometric distribution |
| `hyperlift` | Hyperlift | Statistical lift using hypergeometric distribution (quantile=0.99) |
| `lift` | Lift | Observed vs expected frequency ratio under independence (>1 = positive association) |
| `rule_interest_function` | Rule Interest Function | Rule strength as deviation from independence |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
