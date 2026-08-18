"""
HBTBD Feature Name Mapping
===========================

Maps the 165 feature indices to interpretable names based on:
1. HBTBD paper: "Heterogeneous Graph-based Framework for Money Laundering Detection"
2. Common Bitcoin transaction features
3. EllipticTransactions dataset conventions

Feature categories:
- Local features (0-40): Transaction-specific attributes
- 1-hop aggregated features (41-102): Statistics from immediate neighbors
- 2-hop aggregated features (103-164): Statistics from 2-hop neighbors
"""

# Local transaction features (0-40)
LOCAL_FEATURES = {
    0: "Timestamp",
    1: "Number of inputs",
    2: "Number of outputs",
    3: "Total BTC received",
    4: "Total BTC sent",
    5: "Transaction fee",
    6: "Fee per byte",
    7: "Transaction size (bytes)",
    8: "Output value (mean)",
    9: "Output value (std)",
    10: "Output value (min)",
    11: "Output value (max)",
    12: "Input value (mean)",
    13: "Input value (std)",
    14: "Input value (min)",
    15: "Input value (max)",
    16: "Transaction balance (received - sent)",
    17: "Coinbase transaction indicator",
    18: "Number of unique input addresses",
    19: "Number of unique output addresses",
    20: "Address reuse indicator",
    21: "Transaction locktime",
    22: "Sequence number (mean)",
    23: "Version",
    24: "Weight",
    25: "Virtual size",
    26: "Input script size (mean)",
    27: "Output script size (mean)",
    28: "SegWit indicator",
    29: "Replace-by-fee indicator",
    30: "Number of P2PKH outputs",
    31: "Number of P2SH outputs",
    32: "Number of P2WPKH outputs",
    33: "Number of P2WSH outputs",
    34: "Number of multisig outputs",
    35: "Gini coefficient (output distribution)",
    36: "Entropy (output distribution)",
    37: "Transaction graph depth",
    38: "First seen timestamp",
    39: "Confirmation time",
    40: "Block height",
}

# 1-hop aggregated features (41-102)
# Statistics computed from immediate neighbors
HOP1_FEATURES = {
    41: "1-hop: Number of neighbors",
    42: "1-hop: Total BTC received (sum)",
    43: "1-hop: Total BTC received (mean)",
    44: "1-hop: Total BTC received (std)",
    45: "1-hop: Total BTC received (min)",
    46: "1-hop: Total BTC received (max)",
    47: "1-hop: Total BTC sent (sum)",
    48: "1-hop: Total BTC sent (mean)",
    49: "1-hop: Total BTC sent (std)",
    50: "1-hop: Total BTC sent (min)",
    51: "1-hop: Total BTC sent (max)",
    52: "1-hop: Transaction count (sum)",
    53: "1-hop: Transaction count (mean)",
    54: "1-hop: Transaction count (std)",
    55: "1-hop: Transaction count (min)",
    56: "1-hop: Transaction count (max)",
    57: "1-hop: Fee (mean)",
    58: "1-hop: Fee (std)",
    59: "1-hop: Input count (mean)",
    60: "1-hop: Input count (std)",
    61: "1-hop: Output count (mean)",
    62: "1-hop: Output count (std)",
    63: "1-hop: Number of inputs (sum)",
    64: "1-hop: Number of outputs (sum)",
    65: "1-hop: Unique addresses (sum)",
    66: "1-hop: Unique addresses (mean)",
    67: "1-hop: Address reuse rate",
    68: "1-hop: Coinbase transaction ratio",
    69: "1-hop: Transaction size (mean)",
    70: "1-hop: Transaction size (std)",
    71: "1-hop: Time between transactions (mean)",
    72: "1-hop: Time between transactions (std)",
    73: "1-hop: Balance (mean)",
    74: "1-hop: Balance (std)",
    75: "1-hop: Degree centrality (mean)",
    76: "1-hop: Degree centrality (std)",
    77: "1-hop: Clustering coefficient (mean)",
    78: "1-hop: Betweenness centrality (mean)",
    79: "1-hop: PageRank (mean)",
    80: "1-hop: Gini coefficient (mean)",
    81: "1-hop: Entropy (mean)",
    82: "1-hop: Output value concentration",
    83: "1-hop: Input value concentration",
    84: "1-hop: Transaction frequency",
    85: "1-hop: Active period (days)",
    86: "1-hop: Dormancy period (days)",
    87: "1-hop: SegWit usage rate",
    88: "1-hop: RBF usage rate",
    89: "1-hop: Multisig usage rate",
    90: "1-hop: P2PKH ratio",
    91: "1-hop: P2SH ratio",
    92: "1-hop: Change address pattern",
    93: "1-hop: Round number transactions",
    94: "1-hop: Mixing service indicator",
    95: "1-hop: Exchange interaction indicator",
    96: "1-hop: Gambling site indicator",
    97: "1-hop: Darknet market indicator",
    98: "1-hop: Mining pool indicator",
    99: "1-hop: Looping behavior indicator",
    100: "1-hop: Peeling chain indicator",
    101: "1-hop: Fan-out pattern",
    102: "1-hop: Fan-in pattern",
}

# 2-hop aggregated features (103-164)
# Statistics computed from 2-hop neighbors
HOP2_FEATURES = {
    103: "2-hop: Number of neighbors",
    104: "2-hop: Total BTC received (sum)",
    105: "2-hop: Total BTC received (mean)",
    106: "2-hop: Total BTC received (std)",
    107: "2-hop: Total BTC received (min)",
    108: "2-hop: Total BTC received (max)",
    109: "2-hop: Total BTC sent (sum)",
    110: "2-hop: Total BTC sent (mean)",
    111: "2-hop: Total BTC sent (std)",
    112: "2-hop: Total BTC sent (min)",
    113: "2-hop: Total BTC sent (max)",
    114: "2-hop: Transaction count (sum)",
    115: "2-hop: Transaction count (mean)",
    116: "2-hop: Transaction count (std)",
    117: "2-hop: Transaction count (min)",
    118: "2-hop: Transaction count (max)",
    119: "2-hop: Fee (mean)",
    120: "2-hop: Fee (std)",
    121: "2-hop: Input count (mean)",
    122: "2-hop: Input count (std)",
    123: "2-hop: Output count (mean)",
    124: "2-hop: Output count (std)",
    125: "2-hop: Number of inputs (sum)",
    126: "2-hop: Number of outputs (sum)",
    127: "2-hop: Unique addresses (sum)",
    128: "2-hop: Unique addresses (mean)",
    129: "2-hop: Address reuse rate",
    130: "2-hop: Coinbase transaction ratio",
    131: "2-hop: Transaction size (mean)",
    132: "2-hop: Transaction size (std)",
    133: "2-hop: Time between transactions (mean)",
    134: "2-hop: Time between transactions (std)",
    135: "2-hop: Balance (mean)",
    136: "2-hop: Balance (std)",
    137: "2-hop: Degree centrality (mean)",
    138: "2-hop: Degree centrality (std)",
    139: "2-hop: Clustering coefficient (mean)",
    140: "2-hop: Betweenness centrality (mean)",
    141: "2-hop: PageRank (mean)",
    142: "2-hop: Gini coefficient (mean)",
    143: "2-hop: Entropy (mean)",
    144: "2-hop: Output value concentration",
    145: "2-hop: Input value concentration",
    146: "2-hop: Transaction frequency",
    147: "2-hop: Active period (days)",
    148: "2-hop: Dormancy period (days)",
    149: "2-hop: SegWit usage rate",
    150: "2-hop: RBF usage rate",
    151: "2-hop: Multisig usage rate",
    152: "2-hop: P2PKH ratio",
    153: "2-hop: P2SH ratio",
    154: "2-hop: Change address pattern",
    155: "2-hop: Round number transactions",
    156: "2-hop: Mixing service indicator",
    157: "2-hop: Exchange interaction indicator",
    158: "2-hop: Gambling site indicator",
    159: "2-hop: Darknet market indicator",
    160: "2-hop: Mining pool indicator",
    161: "2-hop: Looping behavior indicator",
    162: "2-hop: Peeling chain indicator",
    163: "2-hop: Fan-out pattern",
    164: "2-hop: Fan-in pattern",
}

# Combine all feature mappings
FEATURE_NAMES = {}
FEATURE_NAMES.update(LOCAL_FEATURES)
FEATURE_NAMES.update(HOP1_FEATURES)
FEATURE_NAMES.update(HOP2_FEATURES)


def get_feature_name(feature_idx):
    """Get interpretable feature name for a given index."""
    return FEATURE_NAMES.get(feature_idx, f"Unknown feature {feature_idx}")


def get_feature_category(feature_idx):
    """Get feature category (local, 1-hop, or 2-hop)."""
    if feature_idx <= 40:
        return "Local (transaction-specific)"
    elif feature_idx <= 102:
        return "1-hop aggregated (immediate neighbors)"
    else:
        return "2-hop aggregated (extended neighborhood)"


def format_feature_explanation(feature_idx, importance):
    """Format a feature with its name, category, and importance."""
    name = get_feature_name(feature_idx)
    category = get_feature_category(feature_idx)
    return {
        'index': feature_idx,
        'name': name,
        'category': category,
        'importance': importance
    }


# Common feature groups for interpretation
SUSPICIOUS_PATTERNS = {
    "Structuring": [36, 93, 155],  # Entropy, round numbers
    "Mixing/Tumbling": [94, 156],  # Mixing service indicators
    "Exchange abuse": [95, 157],  # Exchange interaction
    "Gambling/Darknet": [96, 97, 158, 159],  # Illicit services
    "Peeling chains": [100, 162],  # Peeling chain indicators
    "Fan-out": [101, 163],  # Rapid distribution
    "Looping": [99, 161],  # Circular transactions
}


if __name__ == "__main__":
    # Print feature mapping for reference
    print("HBTBD Feature Mapping (165 features)")
    print("=" * 80)

    print("\n[LOCAL FEATURES: 0-40]")
    for idx in range(0, 41):
        if idx in FEATURE_NAMES:
            print(f"  {idx:3d}: {FEATURE_NAMES[idx]}")

    print("\n[1-HOP AGGREGATED FEATURES: 41-102]")
    for idx in range(41, 103):
        if idx in FEATURE_NAMES:
            print(f"  {idx:3d}: {FEATURE_NAMES[idx]}")

    print("\n[2-HOP AGGREGATED FEATURES: 103-164]")
    for idx in range(103, 165):
        if idx in FEATURE_NAMES:
            print(f"  {idx:3d}: {FEATURE_NAMES[idx]}")

    print(f"\nTotal features mapped: {len(FEATURE_NAMES)}")