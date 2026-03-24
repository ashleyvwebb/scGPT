

QUERY_LIST_PATH = "./query_list.txt"
with open(QUERY_LIST_PATH) as f:
    QUERY_LIST = [line.rstrip('\n') for line in f]

VERSION = "2025-11-08"

VALUE_FILTER = {}

# normal tissue queries
for query in QUERY_LIST:
    if query.endswith("-cancer"):
        continue
    VALUE_FILTER[query] = f"suspension_type != 'na' and disease == 'normal' and tissue_general == '{query}'"

# cancer queries
cancer_queries = [q for q in QUERY_LIST if q.endswith("-cancer")]
if cancer_queries:
    CANCER_LIST_PATH = "./cancer_list.txt"
    with open(CANCER_LIST_PATH) as f:
        CANCER_LIST = [line.rstrip('\n') for line in f]

    if len(CANCER_LIST) == 0:
        raise ValueError("cancer_list.txt is empty but cancer queries were requested.")
    
    cancer_condition = " or ".join(
        f"(disease == '{disease}')" for disease in CANCER_LIST
    )

    for query in cancer_queries:
        tissue = query.removesuffix("-cancer")
        VALUE_FILTER[query] = f"suspension_type != 'na' and tissue_general == '{tissue}' and ({cancer_condition})"


if __name__ == "__main__":
    print("QUERY_LIST:")
    print(QUERY_LIST)
    print()
    print("VALUE_FILTER:")
    for k,v in VALUE_FILTER.items():
        print(f"{k}: {v}")
