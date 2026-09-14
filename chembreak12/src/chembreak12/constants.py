from __future__ import annotations

PROTOCOL_ID = "CB12_PARTITION_V1"
SEED = 12026
PRIMARY_SPLITS = ("Train", "Test1", "Test2", "Test3", "Test4")
ALL_SPLITS = PRIMARY_SPLITS + ("Reserve",)
PRIMARY_SPLIT_SIZES = {
    "Train": 241,
    "Test1": 50,
    "Test2": 50,
    "Test3": 50,
    "Test4": 50,
}
REQUIRED_COLUMNS = {
    "assignment_id",
    "is_reserve",
    "matrix_id",
    "hc_id",
    "hc_category",
    "hd_id",
    "hazard_domain",
    "ot_id",
    "output_type",
    "benchmark_prompt",
    "main_goal",
}
BALANCE_WEIGHTS = {"hc_id": 2.0, "hd_id": 2.0, "ot_id": 3.0}
