BID_PRICES   = [f"p{i}" for i in range(6)]
ASK_PRICES   = [f"p{i}" for i in range(6, 12)]
BID_VOLUMES  = [f"v{i}" for i in range(6)]
ASK_VOLUMES  = [f"v{i}" for i in range(6, 12)]
TRADE_PRICES  = [f"dp{i}" for i in range(4)]
TRADE_VOLUMES = [f"dv{i}" for i in range(4)]
TARGETS = ["t0", "t1"]
META    = ["seq_ix", "step_in_seq", "need_prediction"]
ALL_FEATURES = (
    BID_PRICES + ASK_PRICES + BID_VOLUMES + ASK_VOLUMES
    + TRADE_PRICES + TRADE_VOLUMES
)
