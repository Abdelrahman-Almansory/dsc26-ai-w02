import pandas as pd
from pprint import pprint
from rawaj.audit import audit

df = pd.read_csv("data/sample_transfers.csv")
pprint(audit(df, target="is_mule", id_col="txn_id", time_col="sent_at"))

df = pd.read_csv("data/sample_transfers.csv")

print(df["receiver_in_degree_24h"].describe())
print(df.groupby("is_mule")["receiver_in_degree_24h"].describe())