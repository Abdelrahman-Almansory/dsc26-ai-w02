# Framing: Rawaj Pay mule detection

## Decision

For every instant transfer, Rawaj Pay must decide in the moment whether to hold it for manual review or let it go through. Dina's five analysts can only clear a limited queue each day, so the decision is really a threshold on a risk score: everything at or above it is held. A false positive delays a legitimate customer and uses up analyst time. A false negative lets a mule account move stolen money out, which costs far more, so the line should be set by cost and queue capacity, not by a default 0.5.

## Metric

Only about 2.8% of labelled transfers are mules, so accuracy is misleading: flagging nothing would score about 97% and catch no fraud. I would track precision and recall at the chosen threshold, precision@k for the number of cases the analysts can actually review per day, and an expected cost that weights a missed mule much higher than a needless review. The threshold comes from minimising that cost with `best_threshold`, or from `threshold_for_precision` if Dina promises analysts that a minimum share of flagged cases are real.

## Baseline

The baseline is a simple rule: hold a transfer when `receiver_in_degree_24h`, the number of senders paying the same receiver in the last 24 hours, is at least 4. In the sample extract, labelled mules have a mean in-degree of about 4.2 against about 2.0 for other transfers, which fits the idea that mule accounts collect money from many senders. The sample has only 53 labelled mules, so this is a rough signal, not a precise cut-off. A model has to beat this rule on cost and on precision@k, measured on transfers that come later in time than the data used to build it. The audit also comes first: `case_closed_at` and `reversal_amount` are only known after a case is investigated or money is reversed, so they would make any baseline or model look near-perfect and fail in production. I would drop both before scoring anything.

## Classical ML or LLM

I would start with classical ML, logistic regression first and then gradient-boosted trees, because the data is structured and tabular (amounts, receiver age, in-degree, channel). These models are fast, cheap, easy to explain to analysts and auditors, and simple to re-threshold when costs change. An LLM could later help by summarising a flagged case for the analyst, but it should not produce the risk score itself, since it is slower, costlier and harder to audit for a decision with money at stake.
