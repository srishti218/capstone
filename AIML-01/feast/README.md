# Local Feast feature store

Run `python -m src.features.card_history` from the project root after preparing the IEEE-CIS data. This registers the `card_proxy` entity and `card_history` feature view, writes the Parquet snapshot, and materializes the latest values into SQLite. The generated registry, source and online database live under `feast/data/`.

The snapshot uses the first 60% of training transactions in chronological order. Feast needs an absolute event timestamp, so the job adds the snapshot cutoff in IEEE-CIS relative seconds to a fixed UTC origin. `/score` checks the original cutoff before returning values. `card1` is a dataset proxy key.
