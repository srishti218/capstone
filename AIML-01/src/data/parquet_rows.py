import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def select_rows(path, positions):
    """Read selected physical rows in order while bounding the Arrow working set."""
    positions = np.sort(np.asarray(positions, dtype=np.int64))
    if not len(positions):
        return pd.DataFrame()
    selected = []
    offset = start = 0
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=8192):
            end = np.searchsorted(positions, offset + batch.num_rows)
            if end > start:
                selected.append(batch.take(positions[start:end] - offset).to_pandas())
            if end == len(positions):
                break
            offset += batch.num_rows
            start = end
    return pd.concat(selected, ignore_index=True)
