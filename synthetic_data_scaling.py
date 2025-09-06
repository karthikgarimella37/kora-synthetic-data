import os
from io import BytesIO, StringIO
from typing import Dict, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


def scale_csv_bytes(csv_bytes: bytes, dbtwin_api_key: str, rows: int = 1000, algo: str = "flagship") -> Tuple[bytes, Dict[str, str]]:
    """Scale a CSV via DBTwin API and return generated CSV bytes and response headers."""
    if not dbtwin_api_key:
        raise ValueError("DBTWIN_API_KEY is required.")

    file_obj = BytesIO(csv_bytes)
    file_obj.seek(0)
    files = {"file": ("synthetic_data.csv", file_obj)}
    headers = {"rows": str(rows), "algo": algo, "api-key": dbtwin_api_key}

    url = "https://api.dbtwin.com"
    try:
        # Optional health check
        requests.get(url + "/health", timeout=20)
    except Exception:
        # Continue even if health check fails; POST may still succeed
        pass

    resp = requests.post(url + "/generate", headers=headers, files=files, timeout=120)
    if resp.status_code == 200:
        return resp.content, resp.headers
    else:
        # Propagate server message for easier debugging
        raise RuntimeError(f"DBTwin API error {resp.status_code}: {resp.text}")


def scale_dataframe(df: pd.DataFrame, dbtwin_api_key: str, rows: int = 1000, algo: str = "flagship") -> Tuple[pd.DataFrame, bytes, Dict[str, str]]:
    """Scale a pandas DataFrame using DBTwin and return (scaled_df, csv_bytes, headers)."""
    csv_bytes = df.to_csv(index=False).encode()
    content_bytes, headers = scale_csv_bytes(csv_bytes, dbtwin_api_key, rows=rows, algo=algo)
    scaled_df = pd.read_csv(BytesIO(content_bytes))
    return scaled_df, content_bytes, headers


def scale_csv_text(csv_text: str, dbtwin_api_key: str, rows: int = 1000, algo: str = "flagship") -> Tuple[pd.DataFrame, bytes, Dict[str, str]]:
    """Scale CSV text/string using DBTwin and return (scaled_df, csv_bytes, headers)."""
    csv_text_clean = "\n".join([ln for ln in csv_text.splitlines() if ln.strip() not in ('csv', 'CSV', '```', '```csv', '```CSV')])
    df = pd.read_csv(StringIO(csv_text_clean), comment='`')
    df.rename(columns=lambda c: str(c).strip(), inplace=True)
    if 'csv' in df.columns or 'CSV' in df.columns:
        df = df.drop(columns=[c for c in ('csv', 'CSV') if c in df.columns])
    return scale_dataframe(df, dbtwin_api_key, rows=rows, algo=algo)


if __name__ == "__main__":
    # CLI-style usage for local testing
    dbtwin_api_key = os.getenv("DBTWIN_API_KEY")
    rows = int(os.getenv("DBTWIN_ROWS", "1000"))
    algo = os.getenv("DBTWIN_ALGO", "flagship")

    input_csv_path = "synthetic_data.csv"
    output_csv_path = "synthetic_generated_data.csv"

    df_input = pd.read_csv(input_csv_path, comment='`')
    scaled_df, scaled_bytes, headers = scale_dataframe(df_input, dbtwin_api_key, rows=rows, algo=algo)

    with open(output_csv_path, "wb") as f:
        f.write(scaled_bytes)

    print(scaled_df.head())
    if headers:
        print(headers.get('distribution-similarity-error'))
        print(headers.get('association-similarity'))