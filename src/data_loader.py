"""
Universal Data Loader Module
Supports NASA POWER CSV, General CSV, Excel (.xlsx, .xls), TSV, and TXT meteorological files.
"""
import os
import io
import re
import pandas as pd
import numpy as np


def load_any_dataset(file_source, filename=""):
    """
    Universally loads and normalizes tabular meteorological datasets.
    
    Parameters
    ----------
    file_source : str or bytes or file-like object
        Path to file or in-memory file bytes.
    filename : str
        Original filename for extension detection.
        
    Returns
    -------
    df : pd.DataFrame
        Loaded and column-stripped dataframe with completely blank rows removed.
    metadata : dict
        Extracted metadata or domain diagnosis.
    """
    if not filename and isinstance(file_source, str):
        filename = os.path.basename(file_source)
    metadata = {'filename': filename, 'source_format': 'unknown'}
    ext = os.path.splitext(filename)[1].lower() if filename else ''
    
    # Check if file_source is bytes
    is_bytes = isinstance(file_source, (bytes, io.BytesIO))
    buffer = io.BytesIO(file_source) if isinstance(file_source, bytes) else file_source
    
    if ext in ['.xlsx', '.xls']:
        metadata['source_format'] = 'excel'
        df = pd.read_excel(buffer)
    else:
        # Handle CSV / TSV / Text
        if is_bytes:
            text = buffer.getvalue().decode('utf-8', errors='ignore') if hasattr(buffer, 'getvalue') else buffer.read().decode('utf-8', errors='ignore')
        else:
            with open(file_source, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
        # Check NASA POWER Header
        if "-BEGIN HEADER-" in text:
            metadata['source_format'] = 'nasa_power_csv'
            skip_lines = 0
            for idx, line in enumerate(text.splitlines()):
                if "-END HEADER-" in line:
                    skip_lines = idx + 1
                    break
            df = pd.read_csv(io.StringIO(text), skiprows=skip_lines)
        else:
            metadata['source_format'] = 'standard_csv'
            # Try comma first, then fallback to separator detection
            try:
                df = pd.read_csv(io.StringIO(text))
                if len(df.columns) == 1 and '	' in text[:1000]:
                    df = pd.read_csv(io.StringIO(text), sep='	')
            except Exception:
                df = pd.read_csv(io.StringIO(text), sep=None, engine='python')

    # Standardize column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Remove unnamed columns that are entirely blank
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False) | df.notna().any()]
    
    # Drop rows that are completely empty / all commas / all NaN
    initial_len = len(df)
    df = df.dropna(how='all').reset_index(drop=True)
    metadata['initial_rows'] = initial_len
    metadata['valid_rows'] = len(df)
    
    # Construct Date if YEAR and DOY present
    if 'YEAR' in df.columns and 'DOY' in df.columns and 'Date' not in df.columns:
        dates = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j', errors='coerce')
        df.insert(0, 'Date', dates)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    elif 'DATE' in df.columns:
        df['Date'] = pd.to_datetime(df['DATE'], errors='coerce')
        
    return df, metadata


def load_nasa_power_csv(filepath):
    """Legacy wrapper for NASA POWER CSV files."""
    return load_any_dataset(filepath, filename=os.path.basename(filepath))
