"""
Data export functionality for AHGD Dashboard.

This module provides utilities for exporting data in various formats
(CSV, Excel, Parquet) with proper formatting and compression.
"""

import streamlit as st
import polars as pl
from pathlib import Path
from typing import Optional, List
import io
from datetime import datetime


def export_to_csv(
    data: pl.DataFrame,
    filename: Optional[str] = None,
    compression: bool = False,
) -> bytes:
    """
    Export data to CSV format.

    Args:
        data: Polars DataFrame to export
        filename: Output filename (auto-generated if None)
        compression: Whether to compress output

    Returns:
        Bytes of CSV data
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ahgd_export_{timestamp}.csv"

    # Convert to CSV
    csv_data = data.write_csv()

    return csv_data.encode("utf-8")


def export_to_excel(
    data: pl.DataFrame,
    filename: Optional[str] = None,
    sheet_name: str = "Data",
) -> bytes:
    """
    Export data to Excel format.

    Args:
        data: Polars DataFrame to export
        filename: Output filename (auto-generated if None)
        sheet_name: Name of the Excel sheet

    Returns:
        Bytes of Excel data
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ahgd_export_{timestamp}.xlsx"

    # Convert to pandas for Excel export
    pd_data = data.to_pandas()

    # Write to bytes buffer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd_data.to_excel(writer, sheet_name=sheet_name, index=False)

    return output.getvalue()


def export_to_parquet(
    data: pl.DataFrame,
    filename: Optional[str] = None,
    compression: str = "snappy",
) -> bytes:
    """
    Export data to Parquet format.

    Args:
        data: Polars DataFrame to export
        filename: Output filename (auto-generated if None)
        compression: Compression algorithm ('snappy', 'gzip', 'brotli', 'lz4', 'zstd')

    Returns:
        Bytes of Parquet data
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ahgd_export_{timestamp}.parquet"

    # Write to bytes buffer
    output = io.BytesIO()
    data.write_parquet(output, compression=compression)

    return output.getvalue()


def create_export_button(
    data: pl.DataFrame,
    export_format: str = "CSV",
    filename: Optional[str] = None,
    label: Optional[str] = None,
    help_text: Optional[str] = None,
    key: Optional[str] = None,
) -> bool:
    """
    Create a download button for data export.

    Args:
        data: Polars DataFrame to export
        export_format: Export format ('CSV', 'Excel', 'Parquet')
        filename: Output filename
        label: Button label
        help_text: Help text for button
        key: Unique widget key

    Returns:
        Boolean indicating if download was clicked
    """
    if label is None:
        label = f"📥 Download {export_format}"

    if help_text is None:
        help_text = f"Download data as {export_format}"

    # Generate export data
    if export_format.upper() == "CSV":
        export_data = export_to_csv(data, filename)
        mime_type = "text/csv"
        default_filename = filename or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    elif export_format.upper() == "EXCEL":
        export_data = export_to_excel(data, filename)
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        default_filename = filename or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    elif export_format.upper() == "PARQUET":
        export_data = export_to_parquet(data, filename)
        mime_type = "application/octet-stream"
        default_filename = filename or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"

    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    # Create download button
    return st.download_button(
        label=label,
        data=export_data,
        file_name=default_filename,
        mime=mime_type,
        help=help_text,
        key=key,
    )


def create_export_section(
    data: pl.DataFrame,
    available_formats: Optional[List[str]] = None,
    max_rows_warning: int = 100000,
) -> None:
    """
    Create a complete export section with format selection and download.

    Args:
        data: Polars DataFrame to export
        available_formats: List of formats to offer (default: all)
        max_rows_warning: Show warning if data exceeds this many rows
    """
    if available_formats is None:
        available_formats = ["CSV", "Excel", "Parquet"]

    st.subheader("📥 Export Data")

    # Show data info
    row_count = len(data)
    col_count = len(data.columns)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{row_count:,}")
    with col2:
        st.metric("Columns", f"{col_count:,}")
    with col3:
        # Estimate size
        size_mb = data.estimated_size("mb")
        st.metric("Est. Size", f"{size_mb:.2f} MB")

    # Warning for large datasets
    if row_count > max_rows_warning:
        st.warning(
            f"⚠️ Large dataset ({row_count:,} rows). "
            f"Consider using Parquet format for better performance and smaller file size."
        )

    # Format selection
    st.markdown("#### Select Export Format")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "CSV" in available_formats:
            if create_export_button(
                data=data,
                export_format="CSV",
                key="export_csv",
                help_text="Plain text format, widely compatible",
            ):
                st.success("✅ CSV export ready for download")

    with col2:
        if "Excel" in available_formats:
            # Limit Excel exports for very large datasets
            if row_count > 1000000:
                st.info("Excel format not available for datasets > 1M rows")
            else:
                if create_export_button(
                    data=data,
                    export_format="Excel",
                    key="export_excel",
                    help_text="Excel format with formatting",
                ):
                    st.success("✅ Excel export ready for download")

    with col3:
        if "Parquet" in available_formats:
            if create_export_button(
                data=data,
                export_format="Parquet",
                key="export_parquet",
                help_text="Columnar format, efficient and compressed",
            ):
                st.success("✅ Parquet export ready for download")

    # Export info
    with st.expander("ℹ️ Export Format Information"):
        st.markdown(
            """
            **CSV (Comma-Separated Values)**
            - ✅ Widely compatible with all tools
            - ✅ Human-readable
            - ❌ Larger file size
            - ❌ No data type preservation

            **Excel (XLSX)**
            - ✅ Easy to open in Excel/LibreOffice
            - ✅ Supports formatting
            - ❌ Limited to ~1M rows
            - ❌ Larger file size

            **Parquet**
            - ✅ Very efficient compression
            - ✅ Preserves data types
            - ✅ Fast to read/write
            - ❌ Requires specialized tools
            """
        )


def create_filtered_export(
    data: pl.DataFrame,
    columns: Optional[List[str]] = None,
    export_format: str = "CSV",
) -> None:
    """
    Create export with column selection.

    Args:
        data: Polars DataFrame to export
        columns: Pre-selected columns (all if None)
        export_format: Default export format
    """
    if columns is None:
        columns = data.columns

    st.subheader("🔍 Custom Export")

    # Column selection
    selected_columns = st.multiselect(
        "Select Columns to Export",
        options=data.columns,
        default=columns,
        help="Choose which columns to include in export",
    )

    if not selected_columns:
        st.warning("⚠️ Please select at least one column to export")
        return

    # Filter data
    filtered_data = data.select(selected_columns)

    # Show preview
    st.markdown("#### Preview")
    st.dataframe(filtered_data.head(10).to_pandas(), use_container_width=True)

    # Export button
    st.markdown("#### Download")

    create_export_button(
        data=filtered_data,
        export_format=export_format,
        filename=f"ahgd_custom_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
        key="custom_export",
    )


def create_multi_sheet_excel(
    data_dict: dict,
    filename: Optional[str] = None,
) -> bytes:
    """
    Export multiple DataFrames to a single Excel file with multiple sheets.

    Args:
        data_dict: Dictionary of {sheet_name: DataFrame}
        filename: Output filename

    Returns:
        Bytes of Excel data
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ahgd_export_{timestamp}.xlsx"

    # Write to bytes buffer
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in data_dict.items():
            # Convert Polars to Pandas if needed
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()

            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit

    return output.getvalue()


def create_summary_export(
    data: pl.DataFrame,
    include_stats: bool = True,
) -> bytes:
    """
    Create an Excel export with data and summary statistics.

    Args:
        data: Polars DataFrame to export
        include_stats: Whether to include summary statistics sheet

    Returns:
        Bytes of Excel data
    """
    sheets = {"Data": data}

    if include_stats:
        # Calculate summary statistics
        numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Int64, pl.Float64]]

        if numeric_cols:
            stats_data = data.select(numeric_cols).describe()
            sheets["Summary Statistics"] = stats_data

    return create_multi_sheet_excel(sheets)


# Import pandas for Excel functionality
try:
    import pandas as pd
except ImportError:
    pd = None
