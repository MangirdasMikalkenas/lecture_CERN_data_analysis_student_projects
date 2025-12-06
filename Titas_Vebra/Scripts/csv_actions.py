import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def remove_rows_and_columns(df: pd.DataFrame) -> pd.DataFrame:

    # print('Removing completely empty rows and columns...')
    # Work on a copy so you don't mutate original by accident
    df_clean = df.copy()
    df_clean = df_clean.replace(r'^\s*$', np.nan, regex=True)

    # what to remove
    while True:
        choice = input("Remove (r)ows, (c)olumns, or (b)oth? [r/c/b]: ").strip().lower()
        if choice in ("r", "c", "b"):
            break
        print("Please enter r, c, or b.")

    # threshold
    while True:
        try:
            p = float(input("Enter MAX allowed % of missing values (0–100): "))
            if 0 <= p <= 100:
                break
            else:
                print("Enter a number from 0 to 100.")
        except ValueError:
            print("Invalid number.")

    threshold = p / 100.0

    # compute on ORIGINAL df_clean (before any dropping)
    row_missing_fraction = df_clean.isna().mean(axis=1)
    col_missing_fraction = df_clean.isna().mean(axis=0)

    rows_ok = row_missing_fraction <= threshold
    cols_ok = col_missing_fraction <= threshold

    if choice == "r":
        # only rows filtered
        return df_clean.loc[rows_ok, :]
    elif choice == "c":
        # only cols filtered
        return df_clean.loc[:, cols_ok]
    else:  # "b"
        # both filtered, using original percentages
        return df_clean.loc[rows_ok, cols_ok]

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    # print('Stripping whitespace from column names and cell values...')
   
    # Work on a copy for safety
    df_clean = df.copy()

    # 1) Strip whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()

    # 2) Strip whitespace inside cells (only object/string columns)
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean

def normalize_missing_values(df):

    df_clean = df.copy()

# Add more patterns as needed   
    missing_patterns = [
        r"^\s*$",     # empty / whitespace
        r"(?i)^na$",  
        r"(?i)^n/a$",
        r"(?i)^null$",
        r"(?i)^none$",
        r"^\?$",
        r"^-$",
        r"^\.$",
    ]

    for pattern in missing_patterns:
        df_clean = df_clean.replace(pattern, pd.NA, regex=True)

    return df_clean

def fix_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == "object":  
            # Replace comma-decimal only if the value looks like a number
            df_clean[col] = (
                df_clean[col]
                .str.replace(r"(?<=\d),(?=\d)", ".", regex=True)  # 1,25 -> 1.25
            )

            # Convert to numeric where possible
            df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

    return df_clean

def extract_numeric_and_unit(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()

    # 1) value + unit in the SAME CELL, e.g. "10 mA", "5,5 V"
    value_pattern = re.compile(
        r"^\s*([+-]?(?:\d+(?:[.,]\d*)?|\d*[.,]\d+))\s*([A-Za-zµ°%]+)\s*$"
    )

    # 2) unit in the HEADER, e.g. "Current, A", "Voltage (V)", "Force [N]"
    header_pattern = re.compile(
        r"""
        ^\s*
        (.+?)                               # group 1: base name (lazy)
        (?:                                 # unit part:
            [,\(\[]\s*([A-Za-zµ°%]+)\s*[\)\]]?   # case 1: "Name, A" / "Name (A)" / "Name [A]"
            |
            _([A-Za-zµ°%]+)                 # case 2: "Name_A"
        )
        \s*$
        """,
        re.VERBOSE
    )

    cols = list(df_clean.columns)
    new_columns_order = []

    for col in cols:
        series = df_clean[col]
        did_split = False

        # ---------- CASE 1: value + unit in the cell ----------
        if series.dtype == "object":
            s = series.astype(str)
            extracted = s.str.extract(value_pattern)

            if not extracted.isna().all().all():
                base_name = str(col).strip() or "col"

                # Ensure unique names (for cases like two different "Current" columns)
                num_col = f"{base_name}_value"
                unit_col = f"{base_name}_unit"
                suffix = 2
                while num_col in df_clean.columns or unit_col in df_clean.columns:
                    num_col = f"{base_name}_{suffix}_value"
                    unit_col = f"{base_name}_{suffix}_unit"
                    suffix += 1

                nums = extracted[0].str.replace(",", ".", regex=False)
                df_clean[num_col] = pd.to_numeric(nums, errors="coerce")
                df_clean[unit_col] = extracted[1]

                new_columns_order.extend([num_col, unit_col])
                did_split = True

        if did_split:
            continue

        # ---------- CASE 2: unit in the header ----------
        m = header_pattern.match(str(col))
        if m:
            base_name = m.group(1).strip() or "col"
            header_unit = (m.group(2) or m.group(3)).strip()

            num_col = f"{base_name}_value"
            unit_col = f"{base_name}_unit"
            suffix = 2
            while num_col in df_clean.columns or unit_col in df_clean.columns:
                num_col = f"{base_name}_{suffix}_value"
                unit_col = f"{base_name}_{suffix}_unit"
                suffix += 1

            data = df_clean[col]
            if data.dtype == "object":
                data = data.astype(str).str.replace(",", ".", regex=False)
            df_clean[num_col] = pd.to_numeric(data, errors="coerce")
            df_clean[unit_col] = header_unit

            new_columns_order.extend([num_col, unit_col])
        else:
            # no units -> keep original column
            new_columns_order.append(col)

    df_clean = df_clean[new_columns_order]
    return df_clean

def convert_units_to_SI(df: pd.DataFrame) -> pd.DataFrame:

    import math

    df_clean = df.copy()

    def _convert_one(value, unit):
        if pd.isna(value) or pd.isna(unit):
            return value, unit

        try:
            v = float(value)
        except (TypeError, ValueError):
            return value, unit

        u = str(unit).strip()
        u = u.replace("°", "deg")  # normalize degrees
        u = u.replace("µ", "u")    # normalize micro
        u = u.lower()

        # ---- temperature -> K ----
        if u in ("degc", "c", "celsius",):
            return v + 273.15, "K"
        if u in ("degf", "fahrenheit",):
            return (v - 32.0) * 5.0 / 9.0 + 273.15, "K"
        if u in ("degk", "k", "kelvin",):
            return v, "K"

        # ---- length -> m ----
        length_units = {
            "mm": 1e-3,
            "cm": 1e-2,
            "m": 1.0,
            "km": 1e3,
        }
        if u in length_units:
            return v * length_units[u], "m"

        # ---- mass -> kg ----
        mass_units = {
            "mg": 1e-6,
            "g": 1e-3,
            "kg": 1.0,
            "t": 1e3,
        }
        if u in mass_units:
            return v * mass_units[u], "kg"

        # ---- time -> s ----
        time_units = {
            "ms": 1e-3,
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
        }
        if u in time_units:
            return v * time_units[u], "s"

        # ---- pressure -> Pa ----
        pressure_units = {
            "pa": 1.0,
            "kpa": 1e3,
            "mpa": 1e6,
            "bar": 1e5,
            "mbar": 1e2,
            "atm": 101325.0,
            "psi": 6894.757,
        }
        if u in pressure_units:
            return v * pressure_units[u], "Pa"

        # ---- force -> N ----
        force_units = {
            "n": 1.0,
            "kn": 1e3,
        }
        if u in force_units:
            return v * force_units[u], "N"

        # ---- energy -> J ----
        energy_units = {
            "j": 1.0,
            "kj": 1e3,
        }
        if u in energy_units:
            return v * energy_units[u], "J"

        # ---- percentage -> fraction (0–1) ----
        if u in ("%", "pct"):
            return v / 100.0, "1"   # dimensionless
        
        # ---- voltage -> V ----
        voltage_units = {
            "v": 1.0,
            "mv": 1e-3,
            "kv": 1e3,
        }
        if u in voltage_units:
            return v * voltage_units[u], "V"

        # ---- frequency -> Hz ----
        freq_units = {
            "hz": 1.0,
            "khz": 1e3,
            "mhz": 1e6,
            "ghz": 1e9,
        }
        if u in freq_units:
            return v * freq_units[u], "Hz"

        # ---- electric current -> A ----
        current_units = {
            "a": 1.0,
            "ma": 1e-3,
            "ka": 1e3,
        }
        if u in current_units:
            return v * current_units[u], "A"

        # ---- resistance -> ohm ----
        resistance_units = {
            "ohm": 1.0,
            "kohm": 1e3,
            "mohm": 1e6,
            "ω": 1.0,      # lowercase omega
            "kω": 1e3,
            "mω": 1e6,
        }
        if u in resistance_units:
            return v * resistance_units[u], "ohm"

        # ---- capacitance -> F ----
        capacitance_units = {
            "f": 1.0,
            "mf": 1e-3,
            "uf": 1e-6,
            "nf": 1e-9,
            "pf": 1e-12,
        }
        if u in capacitance_units:
            return v * capacitance_units[u], "F"

        # ---- wavelength -> meters ----
        wavelength_units = {
            "m": 1.0,
            "mm": 1e-3,
            "um": 1e-6,
            "µm": 1e-6,
            "nm": 1e-9,
        }
        if u in wavelength_units:
            return v * wavelength_units[u], "m"

        # unknown unit -> leave as is
        return value, unit

    # Look for <base>_value + <base>_unit pairs
    cols = list(df_clean.columns)
    for col in cols:
        if col.endswith("_value"):
            base = col[:-6]  # remove "_value"
            unit_col = base + "_unit"
            if unit_col in df_clean.columns:
                for idx, (val, unit) in df_clean[[col, unit_col]].iterrows():
                    new_val, new_unit = _convert_one(val, unit)
                    df_clean.at[idx, col] = new_val
                    df_clean.at[idx, unit_col] = new_unit

                # make sure numeric column is numeric dtype
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean

def remove_duplicate_rows(df):
    df_clean = df.copy()

    # Detect duplicates anywhere in the file (not only next to each other)
    dup_mask = df_clean.duplicated(keep="first")

    # Remove ALL duplicate rows, keep only the first appearance
    df_clean = df_clean[~dup_mask]

    return df_clean

def move_rows_or_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Ask ONCE: move rows or columns
    2) Show layout
    3) Ask: which one to move and to where (two numbers)
    4) Do the move, show new layout
    5) Ask: move another? yes/no
    6) Repeat 3–5 until user says no, then return df
    """
    df_clean = df.copy()

    # ---- 1) choose rows or columns (once) ----
    while True:
        kind = input("What do you want to move? (r)ows or (c)olumns: ").strip().lower()
        if kind in ("r", "c"):
            break
        print("Please type 'r' for rows or 'c' for columns.")

    # ---- 2) MOVING ROWS ----
    if kind == "r":
        while True:
            n_rows = len(df_clean)
            if n_rows <= 1:
                print("Not enough rows to move.")
                break

            print("\nCurrent rows (index shown on the left):")
            _print_rows_preview(df_clean)

            try:
                pair = input(f"\nWhich row do you want to move and to where? "
                             f"(enter two numbers 1–{n_rows}, e.g. '2 5'): ")
                src_str, dest_str = pair.split()
                src = int(src_str)
                dest = int(dest_str)
            except ValueError:
                print("Please enter TWO integers separated by space, like: 2 5")
                continue

            if not (1 <= src <= n_rows and 1 <= dest <= n_rows):
                print("Row numbers out of range.")
                continue

            indices = list(range(n_rows))
            row_idx = indices.pop(src - 1)
            indices.insert(dest - 1, row_idx)
            df_clean = df_clean.iloc[indices].reset_index(drop=True)

            print("\nNew row layout:")
            _print_rows_preview(df_clean)

            more = input("\nDo you want to move another row? (yes/no): ").strip().lower()
            if more not in ("yes", "y"):
                break

    # ---- 3) MOVING COLUMNS ----
    else:  # kind == "c"
        while True:
            cols = list(df_clean.columns)
            n_cols = len(cols)
            if n_cols <= 1:
                print("Not enough columns to move.")
                break

            print("\nCurrent columns order:")
            for i, c in enumerate(cols, 1):
                print(f"  {i}) {c}")

            print("\nCurrent table:")
            _print_rows_preview(df_clean)
            try:
                pair = input(f"\nWhich column do you want to move and to where? "
                             f"(enter two numbers 1–{n_cols}, e.g. '1 3'): ")
                src_str, dest_str = pair.split()
                src = int(src_str)
                dest = int(dest_str)
            except ValueError:
                print("Please enter TWO integers separated by space, like: 1 3")
                continue

            if not (1 <= src <= n_cols and 1 <= dest <= n_cols):
                print("Column numbers out of range.")
                continue

            col_name = cols.pop(src - 1)
            cols.insert(dest - 1, col_name)
            df_clean = df_clean[cols]

            print("\nNew column layout:")
            print(df_clean)

            more = input("\nDo you want to move another column? (yes/no): ").strip().lower()
            if more not in ("yes", "y"):
                break

    return df_clean

def _print_rows_preview(df):
    n = len(df)
    print(f"\nTotal rows: {n}")

    if n <= 20:
        print(df)
    else:
        print("\n--- FIRST 10 ROWS ---")
        print(df.head(10))

        print("\n...")

        print("\n--- LAST 10 ROWS ---")
        print(df.tail(10))

def _choose_column(columns, prompt):
    print(prompt)
    for i, col in enumerate(columns):
        print(f"[{i}] {col}")
    choice = input("Enter column number (or 'q' to cancel): ").strip()

    if choice.lower() == "q":
        return None

    try:
        idx = int(choice)
        if 0 <= idx < len(columns):
            return columns[idx]
    except ValueError:
        pass

    print("Invalid choice.")
    return None

def _choose_multiple_columns(columns, prompt):
    print(prompt)
    for i, col in enumerate(columns):
        print(f"[{i}] {col}")
    choice = input(
        "Enter column number(s) separated by commas or spaces (or 'q' to cancel): "
    ).strip()

    if choice.lower() == "q":
        return None

    # split on commas OR whitespace
    tokens = re.split(r"[,\s]+", choice.strip())
    tokens = [t for t in tokens if t]  # remove empty strings

    idxs = []
    for t in tokens:
        try:
            idxs.append(int(t))
        except ValueError:
            print(f"Ignoring invalid token: {t!r}")

    result = []
    for idx in idxs:
        if 0 <= idx < len(columns):
            result.append(columns[idx])
        else:
            print(f"Ignoring invalid index: {idx}")

    if not result:
        print("No valid columns selected.")
        return None

    return result

def _style_axes(ax, x_label, y_label, title):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True)

def plot_data(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Plot action:
      - choose X
      - choose one or more Y
      - choose plot type
      - save PNG to Plots/  (CSV not touched)
    """
    if df.empty:
        print("DataFrame is empty, nothing to plot.")
        return df

    all_cols = list(df.columns)

    # X
    x_col = _choose_column(all_cols, "\nChoose X-axis column:")
    if x_col is None:
        return df

    # Y (multi)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        numeric_cols = all_cols

    y_cols = _choose_multiple_columns(numeric_cols, "\nChoose Y-axis column(s):")
    if not y_cols:
        return df

    print("\nChoose plot type:")
    print("[1] Line (y vs x)")
    print("[2] Scatter (y vs x)")
    print("[3] Bar (y vs x)")
    plot_choice = input("Enter number: ").strip()

    x_raw = df[x_col]
    x_num = pd.to_numeric(x_raw, errors="coerce")
    x_is_mostly_numeric = x_num.notna().mean() > 0.8

    fig, ax = plt.subplots()

    # line / scatter (multi-Y)
    if plot_choice in ("1", "2"):
        if not x_is_mostly_numeric:
            print("X column is not numeric enough for line/scatter. Try bar plot instead.")
            plt.close(fig)
            return df

        any_plotted = False
        for y_col in y_cols:
            y_raw = df[y_col]
            y_num = pd.to_numeric(y_raw, errors="coerce")
            mask = x_num.notna() & y_num.notna()
            x = x_num[mask]
            y = y_num[mask]

            if y.empty:
                print(f"No valid numeric data for {y_col}, skipping.")
                continue

            if plot_choice == "1":
                ax.plot(x, y, label=y_col)
            else:
                ax.scatter(x, y, label=y_col)

            any_plotted = True

        if not any_plotted:
            print("Nothing to plot after cleaning data.")
            plt.close(fig)
            return df

        title = f"{', '.join(y_cols)} vs {x_col} ({'line' if plot_choice == '1' else 'scatter'})"
        _style_axes(ax, x_col, ", ".join(y_cols), title)
        if len(y_cols) > 1:
            ax.legend()

    # bar (use first Y)
    elif plot_choice == "3":
        if len(y_cols) > 1:
            print("Bar plot: using only the first selected Y column.")
        y_col = y_cols[0]

        y_raw = df[y_col]
        y_num = pd.to_numeric(y_raw, errors="coerce")

        if x_is_mostly_numeric:
            mask = x_num.notna() & y_num.notna()
            x = x_num[mask]
            y = y_num[mask]
        else:
            mask = y_num.notna()
            x = x_raw[mask].astype(str)
            y = y_num[mask]

        if y.empty:
            print("No valid data to plot.")
            plt.close(fig)
            return df

        ax.bar(x, y)
        title = f"{y_col} vs {x_col} (bar)"
        _style_axes(ax, x_col, y_col, title)
        plt.xticks(rotation=45)

    else:
        print("Unknown plot type.")
        plt.close(fig)
        return df

    plt.tight_layout()

    # ===== ASK USER WHERE TO SAVE PNG (relative to project folder) =====
    # project folder = one level above Scripts (where this file lives)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    base = os.path.splitext(os.path.basename(file_path))[0]
    y_part = "_".join(y_cols).replace(" ", "_").replace("/", "_")
    default_name = f"{base}_{x_col}_vs_{y_part}.png"

    print("\n==============================")
    print("Saving plot PNG")
    print("==============================")
    print("Project directory:", project_root)
    print("\nEnter output PNG path relative to the Project folder.")
    print("Examples:")
    print(f"   {default_name}")
    print(f"   Plots/{default_name}")
    print("   figures/run1/iv_curve.png\n")

    while True:
        out = input(f"Output PNG path [{default_name}]: ").strip()

        # allow empty -> use default name in project root
        if not out:
            out = default_name

        if not out.lower().endswith(".png"):
            print("Output must end with .png")
            continue

        png_path = os.path.join(project_root, out)

        # Create folder(s) if needed
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        break

    plt.savefig(png_path, dpi=300)
    print("\nPlot saved as:\n  ", png_path)

    plt.close(fig)
    return df