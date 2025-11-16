import pandas as pd
import numpy as np


class CovidDataAnalyzer:
    def __init__(self, file_path=None):
        """
        Initialize the analyzer by loading the dataset from the given file path (if there is one).

        Attributes:
            self.data: stores the full loaded dataset
            self.filtered_data: stores filtered versions of the dataset

        Parameters:
        file_path (str): Path to the CSV file containing COVID-19 data.
        """
        self.data = pd.DataFrame()
        self.filtered_data = pd.DataFrame()

        # If a path is provided at initialization, use the public method to load it.
        if file_path:
            if self.load_data(file_path):
                print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")

        # ================================================================
        #   COLUMN NORMALIZATION
        # ================================================================
    def normalize_columns(self):
        """
        Ensures the dataset uses consistent column names.
        Prevents duplicate renaming by checking first.
        """
        rename_map = {}

        if 'Confirmed' in self.data.columns and 'Confirmed_Cases' not in self.data.columns:
            rename_map['Confirmed'] = 'Confirmed_Cases'

        if 'Country/Region' in self.data.columns and 'Country' not in self.data.columns:
            rename_map['Country/Region'] = 'Country'

        if rename_map:
            self.data.rename(columns=rename_map, inplace=True)
        # ================================================================
        #   LOADING AND DESCRIBING
        # ================================================================
    def load_data(self, file_path):
        """
        Loads the dataset from the given CSV file path into self.data.
        This method is public and can be called externally at any time.

        Parameters:
        file_path (str): The path to the CSV file.

        Returns:
        bool: True if data loaded successfully, False otherwise.
        """
       # 1. Validation
        if not isinstance(file_path, str) or not file_path:
            print("Error: file_path must be a non-empty string.")
            return False

        try:
            self.data = pd.read_csv(file_path)
            self.filtered_data = pd.DataFrame()
            self.normalize_columns()
            return True

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return False
        except pd.errors.EmptyDataError:
            print(f"Warning: File at '{file_path}' is empty.")
            self.data = pd.DataFrame()
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def describe_data(self):
        """
        Prints dataset structure, statistics, column info, and missing values
        """
        if self.data.empty:
            print("Cannot describe data: Dataset is empty.")
            return

        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)

        print("\n" + "=" * 50)
        print("                 DATASET OVERVIEW")
        print("=" * 50)

        print("\n### 1. Dataset Shape")
        print(f"Rows: {self.data.shape[0]}, Columns: {self.data.shape[1]}")

        print("\n### 2. Column Info")
        self.data.info(verbose=False, memory_usage=False)

        print("\n### 3. Numerical Statistics")
        print(self.data.describe().T)

        print("\n### 4. Categorical Statistics")
        print(self.data.describe(include=['object', 'category']).T)

        print("\n### 5. Missing Values")
        print(self.data.isnull().sum())

        print("=" * 50)

    # ================================================================
    #   MISSING VALUES
    # ================================================================
    def handle_missing_values(self):
        """
        Fills missing (NaN) values in the self.data DataFrame:
        - Numeric columns are filled with 0.
        - Categorical (object/string) columns are filled with "Unknown".
        """
        if self.data.empty:
            print("Cannot handle missing values: The dataset is empty. Please load data first.")
            return

        print("\n" + "="*50)
        print("         HANDLING MISSING VALUES")
        print("="*50)

        # 1. Identify Numeric and Categorical Columns

        # Selects columns that are numeric (int, float)
        numeric_cols = self.data.select_dtypes(include=np.number).columns

        # Selects columns that are object (string/categorical)
        categorical_cols = self.data.select_dtypes(include="object").columns

        # 2. Impute Numeric Missing Values with 0

        # Use inplace=True to modify the DataFrame directly
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)
        print(f"Filled missing values in {len(numeric_cols)} numeric column(s) with **0**.")

        # 3. Impute Categorical Missing Values with "Unknown"

        # Use inplace=True to modify the DataFrame directly
        self.data[categorical_cols] = self.data[categorical_cols].fillna("Unknown")
        print(f"Filled missing values in {len(categorical_cols)} categorical column(s) with **'Unknown'**.")

        print("Missing value handling complete. Check `.info()` to verify.")

    # ================================================================
    #   FILTERING
    # ================================================================
    def filter_high_cases(self):
        """
        Filters the dataset (self.data) based on specific high-impact conditions
        and saves the result to self.filtered_data.

        Conditions:
        - 'Confirmed_Cases' > 100,000
        - 'Deaths' > 5,000
        - 'Country' is not "Unknown"
        """
        if self.data.empty:
            print("Cannot filter data: The main dataset (self.data) is empty. Please load data first.")
            return

        self.normalize_columns()

        print("\n" + "="*50)
        print("          APPLYING HIGH-CASE FILTER")
        print("="*50)

        # --- Required columns ---
        required = ['Confirmed_Cases', 'Deaths', 'Country']
        missing = [c for c in required if c not in self.data.columns]

        if missing:
            print(f"Error: Cannot filter. Missing columns: {', '.join(missing)}")
            return
    
        # --- Apply filter conditions ---
        cond_cases = self.data['Confirmed_Cases'] > 100_000
        cond_deaths = self.data['Deaths'] > 5_000
        cond_country = self.data['Country'] != "Unknown"
    
        combined = cond_cases & cond_deaths & cond_country
    
        # --- Save filtered results ---
        self.filtered_data = self.data[combined].copy()
    
        print(f"Original size:  {self.data.shape[0]} rows")
        print(f"Filtered size:  {self.filtered_data.shape[0]} rows")
        print("Filtered data saved to self.filtered_data.")
    
        return self.filtered_data

    def filter_by_date_range(self, start_date, end_date, date_column='Date'):
        """
        Filters the dataset (self.data) for records falling within the
        specified start_date and end_date range (inclusive).

        Parameters:
        start_date (str): The starting date for the filter (e.g., '2020-03-01').
        end_date (str): The ending date for the filter (e.g., '2020-04-30').
        date_column (str): The name of the date column in self.data (default 'Date').
        """
        df = self.filtered_data

        if df.empty:
            print("Dataset is empty. Load data or apply filters first.")
            return

        if date_column not in df.columns:
            print(f"Error: '{date_column}' column not found.")
            return

        print("\n" + "="*50)
        print("          APPLYING DATE RANGE FILTER")
        print("="*50)

        try:
            # Convert dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            df[date_column] = pd.to_datetime(df[date_column])

            # Filter
            condition = (df[date_column] >= start_dt) & (df[date_column] <= end_dt)
            self.filtered_data = df[condition].copy()

            print("Filter complete.")
            print(f"Result rows: {self.filtered_data.shape[0]}")

            return self.filtered_data

        except Exception as e:
            print(f"Error: {e}")

    # ================================================================
    #   GLOBAL STATISTICS (FILTERED DATA ONLY)
    # ================================================================
    def calculate_global_statistics(self):
        """
        Calculates global totals for Confirmed, Deaths, and Recovered cases.
        The method will refuse to run if no filtering has been applied yet.

    Prints:
        - Total confirmed cases
        - Total deaths
        - Total recovered cases
        """
        if self.filtered_data.empty:
            print("Cannot calculate statistics: filtered_data is empty. Apply a filter first.")
            return

        df = self.filtered_data
        required = ['Confirmed_Cases', 'Deaths', 'Recovered']

        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Missing columns: {missing}")
            return

        totals = df[required].sum()

        print("\n" + "=" * 50)
        print("      GLOBAL STATISTICS (FILTERED DATA)")
        print("=" * 50)

        try:
            print(f"Total Confirmed Cases: {int(totals['Confirmed_Cases']):,}")
            print(f"Total Deaths:          {int(totals['Deaths']):,}")
            print(f"Total Recovered:       {int(totals['Recovered']):,}")
        except:
            # If duplicate columns exist, totals['Confirmed_Cases'] may be a Series
            print("⚠ Fixing duplicate-column issue automatically...")
            print(f"Total Confirmed Cases: {int(totals['Confirmed_Cases'].iloc[0]):,}")
            print(f"Total Deaths:          {int(totals['Deaths'].iloc[0]):,}")
            print(f"Total Recovered:       {int(totals['Recovered'].iloc[0]):,}")

    # ================================================================
    #   SAVE + RESET
    # ================================================================
    def save_filtered_data(self, filename):
        """
        Saves the current self.filtered_data DataFrame to a specified CSV file.

        Parameters:
        filename (str): The name and path of the file to save the data to.
        """
        if self.filtered_data.empty:
            print("❗ Cannot save data: self.filtered_data is empty. Apply a filter first.")
            return

        if not isinstance(filename, str) or not filename.endswith('.csv'):
            print("🛑 Error: Filename must be a string and end with '.csv'.")
            return

        print("\n" + "="*50)
        print(f"          SAVING FILTERED DATA")
        print("="*50)

        try:
            # Save the DataFrame to the specified CSV file.
            # index=False prevents pandas from writing the DataFrame's row indices
            # as an extra, unnecessary column in the CSV file.
            self.filtered_data.to_csv(filename, index=False)

            rows = self.filtered_data.shape[0]
            print(f"✅ Successfully saved **{rows} rows** to **{filename}**.")

        except Exception as e:
            print(f"🛑 An error occurred during file saving: {e}")

    def reset_filtered_data(self):
        """
        Resets the self.filtered_data DataFrame to an empty DataFrame.
        """
        self.filtered_data = pd.DataFrame()
        print("Filtered_data has been reset.")

    # ================================================================
    #   AUTOMATED INSIGHTS
    # ================================================================
    def generate_insights(self):
        """
        Automatically prints readable insights from the loaded dataset:
        - Number of countries
        - Country with highest & lowest confirmed cases
        - Average confirmed cases
        - Death rate statistics (if available)
        - Missing value summary
        """

        if self.data.empty:
            print("Cannot generate insights: Dataset is empty.")
            return

        self.normalize_columns()

        if 'Country' not in self.data.columns or 'Confirmed_Cases' not in self.data.columns:
            print("Dataset missing required columns for insights.")
            return

        print("\n" + "="*55)
        print("                 AUTOMATED DATA INSIGHTS")
        print("="*55)

        # 1. Number of countries
        total_countries = self.data['Country'].nunique()
        print(f"There are {total_countries} countries in the dataset.")

        # 2. Highest and lowest confirmed cases
        max_row = self.data.loc[self.data['Confirmed_Cases'].idxmax()]
        min_row = self.data.loc[self.data['Confirmed_Cases'].idxmin()]

        print(f"The country with the highest confirmed cases is {max_row['Country']} "
              f"with {max_row['Confirmed_Cases']:,} cases.")

        print(f"The country with the lowest confirmed cases is {min_row['Country']} "
              f"with {min_row['Confirmed_Cases']:,} cases.")

        # 3. Average confirmed cases
        avg_cases = self.data['Confirmed_Cases'].mean()
        print(f"On average, countries reported about {avg_cases:,.0f} confirmed cases.")

        # 4. Death rate (if deaths exist)
        if 'Deaths' in self.data.columns and self.data['Deaths'].sum() > 0:
            total_cases = self.data['Confirmed_Cases'].sum()
            total_deaths = self.data['Deaths'].sum()
            death_rate = (total_deaths / total_cases) * 100

            print(f"The overall global death rate is approximately {death_rate:.2f}% "
                  f"({total_deaths:,} deaths out of {total_cases:,} confirmed cases).")
        else:
            print("Death statistics are not available in this dataset.")

        # 5. Missing value summary
        missing = self.data.isnull().sum()
        missing = missing[missing > 0]  # Only show columns with missing data

        if not missing.empty:
            print("\nSome columns contain missing values:")
            for col, count in missing.items():
                print(f" - {col}: {count} missing entries")
        else:
            print("No missing values detected in the dataset.")

        print("="*55)
