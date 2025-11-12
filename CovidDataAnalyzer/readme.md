# CovidDataAnalyzer

A Python project that loads and analyzes COVID-19 data using Pandas.
The tool offers automatic insights, descriptive statistics, filtering options,
and the ability to save filtered datasets.

## Features
- Load CSV datasets
- Print dataset shape, column info, missing values, and descriptive statistics
- Generate human-readable insights (highest cases, averages, death rate, etc.)
- Filter by country name and case thresholds
- Save filtered results to a new CSV file

## Getting Started
Install dependencies:
pip install pandas numpy

Dataset source: Kaggle – COVID-19 Dataset
### Example usage:
```python
from covid_analyzer import CovidDataAnalyzer

analyzer = CovidDataAnalyzer("dataset/country_wise_latest.csv")
analyzer.describe_data()
analyzer.generate_insights()
analyzer.filter_high_cases()
analyzer.save_filtered_data("reports/high_cases.csv")

