import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prefect import flow, task, get_run_logger
import scipy
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns 
import time



# Your pipeline should be structured as a series of clearly defined tasks orchestrated
#  inside a single Prefect flow.
# Each major stage of the analysis should be its own @task,
#  and all tasks should be coordinated inside one @flow.
# Use get_run_logger() inside every task instead of print() -- this is one of the core practices from the lesson,
# and it means your results will appear in both the terminal and the Prefect dashboard.

# Task 1:
# Load Multiple Years of Data
# Load data from all ten yearly CSV files into a single DataFrame.
# Your implementation should not duplicate code for each year -- iterate over a list
#  of file paths and load them in a loop.

# You discovered some quirks when you inspected the raw files.
# Make sure you account for those when calling pd.read_csv().


# Add retries=3, retry_delay_seconds=2 to this task's decorator.
# File I/O is exactly the kind of operation that can fail intermittently
#  in production pipelines, and this is where retries earn their keep.


@task(retries=3, retry_delay_seconds=2)
def create_series(list_of_file_paths):
    # create series from the list of file paths,
    # load the data from each file into a DataFrame.
    logger = get_run_logger()
    series = []
    standard_columns = [
        'Ranking', 'Country', 'Regional indicator', 'Happiness score', 'GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
    ]

    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    for file_path in list_of_file_paths:
        try:
            df = pd.read_csv(file_path,sep=';', decimal=',')

            # Log the shape of the DataFrame after loading each file  
            logger.info(f"Shape of {file_path}: {df.shape}")

            df.columns = standard_columns[:len(df.columns)]

            # Extract the year from the file name and add it as a new column in the DataFrame
            year = file_path.split('_')[-1].split('.')[0]   
            df['Year'] = year 
            series.append(df)
            logger.info(f"Successfully loaded data for year: {year}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}. Skipping...")

    if not series:
        logger.warning("No dataframes were loaded!")
        return None
    # Concatenate all the DataFrames in the series into a single DataFrame
    # After loading and merging, save the combined dataset to: assignments_01/outputs/merged_happiness.csv
    combined_df = pd.concat(series, ignore_index=True)
    save_path = os.path.join(output_dir, "merged_happiness.csv")
    combined_df.to_csv(save_path, index=False)

    logger.info(f"Successfully saved merged dataset to {save_path}")
    return combined_df

@task
def cleaning_df(df):
    logger = get_run_logger()
    if df is None:
        logger.warning("No data available for cleaning.")
        return

    # Check for missing values in the DataFrame
    missing_values = df.isnull().sum()
    logger.info("Missing values in each column:")
    logger.info(missing_values)

    # If there are missing values, drop rows with missing values
    if missing_values.any():
        df_cleaned = df.dropna()
        logger.info(f"Dropped {len(df) - len(df_cleaned)} rows with missing values.")
    else:
        df_cleaned = df
        logger.info("No missing values found. No rows dropped.")

    return df_cleaned

# Task 2: Descriptive Statistics
# Compute and log overall descriptive statistics for Happiness score:
#  mean, median, and standard deviation.
# Then compute and log the mean happiness score grouped by year and by region.
# Looking at the regional breakdown is often the most interesting part of this dataset 
# -- you may already have a hypothesis about which regions rank highest before you run the numbers.
@task
def descriptive_statistics(df):
    logger = get_run_logger()
    if df is None:
        logger.warning("No data available for descriptive statistics.")
        return

    # Overall descriptive statistics for Happiness score
    mean = df['Happiness score'].mean()
    median = df['Happiness score'].median()
    std_dev = df['Happiness score'].std()

    logger.info(f"Overall Descriptive Statistics for Happiness score:")
    logger.info(f"Happiness score Mean: {mean}")
    logger.info(f"Happiness score Median: {median}")
    logger.info(f"Happiness score Standard Deviation: {std_dev}")

    # Mean happiness score grouped by year
    mean_by_year = df.groupby('Year')['Happiness score'].mean()
    logger.info("Mean happiness score by year:")
    logger.info(mean_by_year)

    # Mean happiness score grouped by region
    mean_by_region = df.groupby('Regional indicator')['Happiness score'].mean()
    logger.info("Mean happiness score by region:")
    logger.info(mean_by_region)
    
# Task 3: Visual Exploration

# Log a message after each plot is saved so you can see the progress in the Prefect dashboard.

@task
def visual_exploration(df):
    logger = get_run_logger()
    if df is None:
        logger.warning("No data available for visual exploration.")
        return

    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    
    # A histogram of all happiness scores across all years. Save as happiness_histogram.png.
    # Histogram of happiness scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Happiness score'], bins=20, kde=True) 

    plt.title('Distribution of Happiness Scores')
    plt.xlabel('Happiness Score')
    plt.ylabel('Frequency')
    histogram_path = os.path.join(output_dir, "happiness_histogram.png")
    plt.savefig(histogram_path)
    logger.info(f"Saved histogram to {histogram_path}")
    plt.close()

    # Boxplot comparing happiness score distributions across years
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Year', y='Happiness score', data=df)
    plt.title('Happiness Score Distribution by Year')
    boxplot_path = os.path.join(output_dir, "happiness_by_year.png")
    plt.savefig(boxplot_path)
    logger.info(f"Saved boxplot to {boxplot_path}")
    plt.close()

    # Scatter plot showing the relationship between GDP per capita and happiness score
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='GDP per capita', y='Happiness score', data=df)
    plt.title('GDP per Capita vs Happiness Score')
    scatter_plot_path = os.path.join(output_dir, "gdp_vs_happiness.png")
    plt.savefig(scatter_plot_path)
    logger.info(f"Saved scatter plot to {scatter_plot_path}")
    plt.close()

    # A correlation heatmap (using sns.heatmap() with annot=True)
    # showing the Pearson correlations between all numeric columns.
    # Save as correlation_heatmap.png.
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numeric Columns')
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    logger.info(f"Saved correlation heatmap to {heatmap_path}")
    plt.close()


# Task 4: Hypothesis Testing

# Null Hypothesis: There is no real difference in global happiness between 2019 and 2020. Any change is due to random chance.
# Alternative Hypothesis: There is a significant difference in happiness scores between 2019 and 2020.
# Alpha: 0.05. 
# If the p-value is below Alpha, we "reject the null."

@task
def hypothesis_testing(df):
    logger = get_run_logger()
    
   # --- TEST 1: Happiness Scores by Year 2019 andd 2020 ---
    # 1. Filter the data for the two years
    data_2019 = df[df['Year'] == '2019']['Happiness score']
    data_2020 = df[df['Year'] == '2020']['Happiness score']
    
    # 2. Calculate means
    mean_2019 = data_2019.mean()
    mean_2020 = data_2020.mean()
    
    # 3. Perform the Independent Samples T-Test
    t_test_results = stats.ttest_ind(data_2019, data_2020, equal_var=False)
    
    # 4. Logging the technical results
    logger.info(f"Mean Happiness 2019: {mean_2019:.3f} and Mean Happiness 2020: {mean_2020:.3f}")
    logger.info(f"T-Statistic: {t_test_results.statistic:.3f} and P-Value: {t_test_results.pvalue:.4f}")
    
    # 5. Interpretation of the results
    if t_test_results.pvalue < 0.05:
            logger.info("According to the data the happiness affected by pandemic a lot")
    else:
            logger.info("The results of a testing show that the happiness score is remained stable inspite the pandemic")
      
    
    
    # Null Hypothesis H0: Mean NA/ANZ = Mean South AsiaThere is no difference in the average happiness between the North America/ANZ region and the South Asia region.
    # Alternative Hypothesis Ha: Mean NA/ANZ != Mean South AsiaThere is a significant difference in average happiness scores
    # between the North America/ANZ region and the South Asia region.
    # Alpha: 0.05.

    # --- TEST 2: North America/ANZ vs South Asia happiness ---

    region_na = df[df['Regional indicator'] == 'North America and ANZ']['Happiness score']
    region_sa = df[df['Regional indicator'] == 'South Asia']['Happiness score']
    
    t_test_results_reg = stats.ttest_ind(region_na, region_sa, nan_policy='omit')
    
    logger.info(f"HYPOTHESIS TEST (NA/ANZ vs South Asia):")
    logger.info(f"Means : NA/ANZ: {region_na.mean():.3f} and South Asia: {region_sa.mean():.3f}")
    logger.info(f"T-statistic: {t_test_results_reg.statistic:.3f} and P-value: {t_test_results_reg.pvalue:.4e}")

    if t_test_results_reg.pvalue < 0.05:
        logger.info("We reject the null hypothesis. There is a massive, statistically significant "
                    "gap in happiness between North America/ANZ and South Asia, suggesting geography and "
                    "socio-economic factors play a critical role.")
        logger.info("There are a huge difference in happiness of theose two regions," \
                    "because of the difference in socio economic wellness parametrs in the two regions")
    else:
        logger.info("No significant difference found between these regions.")
    return t_test_results

# Task 5: Correlation and Multiple Comparisons
@task
def correlation_analysis(df):
    logger = get_run_logger()
    
    # 1. Identify numeric variables
    numeric_df = df.select_dtypes(include=[np.number])
    explanatory_vars = [col for col in numeric_df.columns if col not in ['Happiness score', 'Ranking']]
    
    correlation_results = []
    
    # 2. Compute number of tests for Bonferroni correction
    num_tests = len(explanatory_vars)
    alpha = 0.05
    adjusted_alpha = alpha / num_tests
    
    logger.info(f"Performing {num_tests} correlation tests.")
    logger.info(f"Original Alpha: {alpha}")
    logger.info(f"Bonferroni Adjusted Alpha: {adjusted_alpha:.4e}")
    
    # 3. Iterate over each variable and run the test
    for column in explanatory_vars:
        # Drop NaN pairs 
        valid_data = df[[column, 'Happiness score']].dropna()
        
        corr_coef, p_value = pearsonr(valid_data[column], valid_data['Happiness score'])
        
        # Determine significance levels
        is_significant_basic = p_value < alpha
        is_significant_adjusted = p_value < adjusted_alpha
        
        # Store results for the report
        result_entry = {
            'variable': column,
            'coef': corr_coef,
            'p_value': p_value,
            'sig_basic': is_significant_basic,
            'sig_adj': is_significant_adjusted
        }
        correlation_results.append(result_entry)
        
        # 4. Log the findings for this specific variable
        logger.info(f"--- Variable: {column} ---")
        logger.info(f"  Correlation: {corr_coef:.3f}")
        logger.info(f"  P-value: {p_value:.4e}")
        
        if is_significant_adjusted:
            logger.info(f"  Result: SIGNIFICANT (Passed strict Bonferroni test)")
        elif is_significant_basic:
            logger.info(f"  Result: POSSIBLY SIGNIFICANT (Passed alpha=0.05, but failed Bonferroni)")
        else:
            logger.info(f"  Result: NOT SIGNIFICANT")
            
    return correlation_results
# Task 6: Summary Report

@task
def summary_report(df, t_test_results, correlation_results):
    logger = get_run_logger()
    
    # 1. Dataset Scope
    countries_count = df['Country'].nunique()
    years_count = df['Year'].nunique()
    logger.info("The analysis covers a comprehensive dataset of global happiness, spanning multiple years and regions.")
    logger.info(f"{countries_count} countries over a {years_count}-year period (2015-2024).")

    # 2. Regional Performance
    mean_by_region = df.groupby('Regional indicator')['Happiness score'].mean().sort_values(ascending=False)
    top3 = ", ".join(mean_by_region.head(3).index)
    bottom3 = ", ".join(mean_by_region.tail(3).index)
    

    logger.info(f"Those regions with the higest happpiness levels: {top3}.")
    logger.info(f"Those regions with the lowest happpiness levels: {bottom3}.")

    # 3. Pandemic Impact (The T-Test result)
    if t_test_results.pvalue < 0.05:
        logger.info("Comparing 2019 to 2020, found that there was a statistically significant change in happiness scores.")
    else:
        logger.info("Comparing 2019 to 2020, found that even the pandemic did not change the country happiness levels.")
   
    # 4. The variable most strongly correlated with happiness score (after Bonferroni correction).
    
    significant_after_correction = [r for r in correlation_results if r['sig_adj']]

    if significant_after_correction:
        # abs beacause the negative dtrong corelation is the same as the positive strong corelation but diff dir
        top_factor = max(significant_after_correction, key=lambda x: abs(x['coef']))
        logger.info(f"CORRELATION ANALYSIS: After applying the Bonferroni correction to "
                    f"account for multiple comparisons, the variable most strongly "
                    f"correlated with Happiness is '{top_factor['variable']}' "
                    f"with a coefficient of {top_factor['coef']:.3f}.")
    else:
        logger.info("CORRELATION ANALYSIS: After Bonferroni correction, no variables "
                    "remained statistically significant.")
@flow
def data_pipeline(list_of_file_paths):
    logger = get_run_logger()
    logger.info("Starting World Happiness Data Pipeline")
    # Execute Task 1
    combined_data = create_series(list_of_file_paths)
    print(combined_data.info())
    # Execute Task 1.1
    cleaned_data = cleaning_df(combined_data)
    # Execute Task 2
    visual_exploration(cleaned_data)
    # Execute Task 3    
    descriptive_statistics(cleaned_data)   
    # Execute Task 4
    t_test_results = hypothesis_testing(cleaned_data)
    # Execute Task 5
    correlation_results = correlation_analysis(cleaned_data)
    # Execute Task 6
    summary_report(cleaned_data, t_test_results, correlation_results)

    return cleaned_data  

  

if __name__ == "__main__":
    # Get the directory of the current script (project_01.py)
    base_path = os.path.dirname(os.path.abspath(__file__))
    #create a list of file paths for the ten yearly CSV files
    file_paths = file_paths = [
        os.path.join(base_path, "data", f"world_happiness_{year}.csv") 
        for year in range(2015, 2025)
    ]
    # Call the data_pipeline flow with the list of file paths
    # Call the flow
    final_df = data_pipeline(file_paths)
    print("Pipeline Complete!")
    print(final_df.head()) 
    print(f"Total rows in merged and cleaned data: {len(final_df)}")
    time.sleep(0.2)

  
