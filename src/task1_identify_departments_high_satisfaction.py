from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round as spark_round

def initialize_spark(app_name="Task1_Identify_Departments"):
    """
    Initialize and return a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """
    Load the employee data from a CSV file into a Spark DataFrame.

    Parameters:
        spark (SparkSession): The SparkSession object.
        file_path (str): Path to the employee_data.csv file.

    Returns:
        DataFrame: Spark DataFrame containing employee data.
    """
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    # Check if file exists before attempting to load
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the file path.")

    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def identify_departments_high_satisfaction(df):
    """
    Identify departments with more than 50% of employees having a Satisfaction Rating > 4 and Engagement Level 'High'.

    Parameters:
        df (DataFrame): Spark DataFrame containing employee data.

    Returns:
        DataFrame: DataFrame containing departments meeting the criteria with their respective percentages.
    """
    # Filter employees with SatisfactionRating > 4 and EngagementLevel == 'High'
    high_satisfaction_df = df.filter((col("SatisfactionRating") > 4) & (col("EngagementLevel") == "High"))

    # Count total employees per department
    total_employees_df = df.groupBy("Department").agg(count("*").alias("TotalEmployees"))

    # Count employees who meet the criteria per department
    high_satisfaction_count_df = high_satisfaction_df.groupBy("Department").agg(count("*").alias("SatisfiedEngaged"))

    # Calculate the percentage
    result_df = high_satisfaction_count_df.join(total_employees_df, "Department").withColumn(
        "Percentage", spark_round((col("SatisfiedEngaged") / col("TotalEmployees")) * 100, 2)
    ).filter(col("Percentage") > 5)  # Filtering departments where percentage > 50%

    # Select only required columns
    return result_df.select("Department", "Percentage")

def write_output(result_df, output_path):
    """
    Write the result DataFrame to a CSV file.

    Parameters:
        result_df (DataFrame): Spark DataFrame containing the result.
        output_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    result_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

def main():
    """
    Main function to execute Task 1.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths (Update this path according to your workspace)
    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-bhanukocherla/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-bhanukocherla/outputs/task1/departments_high_satisfaction.csv"

    # Load data
    try:
        df = load_data(spark, input_file)
    except FileNotFoundError as e:
        print(e)
        spark.stop()
        return

    # Perform Task 1
    result_df = identify_departments_high_satisfaction(df)

    # Show result for debugging (Optional)
    result_df.show()

    # Write the result to CSV
    write_output(result_df, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()
