from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, when

def initialize_spark(app_name="Task3_Compare_Engagement_Levels"):
    """Initialize and return a SparkSession."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """Load employee data from CSV into a Spark DataFrame."""
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the file path.")

    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def convert_engagement_to_numeric(df):
    """
    Convert categorical Engagement Levels ('Low', 'Medium', 'High') to numerical values.

    Returns:
        DataFrame with EngagementLevel converted to numerical values.
    """
    df = df.withColumn("EngagementLevelNumeric", 
        when(col("EngagementLevel") == "Low", 1)
        .when(col("EngagementLevel") == "Medium", 2)
        .when(col("EngagementLevel") == "High", 3)
        .otherwise(None)  # Handle unexpected values
    )
    
    return df

def compare_engagement_levels(df):
    """
    Compare Engagement Levels across Job Titles and find the top-performing Job Title.

    Returns:
        DataFrame with JobTitle and its corresponding average Engagement Level.
    """
    df = convert_engagement_to_numeric(df)

    avg_engagement_df = df.groupBy("JobTitle").agg(avg("EngagementLevelNumeric").alias("AvgEngagementLevel"))

    return avg_engagement_df.orderBy(col("AvgEngagementLevel").desc())

def write_output(result_df, output_path):
    """Ensure directory exists and write results to a CSV file."""
    import os
    output_dir = os.path.dirname(output_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

def main():
    """Main function to execute Task 3."""
    spark = initialize_spark()

    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-bhanukocherla/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-bhanukocherla/outputs/task3/engagement_comparison.csv"

    try:
        df = load_data(spark, input_file)
    except FileNotFoundError as e:
        print(e)
        spark.stop()
        return

    result_df = compare_engagement_levels(df)

    print("Engagement Level Comparison Across Job Titles:")
    result_df.show()

    write_output(result_df, output_file)
    print(f"Results saved to {output_file}")

    spark.stop()

if __name__ == "__main__":
    main()
