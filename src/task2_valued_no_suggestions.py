import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round as spark_round

def initialize_spark(app_name="Task2_Valued_Non_Contributors"):
    """Initialize and return a SparkSession."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """Load employee data from CSV into a Spark DataFrame."""
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the file path.")

    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def identify_valued_non_contributors(df):
    """Identify employees who have a Satisfaction Rating >= 4 but have not provided suggestions."""
    valued_non_contributors_df = df.filter((col("SatisfactionRating") >= 4) & (col("ProvidedSuggestions") == False))
    
    total_employees = df.count()
    valued_non_contributors_count = valued_non_contributors_df.count()
    proportion = (valued_non_contributors_count / total_employees) * 100 if total_employees > 0 else 0

    return valued_non_contributors_count, round(proportion, 2)

def write_output(output_path, valued_non_contributors_count, proportion):
    """Ensure directory exists and write results to a file."""
    
    output_dir = os.path.dirname(output_path)
    
    # Create the directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w") as f:
        f.write(f"Number of Employees Feeling Valued without Suggestions: {valued_non_contributors_count}\n")
        f.write(f"Proportion: {proportion}%\n")

def main():
    """Main function to execute Task 2."""
    spark = initialize_spark()

    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-bhanukocherla/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-bhanukocherla/outputs/task2/valued_non_contributors.txt"

    try:
        df = load_data(spark, input_file)
    except FileNotFoundError as e:
        print(e)
        spark.stop()
        return

    valued_non_contributors_count, proportion = identify_valued_non_contributors(df)

    print(f"Number of Employees Feeling Valued without Suggestions: {valued_non_contributors_count}")
    print(f"Proportion: {proportion}%")

    write_output(output_file, valued_non_contributors_count, proportion)
    print(f"Results saved to {output_file}")

    spark.stop()

if __name__ == "__main__":
    main()
