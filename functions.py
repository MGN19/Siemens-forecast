# VIsualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import holidays

main_color = '#009C8C'

def create_time_features_from_date(df, date_column="DATE"):
    """
    This function creates time-related features from a given date column in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the date column.
    date_column (str): The name of the date column (default is "DATE").
    
    Returns:
    pd.DataFrame: The original DataFrame with added time-related features.
    """
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Create Time Features
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Day'] = df[date_column].dt.day
    df['WeekNr'] = df[date_column].dt.isocalendar().week
    df['NameDayOfWeek'] = df[date_column].dt.day_name()
    df['DayOfWeek'] = df[date_column].dt.weekday
    
    # Create a weekend flag (1 if Saturday/Sunday, 0 otherwise)
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Create a holiday flag based on German holidays
    german_holidays = holidays.Germany(years=df[date_column].dt.year.unique())
    df['IsHoliday'] = df[date_column].apply(lambda x: 1 if x in german_holidays else 0)

    return df

# Visualizations
def plot_nr_sales_by_day_of_week(sales_by_date, date_column="DATE", sales_column="Sales_EUR", color=main_color):
    # Plot the distribution of sales by day of the week with specified color
    plt.figure(figsize=(12, 5))
    sns.countplot(data=sales_by_date, x="NameDayOfWeek", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], palette=[color])
    plt.title("Number of Sales by Day of the Week")
    plt.show()

def plot_total_sales_by_day_of_week(sales_data, date_column="DATE", sales_column="Sales_EUR", color=main_color):
    # Sum the sales for each day of the week
    sales_by_day = sales_data.groupby("NameDayOfWeek")[sales_column].sum().reset_index()

    # Ensure the days of the week are in order
    sales_by_day["NameDayOfWeek"] = pd.Categorical(sales_by_day["NameDayOfWeek"], 
                                                  categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
                                                  ordered=True)
    
    # Sort by day of the week
    sales_by_day = sales_by_day.sort_values("NameDayOfWeek")

    # Plot the summed sales by day of the week
    plt.figure(figsize=(12, 5))
    sns.barplot(data=sales_by_day, x="NameDayOfWeek", y=sales_column, palette=[color])
    plt.title("Total Sales (EUR) by Day of the Week")
    plt.ylabel("Total Sales (EUR)")
    plt.show()
