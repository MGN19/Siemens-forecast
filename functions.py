# VIsualization libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import holidays
import plotly.graph_objects as go
import itertools
from plotly.subplots import make_subplots
from matplotlib.ticker import MaxNLocator, FuncFormatter


main_color = '#009C8C'

# Data Exploration
# Missing Value Analysis
def missing_value_summary(dataframe):
    """
    Provides a summary of missing values in the DataFrame.
    
    Parameters:
        dataframe: The DataFrame to analyze.
    
    Returns:
        pd.DataFrame: Summary of columns with missing values, including unique values, NaN count, and percentage.
    """
    nan_columns = dataframe.columns[dataframe.isna().any()].tolist()
    summary_data = []
    
    for column in nan_columns:
        nan_number = dataframe[column].isna().sum()
        nan_percentage = (nan_number / len(dataframe)) * 100
        unique_values = dataframe[column].nunique()
        summary_data.append({
            'Unique Values': unique_values,
            'NaN Values': nan_number,
            'Percentage NaN': nan_percentage
        })
    
    summary = pd.DataFrame(summary_data, index=nan_columns)
    return summary


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


# Histogram subplots
def plot_boxplots(pivoted_data, num_rows=5, num_cols=3, box_color=main_color):
    num_plots = num_rows * num_cols
    fig = make_subplots(rows=num_rows, cols=num_cols)
    
    for i, product in enumerate(pivoted_data.columns[1:num_plots+1]):
        row = i // num_cols + 1
        col = i % num_cols + 1
        fig.add_trace(
            go.Box(
                y=pivoted_data[product], 
                name=product,
                marker=dict(color=box_color),
                line=dict(color=box_color)
            ), 
            row=row, col=col
        )
    
    fig.update_layout(
        title='Total sales of each product',
        height=800,
        width=1100,
        showlegend=False
    )
    
    fig.show()


# Scatter plot subplots (corr > 50)
def plot_high_correlation_pairs(pivoted_data, threshold=0.50, num_cols=5):
    columns = pivoted_data.columns
    all_pairs = [(x, y, np.corrcoef(pivoted_data[x], pivoted_data[y])[0, 1]) 
                 for x, y in itertools.combinations(columns, 2)]
    
    filtered_pairs = sorted([(x, y, r) for x, y, r in all_pairs if abs(r) > threshold], 
                            key=lambda p: abs(p[2]), reverse=True)
    
    num_rows = -(-len(filtered_pairs) // num_cols) 
    fig = make_subplots(rows=num_rows, cols=num_cols, 
                        subplot_titles=[f"{x} vs {y}" for x, y, _ in filtered_pairs])
    
    for i, (x, y, r) in enumerate(filtered_pairs):
        row, col = divmod(i, num_cols)
        x_data, y_data = pivoted_data[x], pivoted_data[y]
        slope, intercept = np.polyfit(x_data, y_data, 1)
        
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', marker=dict(size=4, opacity=0.6)), row=row+1, col=col+1)
        fig.add_trace(go.Scatter(x=x_data, y=slope * x_data + intercept, mode='lines', line=dict(color='red')), row=row+1, col=col+1)
        
        fig.add_annotation(x=x_data.mean(), y=y_data.max(), text=f"r = {r:.2f}", showarrow=False, font=dict(size=12), row=row+1, col=col+1)
    
    fig.update_layout(title="Pairwise Scatter Analysis (r > 0.50)", height=300*num_rows, width=250*num_cols, showlegend=False)
    fig.show()


# Stacked plot
def plot_stacked_bar(yearly_data):
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", n_colors=14)
    fig, ax = plt.subplots(figsize=(14, 8))

    yearly_data.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", linewidth=0.6, zorder=2)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel("Year", fontsize=16, fontweight="bold", labelpad=15)
    ax.set_ylabel("Value", fontsize=16, fontweight="bold", labelpad=15)
    ax.set_title("Sales per product per year", fontsize=18, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=12, weight="bold")

    ax.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12, title_fontsize=14, frameon=True, facecolor='white', edgecolor='black')

    for rect in ax.patches:
        rect.set_zorder(1)
        rect.set_edgecolor('black')

    # Disable scientific notation on y-axis and use normal formatting
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer formatting
    ax.ticklabel_format(style='plain', axis='y')  # Disable scientific notation on y-axis

    # Add comma as thousands separator to y-axis values
    def comma_format(x, pos):
        return f'{x:,.0f}'  # Format the number with commas

    ax.yaxis.set_major_formatter(FuncFormatter(comma_format))

    plt.tight_layout()
    plt.show()
