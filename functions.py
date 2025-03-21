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
import ipywidgets as widgets
from statsmodels.tsa.seasonal import seasonal_decompose
from IPython.display import display, clear_output
import matplotlib.ticker as ticker
import matplotlib.dates as mdates 

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

#### Visualizations
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

# Function to format tick labels as normal numbers (no scientific notation)
def normal_format(x, pos):
    return f'{x:,.0f}'  # Format with thousands separator and no decimal places

formatter = ticker.FuncFormatter(normal_format)

def plot_sales_data(new_monthly_sales):
    """
    Function to plot sales data as a line plot and histogram for each product.
    """
    for product, sales_data in new_monthly_sales.items():
        dates = sales_data.index  

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), facecolor='white') 

        # Convert sales data to numpy array
        sales_array = np.array(sales_data, dtype=float)

        # Mask 0s 
        masked_sales = np.ma.masked_equal(sales_array, 0)

        # Line plot
        ax1.plot(dates, masked_sales, marker='o', linestyle='-', linewidth=2, markersize=5, color=main_color)  
        ax1.set_title(f'{product} - Line Plot', fontsize=12, fontweight='bold')
        ax1.yaxis.set_major_formatter(formatter) 
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  
        ax1.tick_params(axis='x', rotation=20, labelsize=10)  
        ax1.grid(True, linestyle='--', alpha=0.7)  

        # Histogram 
        ax2.hist(sales_data, bins=10, color=main_color, edgecolor='black', alpha=0.75)  
        ax2.set_title(f'{product} - Histogram', fontsize=12, fontweight='bold')
        ax2.xaxis.set_major_formatter(formatter)  
        ax2.tick_params(axis='x', rotation=20, labelsize=10)  
        ax2.grid(True, linestyle='--', alpha=0.7)  

        plt.subplots_adjust(wspace=0.4)  
        plt.tight_layout()  
        plt.show()

# Sales line plot
def sales_data_line(sales_data, rows=5, cols=3, title="Sales of each product category"):
    """
    Generates a subplot grid for sales data with missing sales (0 values) excluded from the line plot.
    """
    # Create a subplot grid
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=sales_data.columns)

    # Get index and columns
    index_values = sales_data.index
    columns = sales_data.columns

    # Loop through each column and add a trace
    for i, column in enumerate(columns):
        row = (i // cols) + 1
        col = (i % cols) + 1

        # Replace zero sales with None to prevent plotting unwanted lines
        y_values = [val if val > 0 else None for val in sales_data[column]]

        fig.add_trace(
            go.Scatter(
                x=index_values, 
                y=y_values, 
                line=dict(dash='dot', color='black'), 
                marker=dict(color='black', size=4),
                name=column
            ),
            row=row,
            col=col
        )

    # Update layout
    fig.update_layout(
        height=1200, width=900,  
        showlegend=False,  
        title_text=title,
    )

    # Show figure
    fig.show()

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
        title='Total sales revenue of each product',
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
    fig, ax = plt.subplots(figsize=(12, 6))

    yearly_data.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", linewidth=0.6, zorder=2)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel("Year", fontsize=16, fontweight="bold", labelpad=15)
    ax.set_ylabel("Value", fontsize=16, fontweight="bold", labelpad=15)
    ax.set_title("Sales per product per year", fontsize=18, fontweight="bold", pad=20)
    plt.xticks(rotation=0, ha="right", fontsize=12)

    ax.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12, title_fontsize=14, frameon=True, facecolor='white', edgecolor='black')

    for rect in ax.patches:
        rect.set_zorder(1)
        rect.set_edgecolor('black')

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  
    ax.ticklabel_format(style='plain', axis='y')  

    # Add comma as thousands separator to y-axis values
    def comma_format(x, pos):
        return f'{x:,.0f}'  

    ax.yaxis.set_major_formatter(FuncFormatter(comma_format))

    plt.tight_layout()
    plt.show()

# Monthly sales line plot
def monthly_sales_per_year(monthly_sales):
    """Generates an interactive Plotly figure for monthly sales with a year selection dropdown."""
    
    # Extract unique years
    years = monthly_sales.index.year.unique()
    
    # Create figure
    fig = go.Figure()
    traces, visibility_options = [], []

    # Add traces per year
    for i, year in enumerate(years):
        yearly_data = monthly_sales[monthly_sales.index.year == year]
        
        for column in yearly_data.columns:
            fig.add_trace(go.Scatter(
                x=yearly_data.index,
                y=yearly_data[column].replace(0, None),  
                mode='lines',
                name=f"{column} ({year})",
                visible=(i == 0)
            ))
            traces.append(f"{column} ({year})")

    # Create dropdown menu
    for i, year in enumerate(years):
        visibility = [f"({year})" in name for name in traces]
        visibility_options.append(dict(label=str(year), method="update",
                                       args=[{"visible": visibility}, {"title": f"Monthly Sales in {year}"}]))

    # Update layout
    fig.update_layout(
        xaxis=dict(tickformat="%b %Y", dtick="M1", tickangle=-45, tickmode="array", tickvals=monthly_sales.index),
        updatemenus=[{"buttons": visibility_options, "direction": "down", "showactive": True, "x": 0.1, "y": 1.15}],
        title=f"Monthly Sales in {years[0]}",
        xaxis_title="Date",
        yaxis_title="Sales",
        legend_title="Product",
        template="plotly_white"
    )

    return fig

def monthly_sales(monthly_sales_resampled):
    fig = go.Figure()

    # Add traces for each 'Mapped_GCK', skipping zero sales
    for column in monthly_sales_resampled.columns:
        fig.add_trace(go.Scatter(
            x=monthly_sales_resampled.index,
            y=monthly_sales_resampled[column].replace(0, None),
            mode='lines',
            name=column
        ))

    # Update layout with all months on x-axis and no grid lines
    fig.update_layout(
        title='Monthly Sales Over Time',
        xaxis_title='Date',
        yaxis_title='Sales',
        legend_title='Product',
        xaxis=dict(
            tickangle=45,
            tickmode='array',
            tickvals=monthly_sales_resampled.index,
            ticktext=[date.strftime('%b %Y') for date in monthly_sales_resampled.index],
            showgrid=False
        ),
        yaxis=dict(showgrid=False),
        template='plotly_white'
    )

    fig.show()

# Subplots of line plots (percentage change)
def percentage_change(monthly_sales_resampled, num_rows=5, num_cols=3):
    num_plots = num_rows * num_cols
    fig = make_subplots(rows=num_rows, cols=num_cols)

    # Loop through columns (products) and create subplots
    for i, product in enumerate(monthly_sales_resampled.columns[:num_plots]):
        row = i // num_cols + 1
        col = i % num_cols + 1
        
        # Calculate percentage change
        previous_month_sales = monthly_sales_resampled[product].shift(1)
        percentage_change = ((monthly_sales_resampled[product] - previous_month_sales) / previous_month_sales) * 100
        
        # Add the trace to the subplot
        fig.add_trace(go.Scatter(x=monthly_sales_resampled.index, y=percentage_change, mode='lines', name=product, line=dict(color=main_color)),
                      row=row, col=col)
        
        fig.update_yaxes(title_text=f"Change (%) - {product}", row=row, col=col)

    # Update layout and title for the figure
    fig.update_layout(
        height=900, 
        width=1200, 
        title_text="Percentage Change in Sales for Each Product",
        showlegend=False,
    )

    # Show the plot
    fig.show()

# Subplots of percentage change previuos month
def percentage_change_previous_month(monthly_sales_resampled, num_rows=5, num_cols=3):
    num_plots = num_rows * num_cols
    fig = make_subplots(rows=num_rows, cols=num_cols)

    percentage_change_df = monthly_sales_resampled.pct_change(periods=12) * 100

    for i, product in enumerate(monthly_sales_resampled.columns[:num_plots]):
        row = i // num_cols + 1
        col = i % num_cols + 1
        
        fig.add_trace(go.Scatter(x=monthly_sales_resampled.index, y=percentage_change_df[product], mode='lines', name=product, line=dict(color=main_color)),
                      row=row, col=col)
        
        fig.update_yaxes(title_text=f"Change (%) - {product}", row=row, col=col)

    fig.update_layout(
        height=900, 
        width=1200, 
        title_text="Percentage change in sales for each product compared to the same month of previous year",
        showlegend=False
    )

    fig.show()


# Multiplicative
def multiplicative_seasonal_decomposition(df, excluded_columns=None, period=12):
    """
    Creates an interactive dropdown for seasonal decomposition analysis.

    Parameters:
    - df: DataFrame containing time-series data.
    - excluded_columns: List of columns to exclude from selection.
    - period: Seasonal period for decomposition.

    Returns:
    - A widget with a dropdown menu and dynamic seasonal decomposition plots.
    """

    if excluded_columns is None:
        excluded_columns = []

    # Select available columns
    available_columns = [col for col in df.columns if col not in excluded_columns]

    if not available_columns:
        raise ValueError("No available columns after exclusions!")

    # Dropdown widget
    dropdown = widgets.Dropdown(
        options=available_columns,
        value=available_columns[0],  
        description='Select Column:',
        style={'description_width': 'initial'}
    )

    # Output widget
    output = widgets.Output()

    # Function to update the graph dynamically
    def update_graph(change):
        with output:
            clear_output(wait=True) 
            selected_column = dropdown.value
            result = seasonal_decompose(df[selected_column], model='multiplicative', period=period, extrapolate_trend='freq')

            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(12, 8))
            
            result.observed.plot(ax=axes[0], title='Observed')
            result.trend.plot(ax=axes[1], title='Trend')
            result.seasonal.plot(ax=axes[2], title='Seasonal')

            # Residuals as scatter points
            axes[3].scatter(df.index, result.resid, color='red', s=10)  # Red points with size 10
            axes[3].set_title('Residuals')

            plt.suptitle(f'Seasonal Decomposition for {selected_column}', fontsize=14)
            plt.tight_layout()
            plt.show()

    # Attach function to dropdown changes
    dropdown.observe(update_graph, names='value')

    # Display widgets
    display(dropdown, output)

    # Initial plot
    update_graph(None)

# Addictive
def additive_seasonal_decomposition(df, excluded_columns=None, period=12):
    """
    Creates an interactive dropdown for additive seasonal decomposition analysis.

    Parameters:
    - df: DataFrame containing time-series data.
    - excluded_columns: List of columns to exclude from selection.
    - period: Seasonal period for decomposition.

    Returns:
    - A widget with a dropdown menu and dynamic seasonal decomposition plots.
    """

    if excluded_columns is None:
        excluded_columns = []

    # Select  columns
    available_columns = [col for col in df.columns if col not in excluded_columns]

    if not available_columns:
        raise ValueError("No available columns after exclusions!")

    # Dropdown widget
    dropdown = widgets.Dropdown(
        options=available_columns,
        value=available_columns[0],  # Default selection
        description='Select Column:',
        style={'description_width': 'initial'}
    )

    # Output widget
    output = widgets.Output()

    # Function to update the graph dynamically
    def update_graph(change):
        with output:
            clear_output(wait=True)  
            selected_column = dropdown.value
            result = seasonal_decompose(df[selected_column], model='additive', period=period, extrapolate_trend='freq')

            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(12, 8))
            
            result.observed.plot(ax=axes[0], title='Observed')
            result.trend.plot(ax=axes[1], title='Trend')
            result.seasonal.plot(ax=axes[2], title='Seasonal')

            # Residuals as scatter points
            axes[3].scatter(df.index, result.resid, color='red', s=10)  # Red points with size 10
            axes[3].set_title('Residuals')

            plt.suptitle(f'Additive Seasonal Decomposition for {selected_column}', fontsize=14)
            plt.tight_layout()
            plt.show()

    # Attach function to dropdown changes
    dropdown.observe(update_graph, names='value')

    # Display widgets
    display(dropdown, output)

    # Initial plot
    update_graph(None)

# Subplots of line plots - resource prices
def resource_prices(market_data_resampled, resources_prices):
    fig = make_subplots(
        rows=len(resources_prices),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=resources_prices
    )

    for i, resource in enumerate(resources_prices):
        fig.add_trace(
            go.Scatter(
                x=market_data_resampled.index,
                y=market_data_resampled[resource],
                mode='lines',
                name=resource
            ),
            row=i+1,
            col=1
        )
        fig.update_yaxes(
            range=[0, market_data_resampled[resource].max()],
            row=i+1,
            col=1
        )

        # Add a red dotted line at y=100 in each subplot
        fig.add_shape(
            type='line',
            x0=market_data_resampled.index.min(),
            x1=market_data_resampled.index.max(),
            y0=100,
            y1=100,
            line=dict(color='red', width=2, dash='dot'),  # Dash style set to 'dot'
            row=i+1,
            col=1
        )

    fig.update_layout(
        title="Resource Prices Over Time (2004-2022)",
        xaxis_title='Year',
        yaxis_title='Price',
        height=1000,
        showlegend=False
    )

    fig.update_xaxes(showticklabels=True, row=len(resources_prices), col=1)
    for i in range(len(resources_prices)-1):
        fig.update_xaxes(showticklabels=True, row=i+1, col=1)

    fig.show()

def plot_prod_ship_index(prod_ship_index):
    rows, cols = 6, 3
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=prod_ship_index.columns)
    columns = prod_ship_index.columns
    index_values = prod_ship_index.index

    for i, column in enumerate(columns):
        row = (i // cols) + 1
        col = (i % cols) + 1
        max_value = prod_ship_index[column].max()

        fig.add_trace(
            go.Scatter(
                x=index_values, 
                y=prod_ship_index[column], 
                mode='lines', 
                line=dict(color=main_color), 
                name=column
            ),
            row=row,
            col=col
        )

        fig.update_yaxes(range=[0, max_value], row=row, col=col)

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=index_values[0], x1=index_values[-1],
                    y0=100, y1=100,
                    line=dict(color="red", width=2, dash="dot"),
                    xref=f"x{(r-1)*cols + c}",  
                    yref=f"y{(r-1)*cols + c}",
                )
            )

    fig.update_layout(
        height=1400, width=1100,
        showlegend=False,
        title_text="Production and Shipping Index",
    )

    fig.show()


def plot_filtered_prod_ship_index(prod_ship_index):
    rows, cols = 6, 3
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=prod_ship_index.columns)
    columns = prod_ship_index.columns
    index_values = prod_ship_index.index
    years = index_values.year
    filtered_index = index_values[(years >= 2018) & (years <= 2022)]

    for i, column in enumerate(columns):
        row = (i // cols) + 1
        col = (i % cols) + 1
        max_value = prod_ship_index.loc[filtered_index, column].max()

        fig.add_trace(
            go.Scatter(
                x=filtered_index,
                y=prod_ship_index.loc[filtered_index, column], 
                mode='lines',
                line=dict(dash='dot', color=main_color),
                name=column
            ),
            row=row,
            col=col
        )

        fig.update_yaxes(range=[0, max_value + 100], row=row, col=col)

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=filtered_index.min(), x1=filtered_index.max(),
                    y0=100, y1=100,
                    line=dict(color="red", width=2, dash="dot"),
                    xref=f"x{(r-1)*cols + c}",
                    yref=f"y{(r-1)*cols + c}",
                )
            )

    fig.update_layout(
        height=1400, width=1100,
        showlegend=False,
        title_text="Production and Shipping Index (2018-2022)",
        xaxis=dict(range=[filtered_index.min(), filtered_index.max()], title="Year"),
    )

    fig.show()

def plot_production_index(production_data, production_columns):
    traces = []

    for country in production_columns:
        trace = go.Scatter(
            x=production_data.index,  
            y=production_data[country],  
            mode='lines', 
            name=country 
        )
        traces.append(trace)

    layout = go.Layout(
        title="Production Index Over Time by Country",
        xaxis=dict(title='Year'),
        yaxis=dict(title='Production Index'),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_shipment_index(shipment_data, shipment_columns):
    traces = []

    for country in shipment_columns:
        trace = go.Scatter(
            x=shipment_data.index,  
            y=shipment_data[country],  
            mode='lines', 
            name=country 
        )
        traces.append(trace)

    layout = go.Layout(
        title="Shipment Index Over Time by Country",
        xaxis=dict(title='Year'),
        yaxis=dict(title='Shipment Index'),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_producer_price(producer_price_data, producer_prices):
    traces = []

    for country in producer_prices:
        trace = go.Scatter(
            x=producer_price_data.index,  
            y=producer_price_data[country],  
            mode='lines', 
            name=country 
        )
        traces.append(trace)

    layout = go.Layout(
        title="Producer Price Over Time by Country",
        xaxis=dict(title='Year'),
        yaxis=dict(title='Production Price'),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_me_index(me_index_data, me_index):
    traces = []

    for country in me_index:
        country_data = me_index_data[country].dropna()

        trace = go.Scatter(
            x=country_data.index,  
            y=country_data,  
            mode='lines', 
            name=country 
        )
        traces.append(trace)

    layout = go.Layout(
        title="Machinery & Equipment Index Over Time by Country",
        xaxis=dict(title='Year'),
        yaxis=dict(title='Index'),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_ee_index(ee_index_data, ee_index):
    traces = []

    for country in ee_index:
        country_data = ee_index_data[country].dropna()

        trace = go.Scatter(
            x=country_data.index,  
            y=country_data,  
            mode='lines', 
            name=country 
        )
        traces.append(trace)

    layout = go.Layout(
        title="Electronic Index Over Time by Country",
        xaxis=dict(title='Year'),
        yaxis=dict(title='Index'),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

# Outliers
def plot_distribution_and_boxplot(df, column_name, n_bins, out_left=None, out_right=None, color=main_color):
    """
    Plots the histogram and box plot for a specific column with optional outlier boundaries.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): Column to visualize.
        n_bins (int): Number of bins for the histogram.
        out_left (float, optional): Left boundary to exclude outliers. If None, no line is drawn.
        out_right (float, optional): Right boundary to exclude outliers. If None, no line is drawn.
        color (str): Plot color.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    sns.histplot(df[column_name], kde=True, bins=n_bins, color=color, ax=axes[0])
    axes[0].set_title(f"Distribution of {column_name}")
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel("Frequency")

    # Boxplot
    sns.boxplot(x=df[column_name], color=color, ax=axes[1])
    axes[1].set_title(f"Boxplot of {column_name}")
    axes[1].set_xlabel(column_name)

    # Add vertical lines for outlier boundaries in the boxplot only if values are provided
    if out_left is not None:
        axes[1].axvline(x=out_left, color='red', linestyle='-', linewidth=1)
    if out_right is not None:
        axes[1].axvline(x=out_right, color='red', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.show()

# Create Semester column
def get_semester(month):
    if month <= 6:
        return 1  # Semester 1 (January to June)
    else:
        return 2  # Semester 2 (July to December)

# Create Quarter column
def get_quarter(month):
    if month in [1, 2, 3]:
        return 'Q1'  # Quarter 1 (January to March)
    elif month in [4, 5, 6]:
        return 'Q2'  # Quarter 2 (April to June)
    elif month in [7, 8, 9]:
        return 'Q3'  # Quarter 3 (July to September)
    else:
        return 'Q4'  # Quarter 4 (October to December)
    
def count_weekends_in_month(year, month):
    # Generate all the dates for the given month
    first_day = pd.Timestamp(f'{year}-{month:02d}-01')
    last_day = pd.Timestamp(f'{year}-{month:02d}-01') + pd.offsets.MonthEnd()
    all_dates_in_month = pd.date_range(first_day, last_day)
    
    # Count the number of weekends (Saturday=5, Sunday=6)
    weekend_count = sum(all_dates_in_month.weekday.isin([5, 6]))
    return weekend_count, len(all_dates_in_month)

# Count Sundays in the month
def count_sundays_in_month(year, month):
    first_day = pd.Timestamp(f'{year}-{month:02d}-01')
    last_day = pd.Timestamp(f'{year}-{month:02d}-01') + pd.offsets.MonthEnd()
    all_dates_in_month = pd.date_range(first_day, last_day)
    
    # Count Sundays (weekday=6)
    sunday_count = sum(all_dates_in_month.weekday == 6)
    return sunday_count

def count_holidays_in_month(data, year, month):
    german_holidays = holidays.Germany(years=data.index.year.unique())
    # Generate holidays in the given month
    holidays_in_month = [day for day in german_holidays if day.year == year and day.month == month]
    return len(holidays_in_month)

def create_lag_features(df, lag_dict):    
    # Create lag features for each product
    for product, lags in lag_dict.items():
        for lag in lags:
            df[f"{product}_Lag_{lag}"] = df[product].shift(lag)
    
    return df

def create_rolling_mean_features(df, roll_dict):

    for product, windows in roll_dict.items():
        if product in df.columns:
            for window in windows:
                df[f"{product}_RollingMean_{window}"] = df[product].rolling(window).mean()

    return df
