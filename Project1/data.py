import pandas as pd
from sqlalchemy import create_engine
import dateparser
import numpy as np


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 11)

# Credentials to connect to the database
username = "username"
password = "DB_password"
hostname = "DB_host"
dbname = "DB_name"


# Process the initial data
def first_data_handle():
    # Connect to the database of the e-shop
    # engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
    #                        .format(user=username,
    #                                password=password,
    #                                host=hostname,
    #                                dbname=dbname))

    # Read the data
    data = pd.read_csv('sample_data_updated.csv')
    # Round the prices so as to have two decimals
    data.product_price = round(data.product_price, 2)
    # Find the data with zero product_price
    zero_price_data = data[data.product_price == 0.0]
    # Remove the zero price data
    data.drop(zero_price_data.index, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Write the processed data to CSV
    data.to_csv("processed_sales.csv", index=False)


# Process the data and find the products that were purchased more than num_of_sales


# Process the data and find the products that were purchased more than num_of_sales

def products_with_sales(num_of_sales=10):
    try:
        data = pd.read_csv("processed_sales.csv")
        if data.empty:
            print("Error: The dataset is empty.")
            return
    except FileNotFoundError:
        print("Error: File not found. Make sure the file 'processed_sales.csv' exists in the correct path.")
        return

    # Check if 'order_timestamp' column exists in the data
    if 'order_timestamp' not in data.columns:
        print("Error: 'order_timestamp' column not found in the dataset.")
        return

    # Parse 'order_timestamp' column into separate date and time columns with a specified format
    try:
        data['order_date'] = data['order_timestamp'].apply(lambda x: dateparser.parse(x).strftime('%Y-%m-%d'))
        data['order_time'] = data['order_timestamp'].apply(lambda x: dateparser.parse(x).strftime('%H:%M:%S'))
    except Exception as e:
        print(f"Error parsing 'order_timestamp' column: {e}")
        return

    # Check if there are any NaN or empty values in 'order_date' or 'order_time' columns
    if data['order_date'].isnull().any() or data['order_time'].eq('').any():
        print("Error: 'order_date' or 'order_time' contains NaN or empty values.")
        return

    # Find the products that were purchased more than num_of_sales times
    products_with_high_vol = data.groupby(["product_id"])["product_quantity"].sum()
  
    products_with_high_vol = products_with_high_vol[products_with_high_vol >= num_of_sales].index
    #print(products_with_high_vol)
    
    data = data[data["product_id"].isin(products_with_high_vol)]
    
    # Find in which week the products were purchased compared to the date of the first order of the dataset
    data.sort_values(by="order_timestamp", inplace=True)
    first_date = dateparser.parse(data.iloc[0]['order_timestamp'])
    last_date = dateparser.parse(data.iloc[-1]['order_timestamp'])
    shift = 6 - ((last_date - first_date).days % 7)
    data["week"] = 0
    weeks = []
    for i in range(data.shape[0]):
        date = dateparser.parse(data.iloc[i]['order_timestamp'])
        week = (((date - first_date).days + shift) // 7) + 1
        weeks.append(week)
    data["week"] = weeks

    data.reset_index(drop=True, inplace=True)
    
    # Write the processed data to CSV
    data.to_csv(f"products_{num_of_sales}_sales.csv", index=False)

   # print(f"Processed data has been saved to products_{num_of_sales}_sales.csv")




# Data aggregation. Collect the data in a weekly basis
def create_week_data():
    # Connect to the database of the e-shop
    # engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
    #                        .format(user=username,
    #                                password=password,
    #                                host=hostname,
    #                                dbname=dbname))

    # Read the data
    data = pd.read_csv("products_10_sales.csv")
    data["order_timestamp"] = data["order_timestamp"].astype("str")
    # Demand per week
    demand_data = data.groupby(["product_id", "week"]).product_quantity.sum()
    # Mean price per week
    price_data = data.groupby(["product_id", "week"]).product_price.mean().round(2)
    week_data = pd.concat([demand_data, price_data], axis=1)
    week_data = week_data.reset_index(level=["product_id", "week"])
    # Assume a cost for each product based on the minimum of the price
    products = week_data.product_id.unique()
    cost = pd.Series()
    max_prices = pd.Series()
    for product in products:
        min_price = week_data.loc[week_data.product_id == product].product_price.min()
        max_price = week_data.loc[week_data.product_id == product].product_price.max()
        temp_ind = week_data.loc[week_data.product_id == product].index
        for i in temp_ind:
            cost.loc[i] = round(0.8 * min_price, 2)
            max_prices.loc[i] = round(1.2 * max_price, 2)

    week_data["product_cost"] = cost
    week_data["product_max_bound"] = max_prices

    week_data.to_csv("week_data.csv")


def full_weeks(missing_weeks, total_weeks):
    '''
    Return a dataframe with full weeks of our dataset
    Take one product and fill the empty weeks with zeros
    '''
    full_weeks = pd.DataFrame(columns=missing_weeks.columns)
    #print(missing_weeks)

    for week in range(1, total_weeks + 1):
        flag = True
        #print("testing flag: ", missing_weeks.loc[missing_weeks["week"] == week].week.values)
        if not missing_weeks.loc[missing_weeks["week"] == week].week.values:
            flag = False
        temp = []
        temp.append(missing_weeks.loc[0]["product_id"])
        temp.append(week)
        # print("temp ha: ", temp)
        # print(missing_weeks.loc[0])

        if flag:
            #print("Yaha aya")
            temp.append(missing_weeks.loc[missing_weeks["week"] == week].product_quantity.values[0])
            temp.append(missing_weeks.loc[missing_weeks["week"] == week].product_price.values[0])
        else:
            temp.extend([0, 0])
        # print("yaha wala temp: ", temp)
        temp.append(missing_weeks.loc[0]["product_cost"])
        temp.append(missing_weeks.loc[0]["product_max_bound"])
        temp.insert(0,week)
        # print("append ke baad wala temp: ", temp)
        # print("full weeks: ", full_weeks)
        # print("length of full weeks: ",len(full_weeks))
        # Check if the index is within bounds before accessing it
        if week - 1 < len(full_weeks):
            full_weeks.iloc[week - 1] = temp
        else:
            #print(temp)
            full_weeks.loc[len(full_weeks)] = temp
    return full_weeks.copy()



def nn_row(row, full_weeks, number_of_weeks):
    '''
        Return a row in the desired format for the neural network
    '''
    row_data = []
    row_data.append(row["week"])
    row_data.append(row["product_cost"])
    row_data.append(row["product_max_bound"])
    row_data.append(row["product_id"])
    week = row["week"]
    weeks = np.ndarray(shape=(number_of_weeks, 2))
    for i in range(1, number_of_weeks+1):
        temp_week = week - i
        if temp_week < 1:
            p = 0
            q = 0
        else:
            p = full_weeks.loc[full_weeks["week"] == temp_week].product_price.values[0]
            q = full_weeks.loc[full_weeks["week"] == temp_week].product_quantity.values[0]
        weeks[(number_of_weeks - i), 0] = p
        weeks[(number_of_weeks - i), 1] = q

    for i in range(number_of_weeks):
        row_data.append(weeks[i, 0])
        row_data.append(weeks[i, 1])

    row_data.append(row["product_price"])
    row_data.append(row["product_quantity"])
    return row_data


# Process the data to be in the desired format for the neural network
def create_nn_data(number_of_weeks=16):
    # Connect to the database of the e-shop
    # engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
    #                        .format(user=username,
    #                                password=password,
    #                                host=hostname,
    #                                dbname=dbname))

    week_data = pd.read_csv("week_data.csv")
    total_weeks = week_data.week.max()
    #print("total week: ",total_weeks)

    columns = ["week", "product_cost", "product_max_bound", "product_id"]
    for i in range(1, number_of_weeks+2):
        columns.append("P{:d}".format(i))
        columns.append("Q{:d}".format(i))
    
    nn_data = pd.DataFrame(columns=columns)
    #print(type(nn_data))
    #print("nn_data hai yeh:",nn_data)
    products = week_data.product_id.unique()
    #print(products)
    for product in products:
        temp_product = week_data.loc[week_data["product_id"] == product]
        # print(temp_product.shape[0])
        temp_product.index = range(temp_product.shape[0])
        # print(temp_product.copy())
        full_week = full_weeks(temp_product.copy(), total_weeks)

        temp_data = pd.DataFrame(columns=columns)
        for index, row in temp_product.iterrows():
            temp_data.loc[index] = nn_row(row, full_week.copy(), number_of_weeks)
        temp = []
        temp.append(temp_data)
        nn_data = pd.concat(temp, ignore_index=True)

    nn_data.to_csv("nn_data.csv")


# Process the data to be in the desired format for the particle swarm optimization
def pso_data():
    # Connect to the database of the e-shop
    # engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
    #                        .format(user=username,
    #                                password=password,
    #                                host=hostname,
    #                                dbname=dbname))

    nn_data = pd.read_csv("nn_data.csv")
    total_weeks = nn_data.week.max()
    number_of_weeks = int((nn_data.shape[1] - 6) / 2)

    data = nn_data.loc[nn_data["week"] == total_weeks].copy()
    #print(data)
    data.index = range(len(data))

    for i in range(1, number_of_weeks+1):
        data.loc[:, f"P{i}"] = data[f"P{i+1}"]
        data.loc[:, f"Q{i}"] = data[f"Q{i+1}"]
    data.drop(columns=["week", f"P{number_of_weeks+1}", f"Q{number_of_weeks+1}"], inplace=True)
    pso = pd.DataFrame(columns=data.columns)
    pso["product_min_bound"] = 0

    # Read the data to see which products will be priced dynamically
    opt_data = pd.read_csv("sample_data_updated.csv")
    products = opt_data["product_id"]
    for product in products:
        temp = data.loc[data["product_id"] == product].copy()
        initial_price = opt_data.loc[opt_data["product_id"] == product].initial_price.values[0]  # Assuming initial_price is a scalar
        final_price = opt_data.loc[opt_data["product_id"] == product].final_price.values[0]      # Assuming final_price is a scalar
        percentage = 1 - (final_price / initial_price)
        max_value = (1 - (percentage - 0.1)) * initial_price
        max_scalar = max_value.item() if isinstance(max_value, np.ndarray) else max_value
        if max_scalar > initial_price:
            max_scalar = initial_price
        min_value = (1 - (percentage + 0.1)) * initial_price
        min_scalar = min_value.item() if isinstance(min_value, np.ndarray) else min_value
        
        temp.loc[:, "product_max_bound"] = round(float(max_scalar), 2)
        temp["product_min_bound"] = round(float(min_scalar), 2)
        pso = pd.concat([pso, temp])

    pso.index = range(len(pso))

    pso.to_csv("pso_data.csv")

# Helpful function to see the details of a dataframe
def print_details(data):
    print("Number of customers: {}".format(data.customer_id.nunique()))
    print("Number of orders: {}".format(data.order_id.nunique()))
    print("Number of products: {}".format(data.product_id.nunique()))
    print("First date: {}".format(data.order_timestamp.min()))
    print("Last date: {}".format(data.order_timestamp.max()))
    print("Columns in data:")
    print(list(data.columns))
    print(data.shape)
    print(data.head(10))
    print(data.tail(10))

# data = pd.read_csv('sample_data.csv')
# print_details(data)