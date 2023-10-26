import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta

# Generate random strings for IDs
def generate_random_string(length=3):
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for _ in range(length))

# Generate random datetime within a specific range
def generate_random_datetime(start_date, end_date):
    return start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds())))

# Generate sample data
def generate_sample_data(num_rows=1000):
    products = ['P' + generate_random_string() for _ in range(num_rows)]
    prices = [round(random.uniform(10, 100), 2) for _ in range(num_rows)]
    quantities = [random.randint(1, 10) for _ in range(num_rows)]
    customers = ['C' + generate_random_string() for _ in range(num_rows)]
    orders = ['O' + generate_random_string() for _ in range(num_rows)]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    timestamps = [generate_random_datetime(start_date, end_date).strftime('%Y-%m-%d %H:%M:%S') for _ in range(num_rows)]

    data = {
        'product_id': products,
        'product_price': prices,
        'product_quantity': quantities,
        'customer_id': customers,
        'order_id': orders,
        'order_timestamp': timestamps
    }

    return pd.DataFrame(data)

# Generate and save sample data to CSV
sample_data = generate_sample_data(num_rows=1000)
sample_data.to_csv('sample_data.csv', index=False)

# Add initial_price and final_price columns
sample_data['initial_price'] = sample_data['product_price']
sample_data['final_price'] = sample_data.apply(lambda row: round(row['product_price'] * random.uniform(0.8, 1.2),2), axis=1)

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
# Add 50-100 instances where product_quantity is greater than num_of_sales
num_of_instances = random.randint(50, 100)
new_data = []
for _ in range(num_of_instances):
    product_id = 'P' + generate_random_string()
    product_price = round(random.uniform(10, 100), 2)
    product_quantity = random.randint(10, 30)  # Ensure quantity is greater than num_of_sales
    customer_id = 'C' + generate_random_string()
    order_id = 'O' + generate_random_string()
    order_timestamp = generate_random_datetime(start_date, end_date).strftime('%Y-%m-%d %H:%M:%S')
    initial_price = product_price
    final_price = round(product_price * random.uniform(0.8, 1.2),2)  # Random final price within 80-120% of initial price
    
    new_instance = {
        'product_id': product_id,
        'product_price': product_price,
        'product_quantity': product_quantity,
        'customer_id': customer_id,
        'order_id': order_id,
        'order_timestamp': order_timestamp,
        'initial_price': initial_price,
        'final_price': final_price
    }
    
    new_data.append(new_instance)

# Convert the list of dictionaries to a DataFrame
new_instances_df = pd.DataFrame(new_data)

# Concatenate the original sample_data and new_instances_df DataFrames
sample_data = pd.concat([sample_data, new_instances_df], ignore_index=True)

# Save the updated sample data to CSV
sample_data.to_csv('sample_data_updated.csv', index=False)