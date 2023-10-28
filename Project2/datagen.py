import random
import csv


# Function to generate random data
def generate_data(product_id):
    competitor_price = random.randint(400, 800)
    historical_price = random.randint(400, 800)
    demand = random.choice(['High', 'Medium', 'Low'])
    customer_satisfaction = random.choice(['High', 'Medium', 'Low'])
    time_of_day = random.choice(['Morning', 'Afternoon', 'Evening', 'Night'])
    day_of_week = random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    price = random.randint(40, 80)
    return [product_id, competitor_price, historical_price, demand, customer_satisfaction, time_of_day, day_of_week, price]

# Number of rows to generate
num_rows = 10000

# Generate data and write to CSV file
with open('pricing_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['ProductID', 'CompetitorPrice', 'HistoricalPrice', 'Demand', 'CustomerSatisfaction', 'TimeOfDay', 'DayOfWeek', 'Price'])
    # Write data rows
    for i in range(1, num_rows + 1):
        data_row = generate_data(i)
        writer.writerow(data_row)

print(f'{num_rows} rows of data have been generated and saved to pricing_data.csv.')
