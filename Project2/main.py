from datagen import generate_data
from DQN import train



print("Generating Database.........")
generate_data(1000)
print("Data Generated!!!")

print("Model Training......")
train()
