import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
file_path = '/Users/william/Desktop/code/3_personal_projects/3_inprogress/MachineLearningProjects/McCallHousing/data.csv'  # Adjust the path if needed
data = pd.read_csv(file_path)

# Columns to keep
columns_to_keep = [
    'City', 'Subdivision', 'Acres', 'Paved Street', 'Winter Access', 
    'Bedrooms', 'Total Baths', 'Apx Fin SqFt', 'Age', 
    'Garage', 'Carport', 'Foundation', 'Heat', 'Sewer', 
    'Water', 'Sold Price', 'Closed Date', 'Days on Market'
]

# Drop columns that are not in the list of columns to keep
filtered_data = data[columns_to_keep]

# Fill empty cells with 'Unknown'
filtered_data = filtered_data.fillna('Unknown')

# Drop rows where 'Sold Price' is 'Unknown'
filtered_data = filtered_data[filtered_data['Sold Price'] != 'Unknown']

# One-hot encode the 'City' column
filtered_data = pd.get_dummies(filtered_data, columns=['City'])

# Label encode the 'Subdivision' column
label_encoder = LabelEncoder()

# Encode 'Subdivision', 'Age', 'Foundation', 'Heat', 'Sewer', and 'Water'
columns_to_label_encode = ['Subdivision', 'Age', 'Foundation', 'Heat', 'Sewer', 'Water']

for column in columns_to_label_encode:
    filtered_data[column] = label_encoder.fit_transform(filtered_data[column])

# Convert 'Garage' to integers, with "Five or more" as 5
garage_mapping = {
    'None': 0,
    'Unknown': 0,
    'One': 1,
    'Two': 2,
    'Three': 3,
    'Four': 4,
    'Five or More': 5
}
# Convert 'Carport' to integers, with "Five or more" as 5
carport_mapping = {
    'None': 0,
    'Unknown': 0,
    'One': 1,
    'Two': 2,
    'Three': 3,
    'Four': 4,
    'Five or More': 5
}

filtered_data['Garage'] = filtered_data['Garage'].replace(garage_mapping).astype(int)
filtered_data['Carport'] = filtered_data['Carport'].replace(garage_mapping).astype(int)

# Encode 'Paved Street' and 'Winter Access' as binary (0 or 1)
binary_columns = ['Paved Street', 'Winter Access']
for column in binary_columns:
    filtered_data[column] = filtered_data[column].apply(lambda x: 1 if x.lower() == 'yes' else 0)


# Encode 'Closed Date' by extracting year, month, and day
filtered_data['Closed Date'] = pd.to_datetime(filtered_data['Closed Date'], errors='coerce')
filtered_data['Closed Year'] = filtered_data['Closed Date'].dt.year.fillna(0).astype(int)
filtered_data['Closed Month'] = filtered_data['Closed Date'].dt.month.fillna(0).astype(int)
filtered_data['Closed Day'] = filtered_data['Closed Date'].dt.day.fillna(0).astype(int)
filtered_data.drop(columns=['Closed Date'], inplace=True)

# Save the encoded DataFrame to a new CSV file
filtered_data.to_csv('encoded_data.csv', index=False)

print("One-hot encoded data saved to 'encoded_data.csv'")