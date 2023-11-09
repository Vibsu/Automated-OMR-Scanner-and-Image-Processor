import pandas as pd
from usn import detect_bubbles_and_get_integrated_values
from subcode import detect_bubbles
from marksdet1 import detect_bubbles_and_get_integrated_values1
from marksdet2 import detect_bubbles_and_get_integrated_values2


imag_path = "gat2.jpg"
usn = detect_bubbles_and_get_integrated_values(imag_path)
print(f"USN: {usn}")
subcode = detect_bubbles(imag_path)
print(f"Subject Code: {subcode}")
marks1 = detect_bubbles_and_get_integrated_values1(imag_path)
print(f"Marks detected 1: {marks1}")
marks2 = detect_bubbles_and_get_integrated_values2(imag_path)
print(f"Marks detected 2: {marks2}")

# Split the marks1 string and store the values in split_marks1 list
split_marks1 = marks1.split(" ")
split_marks2 = marks2.split(" ")
each_total = list()
each_total.append(int(split_marks1[0]) + int(split_marks1[1]) + int(split_marks1[2]))
each_total.append( int(split_marks1[3]) + int(split_marks1[4]) + int(split_marks1[5]))
each_total.append( int(split_marks1[6]) + int(split_marks1[7]) + int(split_marks1[8]))
each_total.append( int(split_marks1[9]) + int(split_marks1[10]) + int(split_marks1[11]))
each_total.append( int(split_marks2[0]) + int(split_marks2[1]) + int(split_marks2[2]))
each_total.append( int(split_marks2[3]) + int(split_marks2[4]) + int(split_marks2[5]))
each_total.append( int(split_marks2[6]) + int(split_marks2[7]) + int(split_marks2[8]))
each_total.append( int(split_marks2[9]) + int(split_marks2[10]) + int(split_marks2[11]))
final_marks=0
final_marks = max(each_total[0],each_total[1]) + max(each_total[2],each_total[3]) + max(each_total[4],each_total[5]) + max(each_total[6],each_total[7])

# Create a DataFrame to hold the data
data = {
    'USN': [usn],
    'Subject Code': [subcode],
    'Q1': each_total[0],
    'Q2': each_total[1],
    'Q3': each_total[2],
    'Q4': each_total[3],
    'Q5': each_total[4],
    'Q6': each_total[5],
    'Q7': each_total[6],
    'Q8': each_total[7],
    'Total Marks': final_marks
}

# Check if split_marks1 has at least 3 elements before accessing its elements


# Create a DataFrame for the new data
new_df = pd.DataFrame(data)

# Load existing data from the Excel file (if it exists)
try:
    existing_df = pd.read_excel("output_data.xlsx")
    result_df = pd.concat([existing_df, new_df], ignore_index=True)
except FileNotFoundError:
    result_df = new_df

# Save the updated DataFrame to the Excel file
with pd.ExcelWriter("output_data.xlsx", engine="openpyxl", mode="w") as writer:
    result_df.to_excel(writer, index=False)

print("Data has been saved to Excel.")
