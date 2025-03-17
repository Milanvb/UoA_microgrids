import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the processed gap data
file_path = "C:/Users/20193915/OneDrive - TU Eindhoven/UoA-INTERNSHIP/UoA_CODE/load_data_gap_info.csv"
gap_data = pd.read_csv(file_path)

# Extract the directory path to save the plots
output_folder = r"C:\Users\20193915\OneDrive - TU Eindhoven\UoA-INTERNSHIP\UoA_CODE\Outputs_figures"

# Set default font weight to bold
plt.rcParams['font.weight'] = 'bold'

# Ensure columns for gap counts are numeric
gap_categories = ['n small gaps', 'n medium gaps', 'n large gaps', 'n huge gaps']
gap_data[gap_categories] = gap_data[gap_categories].apply(pd.to_numeric, errors='coerce')

# Bar Chart 1: Total Gaps per Category
total_gaps = gap_data[gap_categories].sum()
plt.figure(figsize=(8, 5))
bars = plt.bar(total_gaps.index, total_gaps.values)
plt.title('Total Gaps per Category Among All Households', fontweight='bold')
plt.ylabel('Number of Gaps', fontweight='bold')
plt.xlabel('Gap Category', fontweight='bold')

# Annotate bars with counts
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')

plt.tight_layout()
chart1_path = os.path.join(output_folder, "total_gaps_per_category.png")
plt.savefig(chart1_path)

# Bar Chart 2: Count of Households per Gap Category
# Count households with at least one gap in each category
households_with_gaps = (gap_data[gap_categories] > 0).sum()
plt.figure(figsize=(8, 5))
bars = plt.bar(households_with_gaps.index, households_with_gaps.values)
plt.title('Households with Gaps per Category', fontweight='bold')
plt.ylabel('Number of Households', fontweight='bold')
plt.xlabel('Gap Category', fontweight='bold')

# Annotate bars with counts
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')

plt.tight_layout()
chart2_path = os.path.join(output_folder, "households_with_gaps_per_category.png")
plt.savefig(chart2_path)

print(f"Charts saved to:\n1. {chart1_path}\n2. {chart2_path}")
