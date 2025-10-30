# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:47:46 2025

@author: prigm
"""
# -*- coding: utf-8 -*-
"""
Automated Solver with Dynamic A-Matrix and b-Matrix Selection

@author: prigm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# The cases selected represent the first that pass all ASHRAE 140
cases = ["600FF1.08.xlsx", "650FF1.08.xlsx", "900FF1.08.xlsx", "950FF1.08.xlsx"]

def solve_case(file_path, case_name, pass_fail_ranges):
    print(f"Processing case: {case_name}\n{'-' * 40}")

    data = pd.read_excel(file_path, sheet_name="Model_grey-box3")
    coefficients = pd.to_numeric(data.iloc[24:44, 3], errors='coerce').to_numpy()
    
    daily_schedule = data["Daily schedule ventilation"].to_numpy()
    
    #known b matrix
    b_matrix_fixed = data[["Eq1-known term", "Eq2-known term"]].to_numpy()
    X_solutions = []

    # Iteration through each row assigning the correct a matrix based on the ventilation schdedule
    # The replacment of the Boolean switching mentioned
    for i in range(len(b_matrix_fixed)):
        if daily_schedule[i] == 1: # Occuppied ventilation
            A_3x3 = np.array([
                [coefficients[9], -coefficients[7], 0],
                [-coefficients[1], coefficients[10], -coefficients[0]],
                [0, -coefficients[0], coefficients[11]]
            ])
            eq3_term = data["Eq3-known term"].iloc[i]  # Use occupied ventilation
        else: # Unoccuppied ventilation basically 0 
            A_3x3 = np.array([
                [coefficients[9], -coefficients[7], 0],
                [-coefficients[1], coefficients[10], -coefficients[0]],
                [0, -coefficients[0], coefficients[12]]  # Occupied ventilation
            ])
            eq3_term = data["Eq3-known term (different ventilation)"].iloc[i]  # Use different ventilation term

        b_matrix_row = np.array([b_matrix_fixed[i, 0], b_matrix_fixed[i, 1], eq3_term])

        A_3x3_inverse = np.linalg.inv(A_3x3)

        X_solution = np.dot(A_3x3_inverse, b_matrix_row)
        X_solutions.append(X_solution)

    solution_df = pd.DataFrame(X_solutions, columns=["Theta_m", "Theta_sup", "Theta_air"])

    # Calculate the values of indoor air temperature from the data set
    theta_air_values = solution_df["Theta_air"]
    theta_air_min = theta_air_values.min()
    theta_air_max = theta_air_values.max()
    theta_air_avg = theta_air_values.mean()

    # Apply pass/fail conditions
    min_status = "Pass" if pass_fail_ranges["min"][0] <= theta_air_min <= pass_fail_ranges["min"][1] else "Fail"
    max_status = "Pass" if pass_fail_ranges["max"][0] <= theta_air_max <= pass_fail_ranges["max"][1] else "Fail"
    avg_status = "Pass" if pass_fail_ranges["avg"][0] <= theta_air_avg <= pass_fail_ranges["avg"][1] else "Fail"

    print(f"Summary for {case_name}:")
    print(f"Minimum Theta_air: {theta_air_min:.2f}°C ({min_status})")
    print(f"Maximum Theta_air: {theta_air_max:.2f}°C ({max_status})")
    print(f"Average Theta_air: {theta_air_avg:.2f}°C ({avg_status})")
    print("\n")

    return solution_df

# Pass/fail ranges for each case
pass_fail_ranges = {
    "600FF": {"min": (-18.8, -15.6), "max": (64.9, 75.1), "avg": (24.2, 27.4)},
    "650FF": {"min": (-23.05, -21), "max": (63.2, 73.5), "avg": (18, 20.8)},
    "900FF": {"min": (-6.4, -1.6), "max": (41.8, 46.4), "avg": (24.5, 27.5)},
    "950FF": {"min": (-20.2, -17.8), "max": (35.5, 38.5), "avg": (14.0, 15.3)}
}
for case_file in cases:
    case_name = case_file.replace("1.08.xlsx", "")
    solve_case(case_file, case_name, pass_fail_ranges[case_name])
    
# Process each case
cases = ["600FF1.08.xlsx", "650FF1.08.xlsx", "900FF1.08.xlsx", "950FF1.08.xlsx"]




sensitivity_factors = [round(1.00 + i * 0.01, 2) for i in range(11)]
base_cases = ["600FF", "650FF", "900FF", "950FF"]
min_pass_ranges = {
    "600FF": (-18.8, -15.6),
    "650FF": (-23.0, -21.0),
    "900FF": (-6.4, -1.6),
    "950FF": (-20.2, -17.8)
}

shaded_colors = {
    "600FF": "#c6f5c6",
    "650FF": "#a5e4a5",
    "900FF": "#8ddc8d",
    "950FF": "#75d475"
}



results = {case: [] for case in base_cases}
# Same code and process as listing C.1
def get_min_theta_air(file_path, case):
    data = pd.read_excel(file_path, sheet_name="Model_grey-box3")
    coefficients = pd.to_numeric(data.iloc[24:44, 3], errors='coerce').to_numpy()
    daily_schedule = data["Daily schedule ventilation"].to_numpy()
    b_matrix_fixed = data[["Eq1-known term", "Eq2-known term"]].to_numpy()
    X_solutions = []

    for i in range(len(b_matrix_fixed)):
        if daily_schedule[i] == 1:
            A_3x3 = np.array([
                [coefficients[9], -coefficients[7], 0],
                [-coefficients[1], coefficients[10], -coefficients[0]],
                [0, -coefficients[0], coefficients[11]]
            ])
            eq3_term = data["Eq3-known term"].iloc[i]
        else:
            A_3x3 = np.array([
                [coefficients[9], -coefficients[7], 0],
                [-coefficients[1], coefficients[10], -coefficients[0]],
                [0, -coefficients[0], coefficients[12]]
            ])
            eq3_term = data["Eq3-known term (different ventilation)"].iloc[i]

        b_matrix_row = np.array([b_matrix_fixed[i, 0], b_matrix_fixed[i, 1], eq3_term])
        A_3x3_inverse = np.linalg.inv(A_3x3)
        X_solution = np.dot(A_3x3_inverse, b_matrix_row)
        X_solutions.append(X_solution)

    solution_df = pd.DataFrame(X_solutions, columns=["Theta_m", "Theta_sup", "Theta_air"])
    return solution_df["Theta_air"].min()

for case in base_cases:
    for factor in sensitivity_factors:
        file_name = f"{case}.xlsx" if factor == 1.00 else f"{case}{factor:.2f}.xlsx"
        if os.path.exists(file_name):
            min_theta, is_pass = None, False
            try:
                min_theta = get_min_theta_air(file_name, case)
                is_pass = min_pass_ranges[case][0] <= min_theta <= min_pass_ranges[case][1]
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
            results[case].append((factor, min_theta, is_pass))
        else:
            results[case].append((factor, np.nan, False))
plt.figure(figsize=(10, 6))

for case in base_cases:
    pass_min, pass_max = min_pass_ranges[case]
    plt.axhspan(pass_min, pass_max, color=shaded_colors[case], alpha=0.2)

    for factor, theta, is_pass in results[case]:
        if np.isnan(theta):
            continue
        color = "green" if is_pass else "red"
        plt.scatter(factor, theta, color=color, edgecolor='black', s=60, zorder=5)


    x_vals = [x[0] for x in results[case] if not np.isnan(x[1])]
    y_vals = [x[1] for x in results[case] if not np.isnan(x[1])]
    plt.plot(x_vals, y_vals, marker='o', label=case)
    
plt.xlabel("Cm/Δτ Increase (%)")
plt.xticks(ticks=sensitivity_factors, labels=[f"{(f - 1.00)*100:.0f}%" for f in sensitivity_factors])
plt.ylabel("Minimum Indoor Air Temperature (°C)")
#plt.title("Sensitivity Analysis of Minimum Theta_air to Cm/Δτ Variations")
plt.grid(True, linestyle='--', alpha=0.6)
case_lines = [
    Line2D([0], [0], color='tab:blue', lw=2, label='600FF'),
    Line2D([0], [0], color='tab:orange', lw=2, label='650FF'),
    Line2D([0], [0], color='tab:green', lw=2, label='900FF'),
    Line2D([0], [0], color='tab:red', lw=2, label='950FF'),
]

pass_fail_markers = [
    Line2D([0], [0], marker='o', color='green', linestyle='None', label='Pass'),
    Line2D([0], [0], marker='o', color='red', linestyle='None', label='Fail')
]

combined_legend = case_lines + pass_fail_markers
plt.legend(handles=combined_legend, title="Case & Status", loc='center right')
plt.tight_layout()
plt.show()




# Define your case groups and labels
cases_group_1 = ["600FF1.08.xlsx", "900FF1.08.xlsx"]
cases_group_2 = ["650FF1.08.xlsx", "950FF1.08.xlsx"]
case_labels_group_1 = ["600FF", "900FF"]
case_labels_group_2 = ["650FF", "950FF"]
case_colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]

# Define x-axis ticks for months (approximate)
month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_starts_days = [d / 1 for d in month_starts]  # still in days

# === Group 1: 600FF & 900FF ===
plt.figure(figsize=(12, 6))
for file_path, case, color in zip(cases_group_1, case_labels_group_1, case_colors[:2]):
    data = pd.read_excel(file_path, sheet_name="Model_grey-box3")
    time = pd.to_numeric(data["Hours (incremental)"], errors="coerce") / 24  # Convert hours → days
    theta_air = pd.to_numeric(data["tair,model"], errors="coerce")
    valid = ~time.isna() & ~theta_air.isna()
    time, theta_air = time[valid], theta_air[valid]
    theta_air_smooth = theta_air.rolling(window=24, min_periods=1).mean()
    plt.plot(time, theta_air_smooth, label=case, color=color, linewidth=1.5, alpha=0.8)

plt.xlabel("Month")
plt.ylabel("Indoor Air Temperature (°C)")
plt.xticks(month_starts_days, month_labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === Group 2: 650FF & 950FF ===
plt.figure(figsize=(12, 6))
for file_path, case, color in zip(cases_group_2, case_labels_group_2, case_colors[2:]):
    data = pd.read_excel(file_path, sheet_name="Model_grey-box3")
    time = pd.to_numeric(data["Hours (incremental)"], errors="coerce") / 24
    theta_air = pd.to_numeric(data["tair,model"], errors="coerce")
    valid = ~time.isna() & ~theta_air.isna()
    time, theta_air = time[valid], theta_air[valid]
    theta_air_smooth = theta_air.rolling(window=24, min_periods=1).mean()
    plt.plot(time, theta_air_smooth, label=case, color=color, linewidth=1.5, alpha=0.8)

plt.xlabel("Month")
plt.ylabel("Indoor Air Temperature (°C)")
plt.xticks(month_starts_days, month_labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
