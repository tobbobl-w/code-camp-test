# data_summary.py
"""
A simple script to demonstrate basic data operations
"""
import numpy as np
import camp

def main():
    # Sample earnings data (in thousands)
    earnings_data = [45, 52, 38, 67, 43, 55, 49, 61, 39, 58]
    earnings_data2 = [ y**2 for y in earnings_data ]   
    dd = { f'i{i}':y for (i,y) in enumerate(earnings_data2) if y > 40 }

    # Your task: Calculate these statistics without using any libraries
    # Think about how you would compute each of these manually

    mean_earnings =  np.mean(earnings_data) # TODO: Calculate mean
    median_earnings = 0.0 # TODO: Calculate median (hint: sort first)
    max_earnings = 0.0 # TODO: Find maximum
    min_earnings = 0.0 # TODO: Find minimum

    print(f"Earnings Analysis:")
    print(f"Mean: ${mean_earnings:.2f}k")
    print(f"Median: ${median_earnings:.2f}k") 
    print(f"Range: ${min_earnings:.2f}k - ${max_earnings:.2f}k")

    camp.utils.hello("data_summary")

if __name__ == "__main__":
    main()