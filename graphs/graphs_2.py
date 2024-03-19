import matplotlib.pyplot as plt
import numpy as np

# Sample data
time = np.arange(10)
category_a = np.array([2, 4, 6, 8, 10, 8, 6, 4, 2, 0])
category_b = np.array([0, 2, 4, 6, 8, 10, 8, 6, 4, 2])

# Calculate total area
total_area = category_a + category_b

# Calculate cumulative sums
cumsum_a = np.cumsum(category_a)
cumsum_b = np.cumsum(category_b)

# Find index where they share 50% of the total area
halfway_point = np.argmax(total_area >= total_area.max() / 2)

# Plot stacked area chart
plt.fill_between(time, cumsum_a, label='Category A', color='skyblue', alpha=0.5)
plt.fill_between(time, cumsum_b, label='Category B', color='orange', alpha=0.5)

# Plot vertical line at halfway point
plt.axvline(x=halfway_point, color='red', linestyle='--', label='50% of Total Area')

# Customize plot
plt.xlabel('Time')
plt.ylabel('Total Area')
plt.title('Stacked Area Chart with 50% Highlight')
plt.legend()
plt.grid(True)

plt.show()
