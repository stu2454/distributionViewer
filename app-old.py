import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import johnsonsu

st.set_page_config(page_title="Distribution Explorer", layout="wide")
st.title("Interactive Payment Distribution Explorer")

# Sidebar inputs
st.sidebar.header("Global Settings")
bin_width = st.sidebar.number_input("Bin width (AUD$)", min_value=1, value=500)
current_benchmark = st.sidebar.number_input("Current Benchmark (AUD$)", value=1354)
new_benchmark = st.sidebar.number_input("New Benchmark (AUD$)", value=1500)

st.sidebar.header("Large-N Distribution (Johnson SU)")
a = st.sidebar.number_input("Shape a", value=-6.8112, format="%.4f")
b = st.sidebar.number_input("Shape b", value=3.8942, format="%.4f")
loc = st.sidebar.number_input("Location (loc)", value=-2377.0314, format="%.4f")
scale = st.sidebar.number_input("Scale", value=1295.0398, format="%.4f")
n_large = st.sidebar.number_input("Sample size N (large)", min_value=1, value=1000000)

st.sidebar.header("Distribution 1 (Normal) n=428 sample")
n1 = st.sidebar.number_input("N1", min_value=1, value=428)
mean1 = st.sidebar.number_input("Mean1", value=1354)
sd1 = st.sidebar.number_input("SD1", value=1033)
min1 = st.sidebar.number_input("Min1", value=357)
max1 = st.sidebar.number_input("Max1", value=6223)

st.sidebar.header("Distribution 2 (Normal) n=123 sample")
n2 = st.sidebar.number_input("N2", min_value=1, value=123)
mean2 = st.sidebar.number_input("Mean2", value=189)
sd2 = st.sidebar.number_input("SD2", value=87)
min2 = st.sidebar.number_input("Min2", value=4)
max2 = st.sidebar.number_input("Max2", value=346)

st.sidebar.header("Distribution 3 (Normal) n=5 sample")
n3 = st.sidebar.number_input("N3", min_value=1, value=5)
mean3 = st.sidebar.number_input("Mean3", value=9977)
sd3 = st.sidebar.number_input("SD3", value=3932)
min3 = st.sidebar.number_input("Min3", value=6401)
max3 = st.sidebar.number_input("Max3", value=14285)

# Generate samples
large_sample = johnsonsu.rvs(a, b, loc=loc, scale=scale, size=int(n_large))
s1 = np.random.normal(loc=mean1, scale=sd1, size=int(n1))
s1 = np.clip(s1, min1, max1)
s2 = np.random.normal(loc=mean2, scale=sd2, size=int(n2))
s2 = np.clip(s2, min2, max2)
s3 = np.random.normal(loc=mean3, scale=sd3, size=int(n3))
s3 = np.clip(s3, min3, max3)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.arange(0, max(s3.max(), s1.max(), s2.max(), large_sample.max()) + bin_width, bin_width)

ax.hist(s1, bins=bins, density=True, alpha=0.6, color='lightblue', edgecolor='black', label=f'N={n1}')
ax.hist(s2, bins=bins, density=True, alpha=0.6, color='darkblue', edgecolor='black', label=f'N={n2}')
ax.hist(s3, bins=bins, density=True, alpha=0.6, color='orange', edgecolor='black', label=f'N={n3}')

# Smooth Johnson SU density
x = np.linspace(0, bins[-1], 1000)
y = johnsonsu.pdf(x, a, b, loc=loc, scale=scale)
ax.plot(x, y, '--', color='red', linewidth=2, label='Large-N JSU fit')

# Vertical markers
median_large = johnsonsu.ppf(0.5, a, b, loc=loc, scale=scale)
ax.axvline(median_large, color='red', linestyle='--', label='Median (large N)')
ax.axvline(current_benchmark, color='black', linestyle='--', label='Current Benchmark')
ax.axvline(new_benchmark, color='green', linestyle='--', label='New Benchmark')

ax.set_xlabel('Value (AUD$)')
ax.set_ylabel('Density')
ax.legend()
st.pyplot(fig)
