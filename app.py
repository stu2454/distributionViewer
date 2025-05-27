import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import johnsonsu

# Ensure white backgrounds
plt.style.use('default')
# Force figure and axis facecolors to white
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

st.set_page_config(page_title="Distribution Explorer", layout="wide")
st.title("Interactive Payment Distribution Explorer")

# Sidebar inputs
st.sidebar.header("Global Settings")
bin_width = st.sidebar.number_input(
    "Bin width (AUD$)", min_value=1, value=500, key="bin_width_global"
)
current_benchmark = st.sidebar.number_input(
    "Current Benchmark (AUD$)", value=1354, key="current_benchmark"
)
new_benchmark = st.sidebar.number_input(
    "New Benchmark (AUD$)", value=1500, key="new_benchmark"
)

# Fit Johnson SU parameters from MainData summary
st.sidebar.header("Fit Johnson SU from MainData")
# MainData inputs
n1 = st.sidebar.number_input(
    "N MainData", min_value=1, value=428, key="n_main"
)
mean1 = st.sidebar.number_input(
    "Mean MainData", value=1354, key="mean_main"
)
sd1 = st.sidebar.number_input(
    "SD MainData", value=1033, key="sd_main"
)
skew1 = st.sidebar.number_input(
    "Skewness MainData", value=1.96, format="%.2f", key="skew_main"
)
kurt1 = st.sidebar.number_input(
    "Kurtosis MainData", value=4.24, format="%.2f", key="kurt_main"
)
min1 = st.sidebar.number_input(
    "Min MainData", value=357, key="min_main"
)
max1 = st.sidebar.number_input(
    "Max MainData", value=6223, key="max_main"
)

# LT10 inputs
st.sidebar.header("LT10 (Normal) – payments <10% of benchmark")
n2 = st.sidebar.number_input(
    "N LT10", min_value=1, value=123, key="n_lt10"
)
mean2 = st.sidebar.number_input(
    "Mean LT10", value=189, key="mean_lt10"
)
sd2 = st.sidebar.number_input(
    "SD LT10", value=87, key="sd_lt10"
)
min2 = st.sidebar.number_input(
    "Min LT10", value=4, key="min_lt10"
)
max2 = st.sidebar.number_input(
    "Max LT10", value=346, key="max_lt10"
)

# GT180 inputs
st.sidebar.header("GT180 (Normal) – payments >180% of benchmark")
n3 = st.sidebar.number_input(
    "N GT180", min_value=1, value=5, key="n_gt180"
)
mean3 = st.sidebar.number_input(
    "Mean GT180", value=9977, key="mean_gt180"
)
sd3 = st.sidebar.number_input(
    "SD GT180", value=3932, key="sd_gt180"
)
min3 = st.sidebar.number_input(
    "Min GT180", value=6401, key="min_gt180"
)
max3 = st.sidebar.number_input(
    "Max GT180", value=14285, key="max_gt180"
)

# Function to fit JohnsonSU by moments
def fit_jsu(m, v, skew, kurt_ex):
    from scipy.stats import johnsonsu
    from scipy.optimize import root
    def residuals(params):
        a, b, loc, scale = params
        m0, v0, sk0, exk0 = johnsonsu.stats(a, b, loc=loc, scale=scale, moments='mvsk')
        return [m0 - m, v0 - v, sk0 - skew, exk0 - (kurt_ex - 3)]
    init = [0.0, 1.0, m, np.sqrt(v)]
    sol = root(residuals, init)
    return sol.x

# Fit parameters for large distribution
from scipy.stats import johnsonsu  # ensure import
from scipy.optimize import root

target_variance = sd1 ** 2
a_fit, b_fit, loc_fit, scale_fit = fit_jsu(mean1, target_variance, skew1, kurt1)

# Generate samples
large_sample = johnsonsu.rvs(
    a_fit, b_fit, loc=loc_fit, scale=scale_fit, size=int(1e6)
)
# Sample MainData, LT10, GT180
raw_s1 = np.random.normal(loc=mean1, scale=sd1, size=int(n1*2))  # oversample to ensure enough
raw_s2 = np.random.normal(loc=mean2, scale=sd2, size=int(n2*2))
raw_s3 = np.random.normal(loc=mean3, scale=sd3, size=int(n3*2))

# Enforce non-overlap at boundary
boundary = max2  # exclusive upper bound for LT10
s2 = raw_s2[(raw_s2 >= min2) & (raw_s2 < boundary)]
s1 = raw_s1[(raw_s1 > boundary) & (raw_s1 <= max1)]
s3 = raw_s3[(raw_s3 >= min3) & (raw_s3 <= max3)]

# Pad/truncate to requested N
# Use constant padding to avoid wrap errors on empty arrays
def pad_array(arr, n, pad_val):
    if len(arr) < n:
        return np.pad(arr, (0, n - len(arr)), mode='constant', constant_values=pad_val)
    return arr[:n]

s1 = pad_array(s1, int(n1), boundary)
s2 = pad_array(s2, int(n2), min2)
s3 = pad_array(s3, int(n3), min3)

# Plot
def main_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    # Ensure white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Determine x-axis max
    arrays = [large_sample, s1, s2, s3]
    valid_max = [arr.max() for arr in arrays if arr.size > 0]
    xmax = max(valid_max) if valid_max else current_benchmark
    bins = np.arange(0, xmax + bin_width, bin_width)

    # Plot histograms
    if s1.size > 0:
        ax.hist(
            s1,
            bins=bins,
            density=True,
            alpha=0.6,
            color='lightblue',
            edgecolor='black',
            label=f'MainData (N={n1})',
        )
    if s2.size > 0:
        ax.hist(
            s2,
            bins=bins,
            density=True,
            alpha=0.6,
            color='darkblue',
            edgecolor='black',
            label=f'LT10 (N={n2})',
        )
    if s3.size > 0:
        ax.hist(
            s3,
            bins=bins,
            density=True,
            alpha=0.6,
            color='orange',
            edgecolor='black',
            label=f'GT180 (N={n3})',
        )

    # Plot JSU density
    if large_sample.size > 0:
        x_vals = np.linspace(0, xmax, 1000)
        y_vals = johnsonsu.pdf(x_vals, a_fit, b_fit, loc=loc_fit, scale=scale_fit)
        ax.plot(
            x_vals,
            y_vals,
            linestyle=':',
            color='black',
            linewidth=2,
            label='Large-N JSU fit',
        )
        # Median
        median_val = johnsonsu.ppf(
            0.5,
            a_fit,
            b_fit,
            loc=loc_fit,
            scale=scale_fit,
        )
        ax.axvline(
            median_val,
            color='black',
            linestyle='--',
            label=f'Median (large N) = {median_val:.0f}',
        )

    # Benchmark lines
    ax.axvline(
        current_benchmark,
        color='blue',
        linestyle='--',
        label=f'Current Benchmark = {current_benchmark}',
    )
    ax.axvline(
        new_benchmark,
        color='green',
        linestyle='--',
        label=f'New Benchmark = {new_benchmark}',
    )

    ax.set_xlabel('Value (AUD$)')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')
    return fig

# Render plot
fig = main_plot()
st.pyplot(fig)
