import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import johnsonsu
from scipy.optimize import root

# Set white plot theme
plt.style.use('default')
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

# MainData inputs and median override
st.sidebar.header("MainData (10–180% benchmark)")
# Ordered: N, Min, Max, Mean, Median, SD, Skewness, Kurtosis
n1 = st.sidebar.number_input(
    "N MainData", min_value=0, value=428, key="n_main"
)
min1 = st.sidebar.number_input(
    "Min MainData", value=357, key="min_main"
)
max1 = st.sidebar.number_input(
    "Max MainData", value=6223, key="max_main"
)
mean1 = st.sidebar.number_input(
    "Mean MainData", value=1354, key="mean_main"
)
median1 = st.sidebar.number_input(
    "Median MainData", value=1082, key="median_main"
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

# LT10 inputs
st.sidebar.header("LT10 (<10% benchmark)")
n2 = st.sidebar.number_input(
    "N LT10", min_value=0, value=123, key="n_lt10"
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
st.sidebar.header("GT180 (>180% benchmark)")
n3 = st.sidebar.number_input(
    "N GT180", min_value=0, value=5, key="n_gt180"
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

# Fit Johnson SU via moments
def fit_jsu(m, v, skew, kurt_ex):
    def residuals(params):
        a, b, loc, scale = params
        m0, v0, sk0, exk0 = johnsonsu.stats(
            a, b, loc=loc, scale=scale, moments='mvsk'
        )
        return [m0 - m, v0 - v, sk0 - skew, exk0 - (kurt_ex - 3)]
    init = [0.0, 1.0, m, np.sqrt(v)]
    sol = root(residuals, init)
    return sol.x

# Compute JSU parameters for MainData if available
if n1 > 0:
    a_fit, b_fit, loc_fit, scale_fit = fit_jsu(
        mean1, sd1**2, skew1, kurt1
    )
else:
    a_fit = b_fit = loc_fit = scale_fit = None

# Sampling helper
def sample_norm(n, mean, sd, lo, hi):
    if n <= 0:
        return np.array([])
    arr = np.random.normal(loc=mean, scale=sd, size=int(n*2))
    arr = arr[(arr>=lo)&(arr<=hi)]
    if len(arr) < n:
        arr = np.pad(
            arr, (0, n-len(arr)), mode='constant', constant_values=lo
        )
    return arr[:n]

# Generate draws
large_sample = np.array([])
if a_fit is not None:
    large_sample = johnsonsu.rvs(
        a_fit, b_fit, loc=loc_fit, scale=scale_fit, size=1000000
    )
# Sample MainData directly from fitted Johnson SU, preserving skew
if a_fit is not None and n1 > 0:
    raw_s1 = johnsonsu.rvs(
        a_fit, b_fit, loc=loc_fit, scale=scale_fit, size=int(n1*2)
    )
    s1 = raw_s1[(raw_s1 >= min1) & (raw_s1 <= max1)]
    # Pad/truncate to N, using min1 to pad
    if len(s1) < n1:
        s1 = np.pad(s1, (0, int(n1) - len(s1)), mode='constant', constant_values=min1)
    else:
        s1 = s1[:int(n1)]
else:
    s1 = np.array([])
# Sample LT10 and GT180 with normals for low and high outliers
s2 = sample_norm(n2, mean2, sd2, min2, max2)
s3 = sample_norm(n3, mean3, sd3, min3, max3)

# Plot function
def main_plot():
    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Determine x-axis limit
    arrays = [large_sample, s1, s2, s3]
    max_vals = [a.max() for a in arrays if a.size>0]
    xmax = max(max_vals) if max_vals else new_benchmark
    bins = np.arange(0, xmax+bin_width, bin_width)

    # Compute histograms for display
    hist1, edges = np.histogram(s1, bins=bins, density=True) if s1.size>0 else (np.array([]), bins)
    hist2, _     = np.histogram(s2, bins=bins, density=True) if s2.size>0 else (np.array([]), bins)
    hist3, _     = np.histogram(s3, bins=bins, density=True) if s3.size>0 else (np.array([]), bins)

    # Determine peak of others and scale hist1 to 80% of larger peak
    peak_others = 0
    if hist2.size>0:
        peak_others = max(peak_others, hist2.max())
    if hist3.size>0:
        peak_others = max(peak_others, hist3.max())
    scale1 = 1.0
    if hist1.size>0 and peak_others>0:
        scale1 = 0.8 * peak_others / hist1.max()

    # Plot scaled MainData bars
    if hist1.size>0:
        ax.bar(
            edges[:-1], hist1*scale1, width=bin_width,
            align='edge', alpha=0.6, color='lightblue', edgecolor='black',
            label=f'MainData (N={n1})'
        )

    # Plot LT10 and GT180 as usual
    if s2.size>0:
        ax.hist(
            s2, bins=bins, density=True, alpha=0.6,
            color='darkblue', edgecolor='black',
            label=f'LT10 (N={n2})'
        )
    if s3.size>0:
        ax.hist(
            s3, bins=bins, density=True, alpha=0.6,
            color='orange', edgecolor='black',
            label=f'GT180 (N={n3})'
        )

    # JSU density
    if large_sample.size>0:
        x_vals = np.linspace(0, xmax, 1000)
        y_vals = johnsonsu.pdf(x_vals, a_fit, b_fit, loc=loc_fit, scale=scale_fit)
        y_vals = y_vals * scale1
        ax.plot(x_vals, y_vals, ':', color='black', linewidth=2, label='Large-N JSU fit')
        ax.axvline(median1, color='black', linestyle='--', label=f'Median MainData = {median1}')    # Benchmarks
    ax.axvline(
        current_benchmark, color='blue', linestyle='--',
        label=f'Current Benchmark = {current_benchmark}'
    )
    ax.axvline(
        new_benchmark, color='green', linestyle='--',
        label=f'New Benchmark = {new_benchmark}'
    )

    ax.set_xlabel('Value (AUD$)')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')
    return fig

# Render
fig = main_plot()
st.pyplot(fig)

