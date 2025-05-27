# Distribution Explorer Streamlit App

This application provides an interactive environment for exploring payment distributions for assistive technology under different scenarios. It fits a Johnson SU distribution to your main dataset and overlays three sample distributions.

## Features

- **Dynamic Parameter Inputs**: Adjust:
  - Bin width
  - Current and new benchmark values
  - Main data summary (N, mean, SD, skewness, kurtosis, min, max)
  - LT10 and GT180 sample distributions (N, mean, SD, min, max)
- **Automatic JSU Fitting**: Moment-matches a Johnson SU distribution to your main data summary.
- **Interactive Plot**: Displays three filled histograms (MainData, LT10, GT180) and a dotted JSU density, with vertical benchmark and median lines.

## Deployment and Usage

Streamlit offers multiple deployment options:

### Local Run

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app:  
   ```bash
   streamlit run app.py
   ```
3. Open your browser to `http://localhost:8501`.

### Streamlit Cloud

1. Push the repository to GitHub (public or private).  
2. Sign in to Streamlit Cloud and create a new app from your GitHub repo.  
3. Specify the main branch and `app.py` as the entry point, then deploy.

## File Structure

- `Dockerfile` — Container definition (optional).  
- `requirements.txt` — Python dependencies.  
- `docker-compose.yml` — Service orchestration (optional).  
- `app.py` — Streamlit application source.

## Customisation

- Update default values for any sidebar control in `app.py`.  
- Add or remove sample distributions by extending the input and histogram sections.  
- Modify plot styling and layout via Matplotlib settings in `app.py`.
