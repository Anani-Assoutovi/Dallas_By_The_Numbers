# Dallas Incidents Police Report

<p>
    This analysis is designed to give insight on the incident police report from **June 2014** to **May 7th 2025**.
</p>

<p>
    The [data source](https://www.dallasopendata.com/Public-Safety/Police-Incidents/qv6i-rri7/about_data) does have an **API endpoint** access. However, that source only give up to about **1,000** rows of data. That approach would have been ineffective to this analysis. 
</p>

### Some of the libraries that I use can be found here:
* `python3 -m pip install pandas`
* `python3 -m pip install folium`
* `python3 -m pip install statsmodels`
* `python3 -m pip install streamlit`

### How to run the Python notebook
Because streamlit data limitations, I cut the data into 10,000 for obsevation.
`cd {to directory of python file}`
`streamlit run crime_dashboard.py`

#### Results
A browser (default) open with two (2) tabs:
1) first tab will display the KDE and the future forecast
2) a tab with the Follium HeatMap will open up to display an interactive heatmap



For questions or comments, just reach out to me

# Thank You!



