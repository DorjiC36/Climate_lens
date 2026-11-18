from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="ClimateLens | Climate Dashboard",
    page_icon=":material/public:",
    layout="wide",
)

# Climate-inspired chart palette: forest, glacier, sky and sunlight
INK = "#173B35"
FOREST = "#287A59"
MINT = "#75BFA3"
SKY = "#4D9BC7"
SUN = "#E7B84B"
CORAL = "#E17A5F"
PURPLE = "#8276B8"
CHART_COLORS = [FOREST, SKY, SUN, CORAL, PURPLE, MINT, "#6B8E9B", "#B7A078"]

DATA_PATH = Path(__file__).parent / "climate_change_dataset_clean.csv"
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).parent / "data" / "climate_change_dataset.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    needed = ["Year","Country","Avg Temperature (°C)","CO2 Emissions (Tons/Capita)",
              "Sea Level Rise (mm)","Rainfall (mm)","Population",
              "Renewable Energy (%)","Extreme Weather Events","Forest Area (%)"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing: " + ", ".join(missing))
    for c in needed:
        if c not in ("Country",):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Year","Country"])
    df["Year"] = df["Year"].astype(int)
    return df

try:
    df_all = load_data(str(DATA_PATH))
except Exception as e:
    st.error(f"Could not load climate dataset: {e}")
    st.stop()

LABELS = {
    "Avg Temperature (°C)": "Average temperature (°C)",
    "CO2 Emissions (Tons/Capita)": "CO₂ emissions (tons/capita)",
    "Sea Level Rise (mm)": "Sea-level rise (mm)",
    "Rainfall (mm)": "Rainfall (mm)",
    "Population": "Population",
    "Renewable Energy (%)": "Renewable energy (%)",
    "Extreme Weather Events": "Extreme weather events",
    "Forest Area (%)": "Forest area (%)"
}
VARS = list(LABELS)

def style(fig, height=390):
    fig.update_layout(
        height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=INK),
        margin=dict(l=20,r=20,t=70,b=35),
        legend=dict(title_text="Legend", bgcolor="rgba(255,255,255,0.7)"),
        title=dict(font=dict(size=17,color=INK),x=0.02),
    )
    fig.update_xaxes(showgrid=False, linecolor="#D9E7DF", zeroline=False)
    fig.update_yaxes(gridcolor="#E5EEE8", zeroline=False)
    return fig

def chart_caption(text):
    st.caption(text)

with st.sidebar:
    st.title("ClimateLens", icon=":material/public:")
    st.caption("Climate patterns • data insights")
    page = st.radio("Navigate", ["Dashboard", "Research analysis"], index=0)
    st.divider()
    st.subheader("Data filters", icon=":material/filter_alt:")
    years = sorted(df_all["Year"].dropna().unique())
    yr_range = st.slider("Year range", int(min(years)), int(max(years)),
                         (int(min(years)), int(max(years))))
    country_options = sorted(df_all["Country"].dropna().unique().tolist())
    selected_countries = st.multiselect("Countries", country_options, default=country_options)
    st.caption("Filters apply to both pages.")

df = df_all[df_all["Year"].between(*yr_range)].copy()
if selected_countries:
    df = df[df["Country"].isin(selected_countries)]
if df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

st.caption("Climate data explorer")
if page == "Dashboard":
    st.title("Climate dashboard", icon=":material/monitoring:")
    st.write("Track changes over time, compare countries, and explore climate indicators. Use the filters to focus on the places and years you care about.")
    a,b,c,d = st.columns(4)
    a.metric("Records", f"{len(df):,}", border=True)
    b.metric("Countries", f"{df.Country.nunique():,}", border=True)
    c.metric("Year coverage", f"{int(df.Year.min())}–{int(df.Year.max())}", border=True)
    d.metric("Mean temperature", f"{df['Avg Temperature (°C)'].mean():.2f} °C", border=True)

    st.subheader("Climate indicators through time", icon=":material/trending_up:")
    indicator = st.selectbox("Choose indicator", VARS, index=0, format_func=lambda x: LABELS[x], key="trend_ind")
    annual = df.groupby("Year",as_index=False)[indicator].mean()
    fig = px.line(annual, x="Year", y=indicator, markers=True,
                  color_discrete_sequence=[FOREST],
                  title=f"{LABELS[indicator]} by year")
    fig.update_traces(line_width=3, marker_size=7, fill="tozeroy", fillcolor="rgba(40,122,89,0.10)")
    fig.update_layout(xaxis_title="Year", yaxis_title=LABELS[indicator])
    st.plotly_chart(
        style(fig,420), width="stretch",
        alt=f"{LABELS[indicator]} by year from {int(annual['Year'].min())} to {int(annual['Year'].max())}.",
    )
    chart_caption("Each point is the average across the selected countries for that year.")

    left,right = st.columns(2)
    with left:
        country_metric = st.selectbox("Country comparison metric", VARS, index=1,
                                      format_func=lambda x: LABELS[x], key="country_metric")
        country_avg = df.groupby("Country",as_index=False)[country_metric].mean().dropna()
        country_avg = country_avg.nlargest(15, country_metric).sort_values(country_metric)
        country_order = country_avg["Country"].tolist()
        fig = px.scatter(
            country_avg, x=country_metric, y="Country", color=country_metric,
            color_continuous_scale=["#A8D7E8",FOREST],
            category_orders={"Country": country_order},
            labels={country_metric: LABELS[country_metric], "Country": ""},
            title=f"Country comparison: {LABELS[country_metric]}",
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        fig.update_layout(coloraxis_colorbar=dict(title=LABELS[country_metric]))
        st.plotly_chart(
            style(fig,480), width="stretch",
            alt=f"Country ranking by {LABELS[country_metric]}; higher averages appear farther right.",
        )
        chart_caption("Each dot is one country; farther right means a higher average in the selected period.")
    with right:
        temperature_order = (
            df.groupby("Country")["Avg Temperature (°C)"].median()
            .sort_values()
            .index.tolist()
        )
        fig = px.box(
            df, x="Avg Temperature (°C)", y="Country", points="outliers",
            category_orders={"Country": temperature_order},
            color_discrete_sequence=[SKY],
            title="How yearly temperatures vary by country",
        )
        fig.update_layout(xaxis_title="Average temperature (°C)", yaxis_title="")
        st.plotly_chart(
            style(fig,480), width="stretch",
            alt="Temperature ranges by country; boxes show the middle half of yearly readings.",
        )
        chart_caption("The box shows the middle half of yearly readings; the center line is the median and isolated points are outliers.")

    annual_events = df.groupby("Year",as_index=False)["Extreme Weather Events"].sum()
    fig = px.line(
        annual_events, x="Year", y="Extreme Weather Events", markers=True,
        color_discrete_sequence=[CORAL],
        title="When were extreme weather events recorded?",
    )
    fig.update_traces(line_width=3, marker_size=7)
    fig.update_layout(xaxis_title="Year", yaxis_title="Recorded events")
    st.plotly_chart(
        style(fig,430), width="stretch",
        alt="Total recorded extreme weather events across selected countries by year.",
    )
    chart_caption("Each point is the total number of recorded events across the selected countries for that year.")

    left,right = st.columns(2)
    with left:
        env = df.groupby("Year",as_index=False)[["Renewable Energy (%)","Forest Area (%)"]].mean().melt(
            id_vars="Year",var_name="Indicator",value_name="Mean value")
        env["Indicator"] = env["Indicator"].map(LABELS)
        fig = px.line(env,x="Year",y="Mean value",color="Indicator",markers=True,
                      color_discrete_map={
                          LABELS["Renewable Energy (%)"]: FOREST,
                          LABELS["Forest Area (%)"]: MINT,
                      },
                      title="How renewable energy and forest area change")
        fig.update_layout(xaxis_title="Year", yaxis_title="Annual average (%)")
        st.plotly_chart(
            style(fig,420), width="stretch",
            alt="Annual average renewable energy use and forest area across selected countries.",
        )
        chart_caption("Both lines are annual averages in percent, so their changes can be compared directly.")
    with right:
        rain = df.groupby("Year",as_index=False)["Rainfall (mm)"].mean()
        fig = px.area(rain,x="Year",y="Rainfall (mm)",markers=True,
                      color_discrete_sequence=[SKY],title="How annual rainfall changes")
        fig.update_traces(fillcolor="rgba(77,155,199,0.18)",line_width=3)
        fig.update_layout(xaxis_title="Year", yaxis_title="Rainfall (mm)")
        st.plotly_chart(
            style(fig,420), width="stretch",
            alt="Annual average rainfall across selected countries over time.",
        )
        chart_caption("The line shows the annual average; the shaded area makes changes over time easier to see.")

    st.subheader("Compare countries by climate indicator", icon=":material/bar_chart:")
    profile_columns = [
        "Avg Temperature (°C)",
        "CO2 Emissions (Tons/Capita)",
        "Renewable Energy (%)",
        "Sea Level Rise (mm)",
    ]
    profile_labels = {
        "Avg Temperature (°C)": "Temperature (°C)",
        "CO2 Emissions (Tons/Capita)": "CO₂ emissions per capita (tons)",
        "Renewable Energy (%)": "Renewable energy (%)",
        "Sea Level Rise (mm)": "Sea-level rise (mm)",
    }
    profiles = (
        df.groupby("Country",as_index=False)[profile_columns]
        .mean()
        .dropna(subset=profile_columns)
        .sort_values("Country")
        .reset_index(drop=True)
    )
    if profiles.empty:
        st.info("No countries have complete values for all four indicators in the selected filters.")
    else:
        profile_long = profiles.melt(
            id_vars="Country", value_vars=profile_columns,
            var_name="Indicator", value_name="Average",
        )
        profile_long["Indicator"] = profile_long["Indicator"].map(profile_labels)
        indicator_order = list(profile_labels.values())
        fig = px.bar(
            profile_long, x="Average", y="Country", color="Indicator",
            facet_col="Indicator", facet_col_wrap=2, orientation="h",
            category_orders={"Indicator":indicator_order},
            labels={
                "Average":"Average across selected years",
                "Country":"",
                "Indicator":"Climate indicator",
            },
            color_discrete_sequence=CHART_COLORS,
            title="Country averages by climate indicator",
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(matches=None)
        fig.update_yaxes(matches=None, categoryorder="total ascending", autorange="reversed")
        fig.update_traces(hovertemplate="%{y}<br>%{x:.2f}<extra></extra>")
        st.plotly_chart(
            style(fig,780), width="stretch",
            alt="Four ranked horizontal bar charts compare selected countries' average temperature, CO₂ emissions, renewable energy, and sea-level rise.",
        )
        chart_caption("Each panel ranks country averages for one indicator. Panels use separate scales and units, so compare countries within a panel rather than comparing bar lengths across indicators.")

    decadal = df.assign(Decade=(df["Year"] // 10) * 10)
    sea_decade = decadal.groupby("Decade",as_index=False)["Sea Level Rise (mm)"].mean()
    fig = px.line(
        sea_decade,x="Decade",y="Sea Level Rise (mm)",markers=True,
        color_discrete_sequence=[SKY],
        title="How sea-level rise differs by decade",
    )
    fig.update_traces(line_width=3, marker_size=8)
    fig.update_layout(xaxis_title="Decade", yaxis_title="Average sea-level rise (mm)")
    st.plotly_chart(
        style(fig,430), width="stretch",
        alt="Average sea-level rise for each decade in the selected years.",
    )
    chart_caption("Each point is the average across the selected countries within that decade.")

    st.subheader("Temperature patterns across decades", icon=":material/thermostat:")
    decadal_temperature = (
        df.assign(Decade=(df["Year"] // 10) * 10)
        .groupby(["Country","Decade"],as_index=False)["Avg Temperature (°C)"]
        .mean()
    )
    fig = px.line(
        decadal_temperature,x="Decade",y="Avg Temperature (°C)",color="Country",
        markers=True,color_discrete_sequence=CHART_COLORS,
        title="How average temperature changes by country and decade",
    )
    fig.update_layout(xaxis_title="Decade", yaxis_title="Average temperature (°C)")
    st.plotly_chart(
        style(fig,460), width="stretch",
        alt="Average temperature by decade for each selected country.",
    )
    chart_caption("Each line represents a country. Select fewer countries in the sidebar to compare specific places more easily.")

    st.subheader("Records by climate category", icon=":material/donut_small:")
    left,right = st.columns(2)
    with left:
        temp_categories = pd.cut(
            df["Avg Temperature (°C)"],
            bins=[-np.inf,10,20,30,np.inf],
            labels=["Cold","Mild","Warm","Hot"],
        )
        temp_counts = (
            temp_categories.value_counts(sort=False)
            .rename_axis("Temperature category")
            .reset_index(name="Records")
        )
        if temp_counts["Records"].sum():
            fig = px.pie(
                temp_counts,names="Temperature category",values="Records",hole=.55,
                color="Temperature category",
                color_discrete_map={
                    "Cold": SKY, "Mild": FOREST, "Warm": SUN, "Hot": CORAL,
                },
                title="How temperature records are distributed",
            )
            st.plotly_chart(
                style(fig,400), width="stretch",
                alt="Share of temperature records in cold, mild, warm, and hot categories.",
            )
            chart_caption("Cold: 10 °C or lower; mild: above 10–20 °C; warm: above 20–30 °C; hot: above 30 °C.")
        else:
            st.info("There are no temperature values to group for the selected filters.")
    with right:
        energy_levels = (
            df.loc[df["Renewable Energy (%)"].notna(),"Renewable Energy (%)"]
            .ge(30)
            .map({True:"30% or higher",False:"Below 30%"})
            .value_counts()
            .rename_axis("Renewable energy use")
            .reset_index(name="Records")
        )
        if energy_levels["Records"].sum():
            fig = px.pie(
                energy_levels,names="Renewable energy use",values="Records",hole=.55,
                color="Renewable energy use",
                color_discrete_map={"Below 30%": SKY, "30% or higher": FOREST},
                title="How many records meet the 30% renewable-energy mark?",
            )
            st.plotly_chart(
                style(fig,400), width="stretch",
                alt="Share of records with renewable energy use below or at least 30%.",
            )
            chart_caption("A record counts as high renewable-energy use when the value is at least 30%.")
        else:
            st.info("There are no renewable-energy values to group for the selected filters.")

    st.subheader("Explore an indicator's range", icon=":material/query_stats:")
    overview_indicator = st.selectbox(
        "Choose an indicator to explore",
        VARS,
        format_func=lambda x: LABELS[x],
        key="overview_indicator",
    )
    indicator_values = df[overview_indicator].dropna()
    if indicator_values.empty:
        st.info("There are no recorded values for this indicator in the selected filters.")
    else:
        distribution = df[[overview_indicator]].dropna()
        left,right = st.columns(2)
        with left:
            fig = px.histogram(
                distribution,x=overview_indicator,nbins=20,
                color_discrete_sequence=[SKY],
                title=f"Distribution of {LABELS[overview_indicator]} values",
            )
            fig.update_layout(xaxis_title=LABELS[overview_indicator], yaxis_title="Number of records")
            st.plotly_chart(
                style(fig,420), width="stretch",
                alt=f"Frequency distribution of {LABELS[overview_indicator]} values.",
            )
            chart_caption("Each column counts how many records fall within that value range.")
        with right:
            fig = px.box(
                distribution,y=overview_indicator,points="outliers",
                color_discrete_sequence=[FOREST],
                title=f"Spread of {LABELS[overview_indicator]} values",
            )
            fig.update_layout(yaxis_title=LABELS[overview_indicator], xaxis_title="")
            st.plotly_chart(
                style(fig,420), width="stretch",
                alt=f"Spread and outliers for {LABELS[overview_indicator]} values.",
            )
            chart_caption("The box shows the middle half of values; the center line is the median and isolated points are outliers.")

else:
    st.title("Research analysis", icon=":material/biotech:")
    st.write("Research questions, descriptive findings and statistical relationships in the filtered dataset.")
    questions = [
        "What trends can be seen in climate indicators over the years?",
        "How do average temperature and CO₂ emissions differ across countries and years?",
        "Is there a relationship between CO₂ emissions, average temperature and sea-level rise?",
        "How do renewable energy use and forest area vary in the dataset?",
        "What patterns can be seen in extreme weather events?"
    ]
    for i,q in enumerate(questions,1):
        with st.expander(f"Research Question {i}: {q}",expanded=(i==1)):
            col = ["Avg Temperature (°C)","CO2 Emissions (Tons/Capita)","Sea Level Rise (mm)",
                   "Renewable Energy (%)","Forest Area (%)","Extreme Weather Events"][(i-1)]
            annual = df.groupby("Year",as_index=False)[col].mean().dropna()
            if len(annual)>1:
                slope = np.polyfit(annual["Year"],annual[col],1)[0]
                direction = "increasing" if slope>0 else "decreasing" if slope<0 else "flat"
                st.write(f"Across the selected records, {LABELS[col]} shows a {direction} linear trend (slope {slope:.4f} per year).")
            st.write(f"Mean: **{df[col].mean():.3f}** · Median: **{df[col].median():.3f}** · Valid records: **{df[col].notna().sum():,}**")

    st.subheader("Test a relationship", icon=":material/scatter_plot:")
    st.caption("Pearson correlation measures linear association; it does not establish causation.")
    x_label = st.selectbox("X indicator", VARS, index=1, format_func=lambda x: LABELS[x])
    y_label = st.selectbox("Y indicator", VARS, index=0, format_func=lambda x: LABELS[x])
    pair = df[[x_label,y_label,"Country","Year"]].dropna()
    pair = pair[np.isfinite(pair[[x_label,y_label]].to_numpy(dtype=float)).all(axis=1)]
    if len(pair)>=3 and pair[x_label].nunique()>1 and pair[y_label].nunique()>1:
        r = pair[x_label].corr(pair[y_label])
        st.metric("Pearson correlation (r)",f"{r:+.3f}",help="−1 = negative linear association; 0 = no linear association; +1 = positive linear association.")
        fig = px.scatter(
            pair,x=x_label,y=y_label,color="Country",hover_data=["Year"],
            trendline="ols",trendline_scope="overall",
            color_discrete_sequence=CHART_COLORS,
            title=f"Relationship between {LABELS[y_label]} and {LABELS[x_label]}",
        )
        fig.update_layout(xaxis_title=LABELS[x_label], yaxis_title=LABELS[y_label])
        st.plotly_chart(
            style(fig,480), width="stretch",
            alt=f"Relationship between {LABELS[y_label]} and {LABELS[x_label]}, with an overall fitted trend line.",
        )
        chart_caption("Each dot is a country-year record. The line summarizes the overall linear relationship; correlation does not prove causation.")
    else:
        st.info("Not enough paired variation in these indicators to calculate a correlation.")

    st.subheader("Descriptive statistics", icon=":material/analytics:")
    stats = df[VARS].describe().T
    stats.index = [LABELS[x] for x in stats.index]
    st.dataframe(stats, width="stretch", alt="Summary statistics for the selected climate indicators.")
    st.download_button(":material/download: Download filtered data (CSV)",df.to_csv(index=False).encode("utf-8"),
                       "climatelens_filtered.csv","text/csv")
