
import streamlit as st
import pandas as pd
import plotly.express as px


# Page settings
st.set_page_config(
    page_title="Heart Disease Analysis",
    page_icon="❤️",
    layout="wide"
)


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease.csv")

    # Handle missing values in categorical columns
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].fillna("Unknown")

    return df


df = load_data()


# Dashboard heading
st.title("❤️ Heart Disease Analysis Dashboard")
st.write("Explore patient health indicators and heart disease patterns.")
st.divider()


# SIDEBAR FILTERS
st.sidebar.header("Filter Patient Records")

age_range = st.sidebar.slider(
    "Select Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

gender = st.sidebar.multiselect(
    "Select Gender",
    sorted(df["Gender"].unique()),
    default=sorted(df["Gender"].unique())
)

smoking = st.sidebar.multiselect(
    "Select Smoking Status",
    sorted(df["Smoking"].unique()),
    default=sorted(df["Smoking"].unique())
)

exercise = st.sidebar.multiselect(
    "Select Exercise Habits",
    sorted(df["Exercise Habits"].unique()),
    default=sorted(df["Exercise Habits"].unique())
)

disease = st.sidebar.multiselect(
    "Heart Disease Status",
    sorted(df["Heart Disease Status"].unique()),
    default=sorted(df["Heart Disease Status"].unique())
)


# APPLY FILTERS
filtered_df = df[
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Gender"].isin(gender)) &
    (df["Smoking"].isin(smoking)) &
    (df["Exercise Habits"].isin(exercise)) &
    (df["Heart Disease Status"].isin(disease))
]


# CREATE TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Data Analysis",
    "Visualizations",
    "Insights"
])


# TAB 1: DATASET OVERVIEW
with tab1:

    st.header("Dataset Overview")

    st.write("This dataset contains 10,000 patient records and 21 health-related attributes.")

    # Calculate KPIs
    total_records = len(filtered_df)
    disease_cases = (filtered_df["Heart Disease Status"] == "Yes").sum()

    average_age = filtered_df["Age"].mean()
    average_cholesterol = filtered_df["Cholesterol Level"].mean()

    total_triglycerides = filtered_df["Triglyceride Level"].sum()

    # Display KPI cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Patients", total_records)
    col2.metric("Heart Disease Cases", disease_cases)

    col3.metric(
        "Average Age",
        f"{average_age:.2f}" if pd.notna(average_age) else "N/A"
    )

    col4.metric(
        "Average Cholesterol",
        f"{average_cholesterol:.2f}"
        if pd.notna(average_cholesterol) else "N/A"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Triglyceride Level")
        st.metric("Sum", f"{total_triglycerides:,.2f}")

    with col2:
        st.subheader("Heart Disease Distribution")

        if total_records > 0:
            disease_count = filtered_df["Heart Disease Status"].value_counts()

            fig = px.pie(
                values=disease_count.values,
                names=disease_count.index,
                title="Heart Disease Status"
            )

            st.plotly_chart(fig, use_container_width=True)

    # Display records
    with st.expander("View Patient Records"):
        st.dataframe(filtered_df, use_container_width=True)

    with st.expander("View Dataset Information"):
        st.write("Original Dataset Shape:", df.shape)
        st.write("Filtered Dataset Shape:", filtered_df.shape)
        st.write("Missing Values:")
        st.dataframe(df.isnull().sum(), use_container_width=True)


# TAB 2: DATA ANALYSIS
with tab2:

    st.header("Data Analysis")

    if filtered_df.empty:
        st.warning("No records found. Please change the filters.")

    else:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Numerical Data Summary")

            numerical_data = filtered_df.select_dtypes(include="number")

            st.dataframe(
                numerical_data.describe().round(2),
                use_container_width=True
            )

        with col2:
            st.subheader("Category-wise Record Counts")

            category = st.selectbox(
                "Choose a category",
                [
                    "Gender",
                    "Smoking",
                    "Exercise Habits",
                    "Diabetes",
                    "Stress Level",
                    "Heart Disease Status"
                ]
            )

            category_count = filtered_df.groupby(category).size()

            st.dataframe(
                category_count.rename("Number of Patients"),
                use_container_width=True
            )

        st.divider()

        st.subheader("Numerical Attribute Analysis")

        selected_column = st.selectbox(
            "Select a numerical attribute",
            [
                "Age",
                "Blood Pressure",
                "Cholesterol Level",
                "BMI",
                "Sleep Hours",
                "Triglyceride Level",
                "Fasting Blood Sugar",
                "CRP Level",
                "Homocysteine Level"
            ]
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Average",
            f"{filtered_df[selected_column].mean():.2f}"
        )

        col2.metric(
            "Total",
            f"{filtered_df[selected_column].sum():,.2f}"
        )


# TAB 3: VISUALIZATIONS
with tab3:

    st.header("Data Visualizations")

    if filtered_df.empty:
        st.warning("No records available. Please change the filters.")

    else:

        # BAR CHART
        st.subheader("1. Bar Chart")

        bar_category = st.selectbox(
            "Select category for Bar Chart",
            ["Gender", "Smoking", "Exercise Habits", "Diabetes"]
        )

        bar_data = filtered_df.groupby(
            [bar_category, "Heart Disease Status"]
        ).size().reset_index(name="Patient Count")

        fig1 = px.bar(
            bar_data,
            x=bar_category,
            y="Patient Count",
            color="Heart Disease Status",
            barmode="group",
            title="Heart Disease Distribution by Category"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.caption(
            "Observation: This chart compares heart disease cases "
            "across different categories."
        )

        st.divider()

        # LINE CHART
        st.subheader("2. Line Chart")

        line_data = filtered_df.dropna(
            subset=["Age", "Cholesterol Level"]
        ).copy()

        if not line_data.empty:

            # Group ages into decades
            line_data["Age Group"] = (
                (line_data["Age"] // 10) * 10
            ).astype(int)

            line_data["Age Group"] = (
                line_data["Age Group"].astype(str) + "s"
            )

            line_summary = line_data.groupby(
                "Age Group"
            )["Cholesterol Level"].mean().reset_index()

            line_summary["Age Number"] = (
                line_summary["Age Group"].str.replace("s", "").astype(int)
            )

            line_summary = line_summary.sort_values("Age Number")

            fig2 = px.line(
                line_summary,
                x="Age Group",
                y="Cholesterol Level",
                markers=True,
                title="Average Cholesterol Across Age Groups",
                labels={
                    "Age Group": "Age Group",
                    "Cholesterol Level": "Average Cholesterol"
                }
            )

            st.plotly_chart(fig2, use_container_width=True)

            st.caption(
                "Observation: This chart shows how average cholesterol "
                "levels vary across age groups."
            )

        st.divider()

        # SCATTER PLOT
        st.subheader("3. Scatter Plot")

        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox(
                "Select X-axis",
                ["Age", "Blood Pressure", "BMI", "Sleep Hours"]
            )

        with col2:
            y_axis = st.selectbox(
                "Select Y-axis",
                [
                    "Cholesterol Level",
                    "Triglyceride Level",
                    "Fasting Blood Sugar",
                    "CRP Level"
                ]
            )

        scatter_data = filtered_df.dropna(
            subset=[x_axis, y_axis]
        )

        fig3 = px.scatter(
            scatter_data,
            x=x_axis,
            y=y_axis,
            color="Heart Disease Status",
            title=x_axis + " vs " + y_axis
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "Observation: This chart helps examine the relationship "
            "between two numerical health attributes."
        )


# TAB 4: INSIGHTS
with tab4:

    st.header("Key Insights")

    if filtered_df.empty:
        st.warning("No records available for the selected filters.")

    else:

        total = len(filtered_df)

        disease_yes = (
            filtered_df["Heart Disease Status"] == "Yes"
        ).sum()

        disease_percentage = (disease_yes / total) * 100

        st.subheader("Patient Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Patients", total)
        col2.metric("Heart Disease Cases", disease_yes)
        col3.metric("Heart Disease Percentage", f"{disease_percentage:.2f}%")

        st.divider()

        st.subheader("Observations")

        st.write(
            "1. The selected dataset contains",
            total,
            "patient records."
        )

        st.write(
            "2. The number of patients with heart disease is",
            disease_yes,
            "."
        )

        st.write(
            "3. The average patient age is",
            round(filtered_df["Age"].mean(), 2),
            "years."
        )

        st.write(
            "4. The average cholesterol level is",
            round(filtered_df["Cholesterol Level"].mean(), 2),
            "."
        )

        st.write(
            "5. Use the filters and charts to explore patterns "
            "in different patient groups."
        )

        with st.expander("View Category-wise Analysis"):

            for column in [
                "Gender",
                "Smoking",
                "Exercise Habits",
                "Diabetes",
                "Heart Disease Status"
            ]:

                st.write("**" + column + "**")

                st.write(filtered_df[column].value_counts())


# FOOTER
st.divider()

st.caption("Heart Disease Analysis Dashboard | Dixon Benoy")
