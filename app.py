
import streamlit as st
import joblib 
from PIL import Image
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go


icon  = Image.open("churn_icon.png")

# Configure the application
st.set_page_config(
    page_title = "Customer Churn Prediction",
    page_icon = icon,
    layout = "wide"
)

# Adjust page spacing
st.markdown("""
    <style>
        .block-container {
            padding-top: 0.5rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 1600px;
        }
    </style>
""", unsafe_allow_html=True)

# Display banner

BASE_DIR = Path(__file__).parent

banner = Image.open(BASE_DIR / "churn_banner.png")

banner = banner.resize((1400,400))

st.image(banner, use_container_width=True)


# Display the application description
st.write(
    "Predict whether a telecom customer is likely to churn using trained Logistic Regression model."
)

st.info("Enter customers details to generate a churn prediction.")


# Load the trained Logistic Regression model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "Churn_model.pkl"
    return joblib.load(model_path)

model = load_model()

st.success("Trained Logistic Regression model loaded successfully!")



# Create tabs for prediction and historical analytics
tab1, tab2 = st.tabs(["🎯 Churn Prediction", "📊 Customer Analytics"])

with tab1:
    # Customer Information(identification and personal details)
    st.subheader("Customer Information")
    
    # Customer identification
    st.markdown("##### Customer Identification")
    col_id, col_name = st.columns(2)

    with col_id:
        customer_id = st.text_input(
            "Customer ID *",
            placeholder="Enter customer ID"
            )
     

    with col_name:
        customer_name = st.text_input(
        "Customer Name",
        placeholder="Enter customer name"
    )

   #st.divider()

    # Customer Personal details

    st.markdown("##### Personal Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])

    with col2:
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=2
        )  




    # Phone and internet services

    st.subheader("Phone and Internet Services")

    col3, col4 = st.columns(2)

    with col3:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        if phone_service == "No":
            multiple_lines = "No phone service" 
            st.info("Multiple Lines: No phone service")

        else:
            multiple_lines = st.selectbox("Muliple Lines", ["Yes", "No"])

    with col4:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])  


        
    # Additional internet Services
    st.subheader("Additional Internet Services")

    col5, col6 = st.columns(2)

    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

        st.info("All Additional Services: No internet service")

    else:
        service_options = ["Yes", "No"]

        with col5:
            online_security = st.selectbox("Online Security", service_options)
            online_backup = st.selectbox("Online Backup", service_options)
            device_protection = st.selectbox("Device Protection", service_options)

        with col6:
            tech_support = st.selectbox("Tech Support", service_options)
            streaming_tv = st.selectbox("Streaming Tv", service_options)
            streaming_movies = st.selectbox("Streaming Movies", service_options)


    # Contract and billing information

    st.subheader("Contract and Billing Information")

    col7, col8 = st.columns(2)

    with col7:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer(automatic)", 
            "Credit card(automatic)"])

    with col8:
        monthly_charges = st.number_input(
            "Monthly Charges($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col8:
        total_charges = st.number_input(
            "Total Charges($)",
            min_value=0.0,
            value=140.0,
            step=1.0
        ) 



    # Prepare customer data for prediction
    customer_data = pd.DataFrame([{
        "gender" : gender,
        "SeniorCitizen" : 1 if senior_citizen == "Yes" else 0,
        "Partner" : partner,
        "Dependents" : dependents,
        "tenure" : tenure,
        "PhoneService" : phone_service,
        "MultipleLines" : multiple_lines,
        "InternetService" : internet_service,
        "OnlineSecurity" : online_security,
        "OnlineBackup" : online_backup,
        "DeviceProtection" : device_protection,
        "TechSupport" : tech_support,
        "StreamingTV" : streaming_tv,
        "StreamingMovies" : streaming_movies,
        "Contract" : contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Display customer data for verification
    st.subheader("Customer Data Preview")
    st.dataframe(customer_data, hide_index=True)



    # Predict customer churn
    st.subheader("Churn Prediction")

    if st.button("Predict Churn", type="primary"):

        if not customer_id.strip():
            st.error("Required field: Please enter a customer ID before making a prediction")
            st.stop()

    # Display customer identification
        st.markdown("#### Prediction for Customer")

        st.write("**Customer ID:**", customer_id or "Not provided")
        st.write("**Customer Name:**", customer_name or "Not provided")

        prediction = model.predict(customer_data)[0]
        
        churn_index = list(model.classes_).index("Yes")
        churn_probability = model.predict_proba(customer_data)[0][churn_index]

        if prediction == "Yes":
            st.error("Prediction: Customer is likely to churn.")

        else:
            st.success("Prediction: Customer is likely to stay.")

        st.metric(
            label="Predicted Churn Probability",
            value=f"{churn_probability:.1%}"
        )


        # Display churn probability with risk-based color
        if churn_probability < 0.30:
            progress_color = "#087F8C"  # Teal - Low Risk
        elif churn_probability < 0.50:
            progress_color = "#F4A340"  # Orange - Moderate Risk
        else:
            progress_color = "#E05260"  # Coral - High Risk

        st.markdown(
            f"""
            <div style="
                background-color: #F0F2F6;
                border-radius: 10px;
                height: 12px;
                width: 100%;
                overflow: hidden;
            ">
                <div style="
                    background-color: {progress_color};
                    width: {churn_probability * 100}%;
                    height: 100%;
                    border-radius: 10px;
                "></div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # Display customer churn risk level
        if churn_probability <0.30:
            risk_level = "Low Risk"
            st.success(f"Customer Risk Level: {risk_level}")

        elif churn_probability <0.50:
            risk_level = "Moderate Risk"
            st.warning(f"Customer Risk Level: {risk_level}")

        else:
            risk_level = "High Risk"
            st.error(f"Customer Risk Level: {risk_level}") 



#Customer analytics dashboard

with tab2:
    st.subheader("Customer Analytics Dashboard")

    st.write(
        "Explore historical customer churn patterns "
        "and key insights from the Telco Customer Churn dataset."
    )

    # load historical customer data
    @st.cache_data
    def load_customer_data():
        data_path = BASE_DIR / "Telco-Customer-Churn.csv"
        return pd.read_csv(data_path)

    historical_df = load_customer_data()


    # Calculate historical customer metrics
    total_customers = len(historical_df)
    churned_customers = (historical_df["Churn"] == "Yes").sum()
    retained_customers = (historical_df["Churn"] == "No").sum()
    churn_rate = churned_customers / total_customers

    # Display KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")

    with kpi1:
        with st.container(border=True):
            st.metric("Total Customers", f"{total_customers:,}")

    with kpi2:
        with st.container(border=True):
            st.metric("Churn Rate", f"{churn_rate:.1%}")  

    with kpi3:
        with st.container(border=True):
            st.metric("Retained Customers", f"{retained_customers:,}")

    with kpi4: 
        with st.container(border=True):
            st.metric("Churned Customers", f"{churned_customers:,}")

    # Create two columns for dashboard charts
    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        # Historical churn distribution
        st.subheader("Historical Churn Distribution")

        churn_counts = historical_df["Churn"].value_counts()

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Retained", "Churned"],
                    values=[
                        churn_counts["No"],
                        churn_counts["Yes"]
                    ],
                    hole=0.68,
                    marker=dict(
                        colors=["#087F8C", "#E05260"]
                    ),
                    textinfo="percent",
                    hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>)"
                )
            ]
        )

        fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#142B49"),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.05
        ),
        annotations=[
            dict(
                text="26.5%<br>Churn Rate",
                x=0.5,
                y=0.5,
                font=dict(size=20, color="#142B49"),
                showarrow=False
            )
        ]
    )

        st.plotly_chart(fig, use_container_width=True)


    # Churn rate by contract type
    with chart_col2:
        st.subheader("Churn Rate by Contract Type")

        contract_churn = (
        historical_df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )
        fig_contract = go.Figure()

        fig_contract.add_trace(go.Bar(
                x=contract_churn["Contract"],
                y=contract_churn["Churn Rate"],
                marker_color="#E05260",
                width=0.35,
                text=contract_churn["Churn Rate"].round(1),
                texttemplate="%{text}%",
                textposition="outside"
            )
        )

        fig_contract.update_layout(
            yaxis_title="Churn Rate (%)",
            xaxis_title="",
            yaxis_range=[0, 55],
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False
        )

        st.plotly_chart(fig_contract, use_container_width=True)

    # Create a second row of dashboard charts
    chart_col3, chart_col4 = st.columns(2)  

    with chart_col3:
        st.subheader("Churn By Customer Tenure")

        historical_df["Tenure_Group"] = pd.cut(
            historical_df["tenure"], 
            bins=[-1,12,24,48,72],
            labels=["0-12 months", "13-24 months", " 25-48 months", "49-72 months"]
        ) 

        tenure_churn = pd.crosstab(
            historical_df["Tenure_Group"],
            historical_df["Churn"],
            normalize="index"
        ) * 100

        fig_tenure = go.Figure()

        fig_tenure.add_trace(go.Bar(
            x=tenure_churn.index.astype(str),
            y=tenure_churn["No"],
            name="Retained",
            marker_color="#087F8C"
        ))

        fig_tenure.add_trace(go.Bar(
            x=tenure_churn.index.astype(str),
            y=tenure_churn["Yes"],
            name="Churned",
            marker_color="#E05260"
        ))


        fig_tenure.update_layout(
            barmode="group",
            yaxis_title="Customers(%)",
            xaxis_title="Customer Tenure",
            yaxis_range=[0,100],
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig_tenure, use_container_width=True)

    with chart_col4:
        st.subheader("Churn Rate by Internet Service")

        # Calculate retained and churned customers percentage
        internet_churn = pd.crosstab(
            historical_df["InternetService"],
            historical_df["Churn"],
            normalize="index"
        ) * 100

        # create chart
        fig_internet = go.Figure()

        # Retained customers
        fig_internet.add_trace(go.Bar(
            x=internet_churn.index,
            y=internet_churn["No"],
            name="Retained",
            marker_color="#087F8C",
            text=internet_churn["No"].round(1),
            texttemplate="%{text}%",
            textposition="outside"

        ))  

        #Churned customers
        fig_internet.add_trace(go.Bar(
            x=internet_churn.index,
            y=internet_churn["Yes"],
            name="Churned",
            marker_color="#E05260",
            text=internet_churn["Yes"].round(1),
            texttemplate="%{text}%",
            textposition="outside"
        )) 

        # Chart layout
        fig_internet.update_layout(
            barmode="group",
            xaxis_title="Internet Service",
            yaxis_title="Customers (%)",
            yaxis=dict(range=[0, 110]),
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#17365D"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig_internet, use_container_width=True) 

        # Group customers by monthly charges
        historical_df["Monthly_Charges_Group"] = pd.cut(
            historical_df["MonthlyCharges"],
            bins=[0, 35, 70, 100, 120],
            labels=["$0-35", "$35-70", "$70-100", "$100-120"],
            include_lowest=True
        )

        
        
    # Create third row
    chart_col5, chart_col6 = st.columns([1, 1], gap="large")

    with chart_col5:
        st.subheader("Churn Rate by Monthly Charges")

        monthly_churn = pd.crosstab(
            historical_df["Monthly_Charges_Group"],
            historical_df["Churn"],
            normalize="index"
        ) * 100

        fig_monthly = go.Figure()

        fig_monthly.add_trace(go.Bar(
            x=monthly_churn.index.astype(str),
            y=monthly_churn["No"],
            name="Retained",
            marker_color="#087F8C",
            text=monthly_churn["No"].round(1),
            texttemplate="%{text}%",
            textposition="outside"
        ))

        fig_monthly.add_trace(go.Bar(
            x=monthly_churn.index.astype(str),
            y=monthly_churn["Yes"],
            name="Churned",
            marker_color="#E05260",
            text=monthly_churn["Yes"].round(1),
            texttemplate="%{text}%",
            textposition="outside"
        ))

        fig_monthly.update_layout(
            barmode="group",
            yaxis_title="Customers (%)",
            xaxis_title="Monthly Charges",
            yaxis=dict(range=[0, 110]),
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#142B49"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig_monthly, use_container_width=True)


    with chart_col6:
        st.subheader("Customer Distribution by Monthly Charges")

        monthly_counts = historical_df["Monthly_Charges_Group"].value_counts()
        monthly_counts = monthly_counts.reindex(
            ["$0-35", "$35-70", "$70-100", "$100-120"]
        )

        fig_monthly_dist = go.Figure(
            data=[go.Bar(
                    x=monthly_counts.index,
                    y=monthly_counts.values,
                    marker_color="#087F8C",
                    width=0.45,
                    text=monthly_counts.values,
                    textposition="outside",
                    hovertemplate="%{x}<br>Customers: %{y:,}<extra></extra>"
                )
            ]
        )

        fig_monthly_dist.update_layout(
            xaxis_title="Monthly Charges Range",
            yaxis=dict(
                title="Number of Customers",
                range=[0, 3000]
            ),
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#17365D"),
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )

        st.plotly_chart(fig_monthly_dist, use_container_width=True)

    # Key business sights
    st.divider()

    st.subheader("💡Key Business Insights")

    st.write("Explore the main factors associated with customer churn "
            "and identify opportunities to improve customer retention.")

    # Display key insights
    insight1, insight2 = st.columns(2, gap="large")

    with insight1:
        with st.container(border=True):
            st.markdown("**1. Contract Type Matters**")
            st.write("Month-to-month customers have a much higher "
                    "churn rate (42.7%) compared to one-year (11.3%) "
                    "and two-year contracts (2.8%).")

    with insight2:
        with st.container(border=True):
            st.markdown("**2. Customer Tenure**")
            st.write("Customers who leave have a shorter average tenure "
                    "(17.98 months) than customers who stay "
                    "(37.57 months).") 

    insight3, insight4 = st.columns(2, gap="large")

    with insight3:
        with st.container(border=True):
            st.markdown("**3. Monthly Charges**")
            st.write("Customers paying $70–100 per month show the "
                    "highest churn rate (37.8%) among the monthly "
                    "charge groups.")

    with insight4:
        with st.container(border=True):
            st.markdown("**4. Customer Retention Opportunity**")
            st.write("Focus on early customer engagement, encourage " 
                    "longer-term contracts and offer targeted retention "
                    "incentives to customers with a high risk of churn.")


