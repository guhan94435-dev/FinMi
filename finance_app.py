import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# Set Page Config
st.set_page_config(page_title="FinMi - Financial Intelligence Dashboard", layout="wide")

st.title("📊 FinMi: Strategic Financial Intelligence")
st.markdown("""
This AI-powered framework (FinMi) dynamically updates budgets and revenue predictions using machine learning.
""")
It analyzes historical records alongside macro-economic indicators to provide:
1. **Real-time Forecast Updates** | 2. **Risk Prediction** | 3. **Explainable AI Insights**
""")

# 1. SIDEBAR: DATA & PARAMETERS
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload 'financial_data.xlsx'", type=["xlsx"])

st.sidebar.header("🌍 Macro-Economic Variables (AI Inputs)")
inflation_rate = st.sidebar.slider("Annual Inflation Rate (%)", 0.0, 15.0, 6.5)
monsoon_impact = st.sidebar.select_slider("Monsoon Quality (Agricultural Demand)", options=["Drought", "Below Avg", "Normal", "Good", "Excellent"], value="Normal")
market_sentiment = st.sidebar.slider("Market Sentiment Index (PSU Sector)", 0, 100, 50)

# Factor mapping for AI Model
monsoon_map = {"Drought": 0.85, "Below Avg": 0.92, "Normal": 1.0, "Good": 1.08, "Excellent": 1.15}
monsoon_factor = monsoon_map[monsoon_impact]

if uploaded_file is not None:
    try:
        # Load Data
        df = pd.read_excel(uploaded_file)
        st.success("AI Framework Initialized: Historical Data & Macro-Factors Loaded.")
        
        # 2. CALCULATION ENGINE
        latest = df.iloc[-1]
        
        # Altman Z-Score Calculation
        A = latest['Working_Capital'] / latest['Total_Assets']
        B = latest['Net_Profit'] / latest['Total_Assets']
        C = (latest['Net_Profit'] + (latest['Revenue']*0.07)) / latest['Total_Assets']
        D = latest['Equity'] / latest['Total_Liabilities']
        E = latest['Revenue'] / latest['Total_Assets']
        z_score = (1.2*A) + (1.4*B) + (3.3*C) + (0.6*D) + (1.0*E)

        # Monte Carlo Simulation
        simulations = 5000
        success_count = 0
        for _ in range(simulations):
            sim_growth = np.random.uniform(0.05, 0.15)
            rev_future = latest['Revenue'] * (1 + sim_growth)**2
            if rev_future > (latest['Revenue'] * 0.95):
                success_count += 1
        prob = (success_count / simulations) * 100

        # AI Multi-factor Projection
        X = df[['Year']]
        model_rev = LinearRegression().fit(X, df['Revenue'])
        next_yr_val = df['Year'].max() + 1
        base_forecast = model_rev.predict(pd.DataFrame({'Year': [next_yr_val]}))[0]
        
        # AI-Powered Adjustment Logic
        inflation_impact = 1 - (inflation_rate / 100 * 0.2)
        sentiment_impact = 1 + ((market_sentiment - 50) / 500)
        ai_final_forecast = base_forecast * inflation_impact * monsoon_factor * sentiment_impact

        # 3. DISPLAY RESULTS (KPIs)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Altman Z-Score (Risk)", f"{z_score:.2f}")
            if z_score > 2.99: st.success("ZONE: SAFE")
            elif z_score >= 1.81: st.warning("ZONE: GREY")
            else: st.error("ZONE: DISTRESS")
            
        with col2:
            st.metric("Success Probability (2027)", f"{prob:.1f}%")
            st.progress(prob / 100)
            
        with col3:
            st.metric(f"AI-Adjusted Forecast ({next_yr_val})", f"₹ {ai_final_forecast:,.0f} L")
            delta = ((ai_final_forecast - base_forecast) / base_forecast) * 100
            st.caption(f"Adjustment for Macro-factors: {delta:+.1f}%")

        # --- EXPLAINABLE AI SECTION ---
        st.subheader("🤖 Explainable AI (XAI): Insight Generation")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.write("**How these factors changed your results:**")
            st.write(f"- **Inflation ({inflation_rate}%):** Reduces real purchasing power, adjusted forecast by -{(1-inflation_impact)*100:.1f}%")
            st.write(f"- **Monsoon ({monsoon_impact}):** Impacts industrial demand from agrarian sectors by { (monsoon_factor-1)*100:+.1f}%")
            st.write(f"- **Market Sentiment:** PSU Sector confidence index contributes { (sentiment_impact-1)*100:+.1f}% to the top-line projection.")
        
        with exp_col2:
            st.info(f"""
            **Early Warning Indicator:**
            The model suggests that if Inflation stays above 9% and Monsoon is '{monsoon_impact}', KEL's profitability recovery trajectory will be impacted by approximately {(1 - inflation_impact)*12:.1f} months.
            """)

        # --- NEW: STRATEGIC SENSITIVITY MATRIX ---
        st.subheader("📊 Strategic Sensitivity Matrix (Risk Map)")
        st.write("Visualizing Net Profit outcomes across varied Growth and Inflation scenarios.")
        
        growth_range = np.linspace(0.05, 0.25, 5)
        inf_range = np.linspace(0.04, 0.12, 5)
        matrix_data = []
        for g in growth_range:
            row = []
            for i in inf_range:
                # Profit Approximation Logic
                sim_p = ai_final_forecast * (1 + g - (i*0.5)) - (latest['Revenue'] * 0.90)
                row.append(sim_p)
            matrix_data.append(row)

        fig_map, ax_map = plt.subplots(figsize=(10, 5))
        sns.heatmap(matrix_data, annot=True, fmt=".0f", cmap="RdYlGn", 
                    xticklabels=[f"{x*100:.0f}%" for x in inf_range],
                    yticklabels=[f"{y*100:.0f}%" for y in growth_range], ax=ax_map)
        ax_map.set_xlabel("Inflation Rate")
        ax_map.set_ylabel("Sales Growth")
        st.pyplot(fig_map)

        # --- NEW: AI GOAL SEEK ASSISTANT ---
        st.subheader("🎯 AI Goal Seek: Strategic Target Setter")
        target_profit = st.number_input("Set Target Net Profit (₹ Lakhs):", value=0.0)

        if st.button("Calculate Required Growth"):
            # Required Growth = (Profit + Costs)/Revenue - 1
            # Assuming costs are roughly 92% of current revenue for this model
            required_growth = (target_profit + (latest['Revenue'] * 0.92)) / latest['Revenue'] - 1
            st.markdown(f"**AI Strategy:** To reach a profit of ₹{target_profit}L, you need a Sales Growth of **{required_growth*100:.1f}%**.")
            if required_growth > 0.25:
                st.warning("⚠️ High Risk: This growth rate is significantly above historical averages.")
            else:
                st.success("✅ Realistic Target: This growth rate aligns with current turnaround trends.")

        # --- CALCULATION TRANSPARENCY SECTION ---
        st.markdown("---")
        with st.expander("🔍 VIEW CALCULATION LOGIC & MATHEMATICAL BREAKDOWN"):
            st.subheader("1. Altman Z-Score Formula Details")
            st.latex(r"Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5")
            
            calc_data = {
                "Ratio": ["X1 (Liquidity)", "X2 (Profitability)", "X3 (Productivity)", "X4 (Solvency)", "X5 (Efficiency)"],
                "Calculation": ["Working Cap / Assets", "Retained Earnings / Assets", "EBIT / Assets", "Equity / Liabilities", "Sales / Assets"],
                "Raw Value": [f"{A:.4f}", f"{B:.4f}", f"{C:.4f}", f"{D:.4f}", f"{E:.4f}"],
                "Weighted Contribution": [f"{1.2*A:.4f}", f"{1.4*B:.4f}", f"{3.3*C:.4f}", f"{0.6*D:.4f}", f"{1.0*E:.4f}"]
            }
            st.table(pd.DataFrame(calc_data))
            st.write(f"**Final Summed Z-Score:** {z_score:.4f}")

            st.subheader("2. Monte Carlo Simulation Parameters")
            st.info(f"""
            - **Iterations:** 5,000 randomized scenarios
            - **Growth Range:** 5% to 15% (Uniform Distribution)
            - **Model Type:** Iterative Probabilistic Forecasting
            - **Success Condition:** Revenue at Year(t+2) > Current Sales Baseline
            """)

            st.subheader("3. Linear Regression (Trendline) Equation")
            slope = model_rev.coef_[0]
            intercept = model_rev.intercept_
            st.latex(fr"y = {slope:.2f}x + ({intercept:.2f})")
        st.markdown("---")

        # 4. VISUALIZATION
        st.subheader("📈 Strategic Growth & Profitability Trends")
        fig_trend, ax_trend = plt.subplots(1, 2, figsize=(15, 6))
        ax_trend[0].plot(df['Year'], df['Revenue'], 'g-o', linewidth=2)
        ax_trend[0].set_title('Revenue Trend')
        ax_trend[0].grid(True, linestyle='--', alpha=0.7)
        ax_trend[1].plot(df['Year'], df['Net_Profit'], 'r-o', linewidth=2)
        ax_trend[1].axhline(0, color='black', linewidth=1)
        ax_trend[1].set_title('Net Profit / Loss Trend')
        ax_trend[1].grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_trend)

        # 5. STRATEGIC INSIGHTS
        st.subheader("💡 Strategic Advisory")
        if z_score < 1.81:
            st.info("**High Priority Action:** Company shows signs of financial distress. Recommend immediate debt restructuring and working capital optimization.")
        else:
            st.info("**Growth Advisory:** Company is financially stable. Focus on market expansion and Capex investments.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Ensure your Excel file has columns: Year, Revenue, Net_Profit, Total_Assets, Total_Liabilities, Working_Capital, Equity")

else:
    st.info("Please upload an Excel file from the sidebar to begin analysis.")
