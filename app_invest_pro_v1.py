import os
import json
import datetime as dt

import streamlit as st
import plotly.graph_objects as go

from fpdf import FPDF

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Investment Advisor Pro",
    layout="wide"
)

# =========================
# LANGUAGE
# =========================

LANG = st.sidebar.selectbox(
    "Language",
    ["Bahasa Indonesia", "English"],
    index=0
)

IS_ID = LANG == "Bahasa Indonesia"

def T(id_text, en_text):
    return id_text if IS_ID else en_text


st.title("AI Investment Advisor Pro 💼")

st.caption(T(
    "Demo edukasi. Bukan nasihat finansial.",
    "Educational demo. Not financial advice."
))


# =========================
# SETTINGS
# =========================

DEFAULT_MODEL_GROQ = "llama-3.3-70b-versatile"

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.3
)

groq_model = st.sidebar.text_input(
    "Groq Model",
    DEFAULT_MODEL_GROQ
)


# =========================
# RISK ENGINE
# =========================

RISK_BASE = {

    "Conservative": 25,
    "Moderate": 55,
    "Aggressive": 85
}


def clamp(x, lo, hi):

    return max(lo, min(hi, x))


def risk_score_engine(profile, age, horizon, emergency, debt, volatility):

    score = RISK_BASE.get(profile, 55)

    if age < 25:
        score += 5

    if age > 45:
        score -= 10

    if horizon >= 10:
        score += 8

    if emergency < 3:
        score -= 15

    if debt:
        score -= 10

    score += (volatility - 5) * 2

    return clamp(score, 0, 100)


# =========================
# AUTO PORTFOLIO ALLOCATION ENGINE
# =========================

def auto_allocation(risk_score):

    """
    Output allocation percentage automatically.
    """

    if risk_score <= 35:

        return {

            "Cash / Deposito": 40,
            "Bonds / Obligasi": 35,
            "Stocks / Saham": 15,
            "Gold": 8,
            "Crypto": 2
        }

    elif risk_score <= 65:

        return {

            "Cash / Deposito": 20,
            "Bonds / Obligasi": 30,
            "Stocks / Saham": 35,
            "Gold": 10,
            "Crypto": 5
        }

    else:

        return {

            "Cash / Deposito": 10,
            "Bonds / Obligasi": 15,
            "Stocks / Saham": 55,
            "Gold": 10,
            "Crypto": 10
        }


# =========================
# ASSET DATABASE
# =========================

ASSETS = {

    "Conservative": {

        "stocks": ["BBCA", "TLKM", "UNVR"],
        "crypto": ["BTC", "ETH"]
    },

    "Moderate": {

        "stocks": ["BBCA", "BBRI", "ASII", "ICBP"],
        "crypto": ["BTC", "ETH", "SOL"]
    },

    "Aggressive": {

        "stocks": ["BBRI", "ADRO", "GOTO", "MDKA"],
        "crypto": ["BTC", "ETH", "SOL", "AVAX"]
    }

}


# =========================
# GROWTH SIMULATION
# =========================

def simulate(monthly, years, rate):

    r = rate / 12

    value = 0

    series = []

    for i in range(years * 12):

        value = value * (1 + r) + monthly

        series.append(value)

    return series


# =========================
# PDF ENGINE FIX
# =========================

class PDF(FPDF):

    def header(self):

        self.set_font("Arial", "B", 14)

        self.cell(0, 10, "AI Investment Advisor Pro", ln=True)


def sanitize(text):

    return text.encode("latin-1", "replace").decode("latin-1")


def make_pdf(title, body):

    pdf = PDF()

    pdf.add_page()

    pdf.set_font("Arial", size=11)

    pdf.multi_cell(0, 6, sanitize(title))

    pdf.ln(3)

    pdf.multi_cell(0, 6, sanitize(body))

    result = pdf.output(dest="S")

    # CRITICAL FIX:
    if isinstance(result, bytes):
        return result
    else:
        return result.encode("latin-1", "replace")



# =========================
# LLM CALL
# =========================

def call_groq(prompt):

    try:

        from groq import Groq

        key = st.secrets["GROQ_API_KEY"]

        client = Groq(api_key=key)

        completion = client.chat.completions.create(

            model=groq_model,

            messages=[

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            temperature=temperature

        )

        return completion.choices[0].message.content

    except Exception as e:

        return str(e)


# =========================
# UI INPUT
# =========================

with st.form("form"):

    name = st.text_input("Name")

    age = st.number_input("Age", 18, 80, 25)

    income = st.number_input("Monthly income", 0, 100000000, 10000000)

    invest = st.number_input("Monthly investment", 0, 100000000, 1000000)

    risk_profile = st.selectbox(

        "Risk Profile",

        ["Conservative", "Moderate", "Aggressive"]

    )

    horizon = st.slider("Horizon (years)", 1, 30, 10)

    emergency = st.slider("Emergency fund (months)", 0, 24, 3)

    volatility = st.slider("Volatility tolerance", 0, 10, 5)

    debt = st.checkbox("Large debt")

    submit = st.form_submit_button("Generate")


# =========================
# RUN
# =========================

if submit:

    score = risk_score_engine(

        risk_profile,

        age,

        horizon,

        emergency,

        debt,

        volatility

    )

    st.subheader("Risk Score")

    st.metric("Score", score)


    # allocation

    alloc = auto_allocation(score)

    st.subheader("Auto Portfolio Allocation")

    st.json(alloc)


    # assets

    stocks = ASSETS[risk_profile]["stocks"]

    crypto = ASSETS[risk_profile]["crypto"]


    st.subheader("Recommended Stocks")

    st.write(stocks)


    st.subheader("Recommended Crypto")

    st.write(crypto)


    # simulate

    series = simulate(invest, horizon, 0.08)


    fig = go.Figure()

    fig.add_trace(go.Scatter(y=series))

    st.plotly_chart(fig)


    # LLM

    prompt = f"""

    User profile:

    risk score: {score}

    allocation: {alloc}

    give investment advice

    """

    ai = call_groq(prompt)

    st.subheader("AI Recommendation")

    st.write(ai)


    # PDF

    body = f"""

Name: {name}

Risk score: {score}

Allocation: {alloc}

Stocks: {stocks}

Crypto: {crypto}

{ai}

"""

    pdf = make_pdf("Investment Report", body)

    st.download_button(

        "Download PDF",

        pdf,

        "report.pdf"

    )
