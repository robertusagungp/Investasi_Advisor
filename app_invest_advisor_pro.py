import os
import math
import json
import datetime as dt

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF

# =========================
# PAGE
# =========================
st.set_page_config(page_title="AI Investment Advisor Pro", layout="wide")

# -------------------------
# Language
# -------------------------
LANG = st.sidebar.selectbox("Language", ["Bahasa Indonesia", "English"], index=0)
IS_ID = LANG == "Bahasa Indonesia"

def T(id_text: str, en_text: str) -> str:
    return id_text if IS_ID else en_text

st.title(T("AI Investment Advisor Pro 💼", "AI Investment Advisor Pro 💼"))
st.caption(T(
    "Demo edukasi: rekomendasi strategi investasi berbasis profil pengguna (bukan nasihat finansial).",
    "Educational demo: investment strategy suggestions based on user profile (not financial advice)."
))

with st.expander(T("⚠️ Disclaimer", "⚠️ Disclaimer"), expanded=True):
    st.write(T(
        "- Ini hanya edukasi & simulasi.\n"
        "- Tidak ada jaminan return.\n"
        "- Selalu utamakan dana darurat dan kesehatan cashflow.\n"
        "- Untuk keputusan besar, konsultasikan dengan penasihat berlisensi.",
        "- Educational simulation only.\n"
        "- No guaranteed returns.\n"
        "- Prioritize emergency fund and cashflow health.\n"
        "- For major decisions, consult a licensed advisor."
    ))

# =========================
# SETTINGS
# =========================
st.sidebar.header(T("Pengaturan AI", "AI Settings"))

DEFAULT_MODEL_OLLAMA = "llama3"
DEFAULT_MODEL_GROQ = "llama3-70b-8192"

temperature = st.sidebar.slider(T("Kreativitas (temperature)", "Creativity (temperature)"), 0.0, 1.0, 0.35, 0.05)

use_ollama_first = st.sidebar.toggle(T("Coba Ollama dulu (lokal)", "Try Ollama first (local)"), value=True)
ollama_model = st.sidebar.text_input(T("Ollama model", "Ollama model"), value=DEFAULT_MODEL_OLLAMA)

groq_model = st.sidebar.text_input(T("Groq model", "Groq model"), value=DEFAULT_MODEL_GROQ)
st.sidebar.caption(T(
    "Cloud: Ollama biasanya tidak tersedia → otomatis fallback ke Groq (butuh GROQ_API_KEY).",
    "Cloud: Ollama usually unavailable → auto fallback to Groq (requires GROQ_API_KEY)."
))

# =========================
# RISK SCORE + RETURN RANGE (conservative ranges, not promises)
# =========================
RISK_BASE = {
    "Conservative": 25,
    "Moderate": 55,
    "Aggressive": 85
}

# Annual nominal ranges (very rough educational ranges)
RETURN_RANGE = {
    "Conservative": (0.03, 0.06),
    "Moderate": (0.05, 0.10),
    "Aggressive": (0.08, 0.16)
}

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def risk_score_engine(
    risk_profile: str,
    age: int,
    horizon_years: int,
    emergency_months: int,
    has_large_debt: bool,
    volatility_tol_0_10: int,
    time_per_week: str,
) -> int:
    """
    Simple heuristic: start from profile base then adjust.
    Produces 0-100, for convincing UI (not a scientific score).
    """
    score = RISK_BASE.get(risk_profile, 55)

    # age adjustment (younger can take more risk, older less)
    if age < 25:
        score += 5
    elif age >= 45:
        score -= 10

    # horizon adjustment
    if horizon_years >= 10:
        score += 8
    elif horizon_years <= 2:
        score -= 12

    # emergency fund & debt
    if emergency_months < 3:
        score -= 15
    if has_large_debt:
        score -= 10

    # volatility tolerance
    score += (volatility_tol_0_10 - 5) * 2  # -10..+10

    # time to monitor (more time = can handle slightly more)
    if time_per_week in ["7+ jam", "7+ hours"]:
        score += 4
    elif time_per_week in ["< 1 jam", "< 1 hour"]:
        score -= 3

    return int(clamp(score, 0, 100))

def compute_monthly_rate(annual_rate: float) -> float:
    return (1 + annual_rate) ** (1/12) - 1

def simulate_dca_growth(monthly_contribution: float, years: int, annual_rate: float):
    """
    Future value of recurring contributions with monthly compounding.
    """
    months = years * 12
    r = compute_monthly_rate(annual_rate)
    if r == 0:
        fv = monthly_contribution * months
    else:
        fv = monthly_contribution * (((1 + r) ** months - 1) / r)
    total_contrib = monthly_contribution * months
    return fv, total_contrib

def build_simulation_series(monthly_contribution: float, years: int, annual_rate: float):
    """
    Build month-by-month equity curve (for chart).
    """
    months = years * 12
    r = compute_monthly_rate(annual_rate)
    values = []
    v = 0.0
    for m in range(1, months + 1):
        v = v * (1 + r) + monthly_contribution
        values.append(v)
    return values

# =========================
# PDF EXPORT
# =========================
class SimplePDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "AI Investment Advisor Pro", ln=True)
        self.ln(2)

def make_pdf_bytes(title: str, body: str) -> bytes:
    pdf = SimplePDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_font("Arial", size=11)

    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 7, title)
    pdf.ln(2)

    pdf.set_font("Arial", size=11)
    for line in body.split("\n"):
        pdf.multi_cell(0, 6, line)

    return pdf.output(dest="S").encode("latin-1", errors="ignore")

# =========================
# LLM CALL: Ollama -> fallback Groq
# =========================
def llm_generate(prompt: str, temperature: float) -> str:
    # 1) Try Ollama first (if enabled)
    if use_ollama_first:
        try:
            import ollama
            resp = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": float(temperature)},
            )
            return resp["message"]["content"]
        except Exception:
            pass

    # 2) Fallback Groq
    groq_key = None
    # Streamlit Cloud secrets (preferred)
    try:
        groq_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        groq_key = None

    # Env var fallback
    if not groq_key:
        groq_key = os.environ.get("GROQ_API_KEY")

    if not groq_key:
        return T(
            "ERROR: GROQ_API_KEY belum di-set di Streamlit Secrets. Tambahkan:\nGROQ_API_KEY=\"gsk_...\"",
            "ERROR: GROQ_API_KEY not set in Streamlit Secrets. Add:\nGROQ_API_KEY=\"gsk_...\""
        )

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        completion = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"ERROR calling Groq: {e}"

# =========================
# UI INPUTS
# =========================
st.subheader(T("Profil Anda", "Your Profile"))

with st.form("profile_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        name = st.text_input(T("Nama", "Name"), value="")
        age = st.number_input(T("Umur", "Age"), min_value=18, max_value=80, value=25, step=1)
        gender = st.selectbox(T("Gender", "Gender"), ["Prefer not to say", "Male", "Female"])

    with c2:
        monthly_income = st.number_input(T("Pendapatan per bulan (Rp)", "Monthly income (IDR)"),
                                         min_value=0, value=10_000_000, step=500_000)
        monthly_invest = st.number_input(T("Budget investasi per bulan (Rp)", "Monthly investing budget (IDR)"),
                                         min_value=0, value=1_000_000, step=100_000)
        living_cost = st.number_input(T("Estimasi biaya hidup per bulan (Rp)", "Estimated monthly living cost (IDR)"),
                                      min_value=0, value=5_000_000, step=250_000)

    with c3:
        risk_profile = st.select_slider("Risk profile", options=["Conservative", "Moderate", "Aggressive"], value="Moderate")
        horizon_years = st.slider(T("Horizon (tahun)", "Horizon (years)"), 1, 30, 10)
        experience = st.selectbox(T("Pengalaman investasi", "Invest experience"), ["Pemula", "Menengah", "Mahir"] if IS_ID else ["Beginner", "Intermediate", "Advanced"])

    st.markdown("---")
    d1, d2, d3 = st.columns(3)

    with d1:
        emergency_months = st.slider(T("Dana darurat (bulan biaya hidup)", "Emergency fund (months of living cost)"), 0, 24, 3)
        has_large_debt = st.selectbox(T("Ada cicilan besar mengganggu cashflow?", "Any large debt affecting cashflow?"), [T("Tidak", "No"), T("Ya", "Yes")]) == T("Ya", "Yes")
        sharia = st.selectbox(T("Butuh syariah?", "Require Sharia-compliant?"), [T("Tidak", "No"), T("Ya", "Yes")])

    with d2:
        volatility_tol = st.slider(T("Toleransi fluktuasi (0-10)", "Volatility tolerance (0-10)"), 0, 10, 5)
        drawdown_reaction = st.selectbox(
            T("Jika turun -20%, Anda cenderung…", "If portfolio drops -20%, you tend to…"),
            [T("Panik jual", "Panic sell"),
             T("Diam tapi stres", "Hold but stressed"),
             T("Tetap rutin DCA", "Keep DCA"),
             T("Tambah beli", "Buy more")]
        )
        liquidity_need = st.selectbox(
            T("Butuh cair cepat?", "Need high liquidity?"),
            [T("Tinggi", "High"), T("Sedang", "Medium"), T("Rendah", "Low")]
        )

    with d3:
        time_per_week = st.selectbox(
            T("Waktu memantau per minggu", "Time to monitor per week"),
            ["< 1 jam", "1–3 jam", "3–7 jam", "7+ jam"] if IS_ID else ["< 1 hour", "1–3 hours", "3–7 hours", "7+ hours"]
        )
        goals = st.multiselect(
            T("Tujuan utama", "Primary goals"),
            [T("Dana darurat", "Emergency fund"),
             T("Beli rumah", "Buy a home"),
             T("Pendidikan", "Education"),
             T("Pensiun", "Retirement"),
             T("Freedom/wealth", "Wealth building"),
             T("Trading jangka pendek", "Short-term trading")],
            default=[T("Freedom/wealth", "Wealth building")]
        )
        notes = st.text_area(T("Catatan tambahan (opsional)", "Extra notes (optional)"), height=110)

    submitted = st.form_submit_button(T("✨ Generate rekomendasi", "✨ Generate recommendation"))

# =========================
# RUN
# =========================
if submitted:
    if monthly_invest <= 0:
        st.error(T("Budget investasi per bulan harus > 0.", "Monthly investment budget must be > 0."))
        st.stop()

    # compute risk score
    risk_score = risk_score_engine(
        risk_profile=risk_profile,
        age=int(age),
        horizon_years=int(horizon_years),
        emergency_months=int(emergency_months),
        has_large_debt=bool(has_large_debt),
        volatility_tol_0_10=int(volatility_tol),
        time_per_week=time_per_week
    )

    # simulate range
    rmin, rmax = RETURN_RANGE[risk_profile]
    fv_min, contrib = simulate_dca_growth(float(monthly_invest), int(horizon_years), rmin)
    fv_max, _ = simulate_dca_growth(float(monthly_invest), int(horizon_years), rmax)

    # series for chart (use mid rate)
    mid_rate = (rmin + rmax) / 2
    series = build_simulation_series(float(monthly_invest), int(horizon_years), mid_rate)
    months = list(range(1, len(series) + 1))

    # display score + simulation
    st.subheader(T("Skor Risiko (0–100)", "Risk Score (0–100)"))
    st.metric(T("Skor", "Score"), f"{risk_score}/100")

    st.subheader(T("Simulasi Pertumbuhan (Edukasi)", "Growth Simulation (Educational)"))
    st.caption(T(
        "Ini bukan janji return. Ini hanya ilustrasi berbasis range asumsi konservatif.",
        "No return guarantees. This is an illustration using conservative assumption ranges."
    ))

    cA, cB, cC = st.columns(3)
    with cA:
        st.write(T("Total kontribusi", "Total contribution"))
        st.write(f"Rp {int(contrib):,}".replace(",", "."))
    with cB:
        st.write(T("Estimasi konservatif", "Conservative estimate"))
        st.write(f"Rp {int(fv_min):,}".replace(",", "."))
    with cC:
        st.write(T("Estimasi optimis", "Optimistic estimate"))
        st.write(f"Rp {int(fv_max):,}".replace(",", "."))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=series, mode="lines", name=T("Proyeksi (mid)", "Projection (mid)")))
    fig.update_layout(
        height=360,
        xaxis_title=T("Bulan", "Month"),
        yaxis_title=T("Nilai Portofolio (Rp)", "Portfolio Value (IDR)"),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Build profile payload for LLM
    profile = {
        "name": name.strip() or "User",
        "age": int(age),
        "gender": gender,
        "monthly_income_idr": int(monthly_income),
        "monthly_invest_budget_idr": int(monthly_invest),
        "estimated_living_cost_idr": int(living_cost),
        "emergency_fund_months": int(emergency_months),
        "has_large_debt": bool(has_large_debt),
        "risk_profile": risk_profile,
        "risk_score_0_100": risk_score,
        "investment_horizon_years": int(horizon_years),
        "experience": experience,
        "goals": goals,
        "liquidity_need": liquidity_need,
        "volatility_tolerance_0_10": int(volatility_tol),
        "drawdown_reaction": drawdown_reaction,
        "sharia_required": sharia,
        "time_per_week_for_investing": time_per_week,
        "notes": notes.strip()
    }

    # Prompt design: structured & safe
    if IS_ID:
        system_brief = (
            "Anda adalah asisten edukasi investasi. Jangan memberi janji return atau rekomendasi spesifik ticker. "
            "Berikan strategi praktis sesuai profil Indonesia. Jika dana darurat <3 bulan atau ada utang besar, "
            "prioritaskan stabilisasi dulu."
        )
        output_format = """
Tulis output dengan format:
1) Ringkasan profil (1 paragraf)
2) Interpretasi skor risiko (0-100) + alasan
3) Rekomendasi alokasi (tabel) untuk: Cash/Deposito, Obligasi, Saham/ETF, Emas, Crypto
4) Strategi eksekusi: DCA + rebalancing + frekuensi
5) Aturan sederhana "Jika X maka Y" (3-6 aturan)
6) Checklist 7 hari ke depan
7) Disclaimer singkat
"""
        user_prompt = f"""
{system_brief}

Profil (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

Catatan simulasi:
- Total kontribusi: Rp {int(contrib):,}
- Range estimasi edukasi: Rp {int(fv_min):,} s/d Rp {int(fv_max):,}
Jelaskan bahwa simulasi hanya ilustrasi dan bukan janji return.

{output_format}
"""
    else:
        system_brief = (
            "You are an investment education assistant. Do not promise returns or recommend specific tickers. "
            "Provide practical strategies. If emergency fund <3 months or large debt exists, prioritize stabilization."
        )
        output_format = """
Output format:
1) Profile summary (1 paragraph)
2) Risk score interpretation (0-100) + reasons
3) Suggested allocation (table): Cash, Bonds, Stocks/ETF, Gold, Crypto
4) Execution plan: DCA + rebalancing + frequency
5) Simple rules “If X then Y” (3-6 rules)
6) 7-day checklist
7) Short disclaimer
"""
        user_prompt = f"""
{system_brief}

Profile (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

Simulation notes:
- Total contribution: IDR {int(contrib):,}
- Educational estimate range: IDR {int(fv_min):,} to IDR {int(fv_max):,}
Explain that simulation is only an illustration, not a return guarantee.

{output_format}
"""

    st.subheader(T("Rekomendasi AI", "AI Recommendation"))

    with st.spinner(T("Menghubungi AI...", "Contacting AI...")):
        ai_text = llm_generate(user_prompt, temperature=temperature)

    st.write(ai_text)

    # Download TXT
    st.download_button(
        T("Download hasil (TXT)", "Download result (TXT)"),
        data=ai_text.encode("utf-8"),
        file_name="investment_recommendation.txt",
        mime="text/plain"
    )

    # PDF content
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = T("Investment Recommendation Report", "Investment Recommendation Report")
    body = (
        f"{T('Generated at', 'Generated at')}: {now}\n"
        f"{T('Name', 'Name')}: {profile['name']}\n"
        f"{T('Risk score', 'Risk score')}: {risk_score}/100\n"
        f"{T('Horizon', 'Horizon')}: {horizon_years} {T('tahun', 'years')}\n"
        f"{T('Monthly budget', 'Monthly budget')}: Rp {int(monthly_invest):,}\n"
        "\n"
        + ai_text
    )

    try:
        pdf_bytes = make_pdf_bytes(title, body)
        st.download_button(
            T("Download PDF", "Download PDF"),
            data=pdf_bytes,
            file_name="investment_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.warning(T("Gagal membuat PDF. Coba ulang atau cek dependency fpdf2.", "Failed to generate PDF. Check fpdf2 dependency."))
        st.code(str(e))
