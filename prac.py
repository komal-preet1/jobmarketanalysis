# app_updated.py
"""
Job Market Analysis Dashboard - Updated Version (uses user's Jooble scraper)
Submitted by: Komalpreet Kaur
Submitted to: Mrs. Bindu
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit_echarts import st_echarts
import plotly.express as px
from bs4 import BeautifulSoup
import time
import traceback

# Selenium + Chrome imports (user's scraper code relies on these)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------- JOOBLE SELENIUM SCRAPER (user-provided) -----------------------
def scrape_jooble_tuned(pages=3, delay=1, headless=False, timeout=12, debug=False):
    """
    Scrape India job listings from Jooble using selectors tuned to the provided HTML snippet.
    Returns DataFrame with columns: ['title','experience','company','City','Date Posted','Skills'].
    """
    # Chrome options
    options = ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--start-maximized")

    # start driver
    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        raise RuntimeError(f"Could not start Chrome WebDriver: {e}")

    base_url = "https://in.jooble.org/SearchResult?rgns=india&p={}"
    rows = []

    try:
        for page in range(1, max(1, int(pages)) + 1):
            url = base_url.format(page)
            if debug:
                print(f"[scraper] opening: {url}", flush=True)
            driver.get(url)

            # wait a bit for JS to render job cards
            try:
                WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-test-name="_jobCard"], li[data-test-name="jobCard"]'))
                )
            except Exception:
                # If wait times out, we still attempt parsing the page content
                if debug:
                    print(f"[scraper] wait timed out (page {page}) — continuing to parse current HTML", flush=True)

            time.sleep(1.0 + float(delay))
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Primary selector based on your snippet:
            job_cards = soup.select('div[data-test-name="_jobCard"], li[data-test-name="jobCard"]')

            # fallback heuristics
            if not job_cards:
                job_cards = soup.find_all(["article", "li", "div"], class_=lambda c: c and re.search(r'(job|card|vacancy)', c, re.I))

            if debug:
                print(f"[scraper] page {page} — found {len(job_cards)} candidate cards", flush=True)

            for job in job_cards:
                try:
                    # Title: anchor with class job_card_link OR data-test-name jobTitle OR heading fallback
                    title_el = (job.select_one("a.job_card_link") or
                                job.select_one("a[data-test-name='jobTitle']") or
                                job.select_one("h2 a") or
                                job.select_one("h3 a") or
                                job.find(["h1", "h2", "h3"]))
                    title = title_el.get_text(strip=True) if title_el else "N/A"

                    # Company: element with data-test-name _companyName
                    comp_el = job.select_one("[data-test-name='_companyName']") or job.select_one("p[data-test-name='_companyName']")
                    company = comp_el.get_text(strip=True) if comp_el else "N/A"

                    # Location and Date: look for '.caption' elements (snippet shows caption for location/date)
                    city = "N/A"
                    date_posted = "N/A"
                    # find all caption divs under job
                    caption_els = job.select(".caption")
                    if caption_els:
                        # heuristics: first caption is likely location, later one might be date (has 'ago')
                        for c in caption_els:
                            txt = c.get_text(strip=True)
                            if not txt:
                                continue
                            if re.search(r'\b(ago|day|days|hour|hours|week|weeks|month|months)\b', txt, re.I):
                                date_posted = txt
                            else:
                                # if city not set and it looks like a place (no digits, <=6 words)
                                if city == "N/A" and 1 < len(txt) <= 60 and not any(ch.isdigit() for ch in txt) and len(txt.split()) <= 6:
                                    city = txt
                    # fallback: search for elements with known classes that contained caption in snippet
                    if city == "N/A":
                        maybe = job.find(lambda tag: tag.name in ["div", "span"] and tag.get("class") and any("caption" in c for c in tag.get("class")))
                        if maybe:
                            city = maybe.get_text(strip=True)
                    if date_posted == "N/A":
                        maybe_date = job.find(lambda tag: tag.name in ["div", "span"] and tag.get_text(strip=True) and re.search(r'\b(ago|day|days|hour|hours|week|weeks|month|months)\b', tag.get_text(strip=True), re.I))
                        if maybe_date:
                            date_posted = maybe_date.get_text(strip=True)

                    # Tags/skills: jobTag elements in your snippet
                    tag_els = job.select("div[data-test-name='_jobTag'], [data-test-name='_jobTag']")
                    tokens = []
                    for t in tag_els:
                        txt = t.get_text(" ", strip=True)
                        if txt and not re.search(r'\b(report|save|apply)\b', txt, re.I):
                            tokens.append(txt)
                    # dedupe and join
                    skills = ", ".join(list(dict.fromkeys(tokens))) if tokens else "N/A"

                    # Experience attempt: search card text for 'x years' pattern
                    card_text = job.get_text(separator=" | ", strip=True)
                    exp_match = re.search(r'(\d+\+?\s*(?:years|yrs|year))', card_text, flags=re.I)
                    experience = exp_match.group(1) if exp_match else "N/A"

                    rows.append([title, experience, company, city, date_posted, skills])

                except Exception as e_job:
                    if debug:
                        print("[scraper] exception parsing job:", e_job, flush=True)
                        traceback.print_exc()

            # polite pause
            time.sleep(float(delay))
    finally:
        driver.quit()

    df = pd.DataFrame(rows, columns=['title', 'experience', 'company', 'City', 'Date Posted', 'Skills'])
    df.replace({None: "N/A", "": "N/A"}, inplace=True)
    return df

# ---------------------- LOAD EXCEL DATA (fallback) ----------------------
@st.cache_data
def load_data():
    try:
        return pd.read_excel("timesjobs_jobs.xlsx")
    except Exception:
        cols = ['title', 'experience', 'company', 'City', 'Date Posted', 'Skills']
        return pd.DataFrame(columns=cols)

# ---------------------- App start ----------------------
st.set_page_config(page_title="Job Market Dashboard", page_icon="📊", layout="wide")
st.title("📊 Job Market Analysis Dashboard - Final (Komalpreet Kaur)")

# Load Excel dataset (if present)
excel_df = load_data()
current_df = excel_df.copy()

# ---------------------- Sidebar: header + dataset selection ----------------------
st.sidebar.markdown(
    """
    <div style="text-align:center;padding:6px 0 0 0">
        <h3 style='margin:0;color:#0b63b8;'>📊 Job Analysis</h3>
        <p style='margin:0;font-size:12px;color:gray;'>Interactive Dashboard — India</p>
        <hr style='margin-top:8px'/>
    </div>
    """, unsafe_allow_html=True
)

dataset_choice = "Excel Dataset"
if "live_data" in st.session_state and st.session_state.live_data is not None:
    dataset_choice = st.sidebar.radio("Choose dataset to analyze:", ["Excel Dataset", "Live Scraped Data"], index=0)
else:
    st.sidebar.info("ℹ️ Live Scraped Data not available yet. Using Excel Dataset.")

if dataset_choice == "Live Scraped Data" and "live_data" in st.session_state and st.session_state.live_data is not None:
    current_df = st.session_state.live_data.copy()
    data_source = "Live Scraped Data"
else:
    current_df = excel_df.copy()
    data_source = "Excel Dataset"

st.sidebar.markdown("---")

# ---------------------- Global Filters (Company & Skill) ----------------------
st.sidebar.subheader("Global Filters")
all_companies = sorted([c for c in current_df['company'].dropna().unique()]) if 'company' in current_df.columns else []
company_choice = st.sidebar.selectbox("Filter by Company (Global)", ["All"] + all_companies)

def extract_skill_set(df):
    if 'Skills' not in df.columns:
        return []
    skills_raw = df['Skills'].dropna().astype(str)
    tokens = []
    for s in skills_raw:
        parts = re.split(r'[;,/|]', s)
        for p in parts:
            p = p.strip().lower()
            if p and len(p) <= 60 and not re.search(r'\b(ago|day|hour|work|office|hybrid|report|apply)\b', p):
                tokens.append(p)
    return sorted(set(tokens))

all_skills = extract_skill_set(current_df) if 'Skills' in current_df.columns else []
skill_choice = st.sidebar.selectbox("Filter by Skill (Global)", ["All"] + [s.title() for s in all_skills])

# Apply global filters to a view copy
df_view = current_df.copy()
if company_choice != "All" and 'company' in df_view.columns:
    df_view = df_view[df_view['company'] == company_choice]

if skill_choice != "All" and 'Skills' in df_view.columns:
    skill_lower = skill_choice.lower()
    df_view = df_view[df_view['Skills'].fillna("").str.lower().str.contains(re.escape(skill_lower))]

# Status banner
st.markdown(f"### 📊 Currently analyzing: **{data_source}**  —  Rows: **{len(df_view)}**")

# ---------------------- Top Summary Metrics ----------------------
st.markdown("#### 🔍 Quick Insights")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Jobs", len(df_view))
with c2:
    st.metric("Unique Cities", int(df_view['City'].nunique()) if 'City' in df_view.columns else 0)
with c3:
    try:
        top_company = df_view['company'].value_counts().idxmax()
    except Exception:
        top_company = "N/A"
    # prevent overly long company display by truncating in UI metric
    disp_company = top_company if len(str(top_company)) <= 30 else str(top_company)[:27] + "..."
    st.metric("Top Hiring Company", disp_company)
with c4:
    top_skill = "N/A"
    try:
        skills_ser = pd.Series([s.strip().lower() for s in ",".join(df_view['Skills'].dropna().astype(str)).split(",") if s.strip()])
        if not skills_ser.empty:
            top_skill = skills_ser.value_counts().idxmax().title()
    except Exception:
        top_skill = "N/A"
    st.metric("Most Demanded Skill", top_skill)

st.markdown("---")

# ---------------------- SIDEBAR MODE NAV ----------------------
mode = st.sidebar.radio("Choose Analysis Mode", ["Home", "City-wise", "All Cities (India)", "Compare Two Cities", "Job Map", "Live Scraping"])

# Helper for boxplot stats
def calc_box_stats(values):
    if not values:
        return [0.0,0.0,0.0,0.0,0.0]
    values = [float(v) for v in values if pd.notna(v)]
    if not values:
        return [0.0,0.0,0.0,0.0,0.0]
    values = sorted(values)
    q1 = float(np.percentile(values,25))
    q2 = float(np.percentile(values,50))
    q3 = float(np.percentile(values,75))
    return [float(min(values)), q1, q2, q3, float(max(values))]

# -------------------------------HOME-PAGE--------------------------------
if mode == "Home":
    st.subheader("Welcome to the Job Market Analysis Dashboard 👩‍💻📊")
    st.markdown(
        """
        This dashboard analyses job postings across India using a pre-scraped Excel dataset and an optional live Jooble scraper.
        Use the sidebar to switch modes and the global filters to focus on companies or skills.
        \n\n**Project:** Job Market Analysis in India\n**Submitted by:** Komalpreet Kaur\n**Submitted to:** Mrs. Bindu
        """
    )
    try:
        st.image("career_growth.jpg")
    except:
        st.write("Put career_growth.jpg in the app folder to show banner.")

# ------------------------------ CITY-WISE ------------------------------
elif mode == "City-wise":
    st.sidebar.header("City Filter Options")
    cities = sorted(df_view['City'].dropna().unique()) if 'City' in df_view.columns else ["N/A"]
    selected_city = st.sidebar.selectbox("Select a City", cities)
    exp_levels = sorted(df_view['experience'].dropna().unique()) if 'experience' in df_view.columns else ["N/A"]
    selected_exp = st.sidebar.selectbox("Select Experience Level", exp_levels)

    viz_options = [
        "Top Hiring Companies",
        "Experience Level Distribution",
        "Top Job Roles",
        "Job Role Treemap",
        "Experience vs Skills",
        "Skills Word Cloud",
        "Job Title Distribution",
        "Average Experience per Job Role",
        "Job Role vs Experience",
        "Skills Distribution"
    ]
    selected_viz = st.sidebar.selectbox("Select City Visualization", viz_options)
    city_data = df_view[df_view['City'] == selected_city]
    st.subheader(f" Job Data for {selected_city} — {len(city_data)} Jobs Found")
    st.dataframe(city_data.reset_index(drop=True), use_container_width=True)

    # --- City visualizations (same as previous app) ---
    if selected_viz == "Top Hiring Companies":
        company_count = city_data['company'].value_counts().head(10)
        option = {
            "xAxis": {"type": "category", "data": list(map(str, company_count.index.tolist()))},
            "yAxis": {"type": "value"},
            "series": [{"data": [int(v) for v in company_count.values.tolist()], "type": "bar", "colorBy": "data"}],
            "tooltip": {"show": True}
        }
        st_echarts(option, height="400px")

    elif selected_viz == "Experience Level Distribution":
        exp_counts = city_data['experience'].value_counts().head(10)  # Top 10
        option = {
            "tooltip": {"trigger": 'item', "formatter": '{b}: {c} Jobs'},
            "series": [{
                "name": "Experience", "type": "pie", "radius": ["40%", "70%"],
                "data": [{"value": int(v), "name": str(k)} for k, v in exp_counts.items()]
            }]
        }
        st_echarts(option, height="400px")

    elif selected_viz == "Top Job Roles":
        top_roles = city_data['title'].value_counts().head(10)
        option = {
            "xAxis": {"type": "category", "data": list(map(str, top_roles.index.tolist()))},
            "yAxis": {"type": "value"},
            "series": [{"data": [int(v) for v in top_roles.values.tolist()], "type": "bar", "colorBy": "data"}],
            "tooltip": {"show": True}
        }
        st_echarts(option, height="400px")

    elif selected_viz == "Job Role Treemap":
        role_counts = city_data['title'].value_counts().head(20)
        option = {
            "series": [{
                "type": "treemap",
                "data": [{"name": str(k), "value": int(v)} for k, v in role_counts.items()],
                "label": {"show": True}
            }]
        }
        st_echarts(option, height="500px")

    elif selected_viz == "Experience vs Skills":
        exp_data = city_data[city_data['experience'] == selected_exp]

        if exp_data.empty or exp_data['Skills'].dropna().empty:
            st.warning(f"No skills found for experience level **{selected_exp}** in {selected_city}.")
        else:
            all_skills = ' '.join(exp_data['Skills'].dropna())
            skill_list = re.split(r'[,|/;]', all_skills.lower())
            skill_list = [s.strip() for s in skill_list if s.strip()]

            if not skill_list:  # no skills after cleaning
                st.warning(f"No skills data available for {selected_exp} in {selected_city}.")
            else:
                skill_freq = pd.Series(skill_list).value_counts().head(15)
                option = {
                    "title": {
                        "text": f"Top Skills for {selected_exp} in {selected_city}",
                        "left": "center",
                        "textStyle": {"fontSize": 16, "fontWeight": "bold"}
                    },
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {
                        "type": "category",
                        "data": list(map(str, skill_freq.index.tolist())),
                        "axisLabel": {"rotate": 30, "fontSize": 12}
                    },
                    "yAxis": {"type": "value", "name": "Job Count"},
                    "series": [{
                        "data": [int(v) for v in skill_freq.values.tolist()],
                        "type": "bar",
                        "label": {"show": True, "position": "top"}
                    }]
                }
                st_echarts(option, height="500px")

    elif selected_viz == "Skills Word Cloud":
        all_skills = ' '.join(city_data['Skills'].dropna())
        skill_list = re.split(r'[,|/;]', all_skills.lower())
        skill_freq = pd.Series([s.strip() for s in skill_list if s.strip()]).value_counts()
        if not skill_freq.empty:
            freq_dict = {str(k): int(v) for k, v in skill_freq.items()}
            wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(freq_dict)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    elif selected_viz == "Job Title Distribution":
        title_trend = city_data['title'].value_counts().head(15)
        option = {
            "xAxis": {"type": "category", "data": list(map(str, title_trend.index.tolist()))},
            "yAxis": {"type": "value"},
            "series": [{"data": [int(v) for v in title_trend.values.tolist()], "type": "line", "smooth": True, "symbol": "circle"}],
            "tooltip": {"show": True}
        }
        st_echarts(option, height="400px")

    elif selected_viz == "Average Experience per Job Role":
        exp_df = city_data.copy()
        exp_df['exp_years'] = exp_df['experience'].str.extract(r'(\d+)').astype(float)
        avg_exp = exp_df.groupby('title')['exp_years'].mean().dropna().sort_values(ascending=False).head(10)
        option = {
            "xAxis": {"type": "category", "data": list(map(str, avg_exp.index.tolist()))},
            "yAxis": {"type": "value"},
            "series": [{"data": [float(v) for v in avg_exp.values.tolist()], "type": "bar", "colorBy": "data"}],
            "tooltip": {"show": True}
        }
        st_echarts(option, height="400px")

    elif selected_viz == "Job Role vs Experience":
        exp_df = city_data.copy()
        exp_df['exp_years'] = exp_df['experience'].str.extract(r'(\d+)').astype(float)
        role_exp = exp_df[['title', 'exp_years']].dropna()
        top_roles = role_exp['title'].value_counts().head(10).index
        series_data = [calc_box_stats(role_exp[role_exp['title'] == role]['exp_years'].tolist()) for role in top_roles]
        option = {
            "tooltip": {"show": True},
            "xAxis": {"type": "category", "data": list(map(str, top_roles.tolist()))},
            "yAxis": {"type": "value"},
            "series": [{"type": "boxplot", "data": series_data}]
        }
        st_echarts(option, height="400px")

    elif selected_viz == "Skills Distribution":
        all_skills = ' '.join(city_data['Skills'].dropna())
        skill_list = re.split(r'[,|/;]', all_skills.lower())
        skill_freq = pd.Series([s.strip() for s in skill_list if s.strip()]).value_counts().head(10)
        option = {
            "tooltip": {"trigger": "item", "formatter": "{b}: {c}"},
            "series": [{
                "name": "Skills", "type": "pie", "radius": ["40%", "70%"],
                "data": [{"name": str(k), "value": int(v)} for k, v in skill_freq.items()]
            }]
        }
        st_echarts(option, height="400px")

# ------------------------------ ALL INDIA ------------------------------
elif mode == "All Cities (India)":
    st.subheader(" India-Wide Job Market Overview")
    st.sidebar.header("All Cities (India) View")
    india_viz = st.sidebar.selectbox("Select India Visualization", [
        "Top Hiring Companies",
        "Most Demanding Skills",
        "Experience Distribution",
        "Top Job Titles",
        "India-wide Skills",
        "Experience per Job Role"
    ])

    if india_viz == "Top Hiring Companies":
        top_companies = df_view['company'].value_counts().head(10)
        option = {
            "xAxis": {"type": "category", "data": list(map(str, top_companies.index.tolist()))},
            "yAxis": {"type": "value"},
            "series": [{"data": [int(v) for v in top_companies.values.tolist()], "type": "bar", "colorBy": "data"}],
            "tooltip": {"show": True}
        }
        st_echarts(option, height="400px")

    elif india_viz == "Most Demanding Skills":
        all_skills = ' '.join(df_view['Skills'].dropna())
        skill_list = re.split(r'[,|/;]', all_skills.lower())
        skill_freq = pd.Series([s.strip() for s in skill_list if s.strip()]).value_counts().head(20)
        option = {
            "xAxis": {"type": "category", "data": list(map(str, skill_freq.index.tolist()))},
            "yAxis": {"type": "value"},
            "series": [{"data": [int(v) for v in skill_freq.values.tolist()], "type": "bar", "colorBy": "data"}],
            "tooltip": {"show": True}
        }
        st_echarts(option, height="400px")

    elif india_viz == "Experience Distribution":
        exp_counts = df_view['experience'].value_counts().head(10)  # Top 10
        option = {
            "tooltip": {"trigger": 'item', "formatter": '{b}: {c} Jobs'},
            "series": [{
                "name": "Experience", "type": "pie", "radius": ["40%", "70%"],
                "data": [{"value": int(v), "name": str(k)} for k, v in exp_counts.items()]
            }]
        }
        st_echarts(option, height="400px")

    elif india_viz == "Top Job Titles":
        top_titles = df_view['title'].value_counts().head(10)
        option = {
            "xAxis": {"type": "category", "data": list(map(str, top_titles.index.tolist()))},
            "yAxis": {"type": "value"},
            "series": [{"data": [int(v) for v in top_titles.values.tolist()], "type": "bar", "colorBy": "data"}],
            "tooltip": {"show": True}
        }
        st_echarts(option, height="400px")

    elif india_viz == "India-wide Skills":
        all_skills = ' '.join(df_view['Skills'].dropna())
        skill_list = re.split(r'[,|/;]', all_skills.lower())
        skill_freq = pd.Series([s.strip() for s in skill_list if s.strip()]).value_counts().head(10)
        option = {
            "tooltip": {"trigger": "item", "formatter": "{b}: {c}"},
            "series": [{
                "name": "Skills", "type": "pie", "radius": ["40%", "70%"],
                "data": [{"name": str(k), "value": int(v)} for k, v in skill_freq.items()]
            }]
        }
        st_echarts(option, height="400px")

    elif india_viz == "Experience per Job Role":
        exp_df = df_view.copy()
        exp_df['exp_years'] = exp_df['experience'].str.extract(r'(\d+)').astype(float)
        role_exp = exp_df[['title', 'exp_years']].dropna()
        top_roles = role_exp['title'].value_counts().head(10).index
        series_data = [calc_box_stats(role_exp[role_exp['title'] == role]['exp_years'].tolist()) for role in top_roles]
        option = {
            "tooltip": {"show": True},
            "xAxis": {"type": "category", "data": list(map(str, top_roles.tolist()))},
            "yAxis": {"type": "value"},
            "series": [{"type": "boxplot", "data": series_data}]
        }
        st_echarts(option, height="400px")

# ------------------------------ COMPARE TWO CITIES ------------------------------
elif mode == "Compare Two Cities":
    st.subheader(" Compare Two Cities Side by Side")
    unique_cities = sorted(df_view['City'].dropna().unique()) if 'City' in df_view.columns else ["N/A"]
    if len(unique_cities) < 2:
        st.warning("Not enough cities in data to compare.")
        unique_cities = unique_cities + unique_cities
    city1 = st.sidebar.selectbox("Select City 1", unique_cities)
    city2 = st.sidebar.selectbox("Select City 2", unique_cities, index=1 if len(unique_cities)>1 else 0)

    col1, col2 = st.columns(2)
    for col, city in zip([col1, col2], [city1, city2]):
        with col:
            st.markdown(f"### {city}")
            city_df = df_view[df_view['City'] == city]

            # Job Role Distribution
            role_counts = city_df['title'].value_counts().head(5)
            option = {
                "xAxis": {"type": "category", "data": list(map(str, role_counts.index.tolist()))},
                "yAxis": {"type": "value"},
                "series": [{"data": [int(v) for v in role_counts.values.tolist()], "type": "bar", "colorBy": "data"}],
                "tooltip": {"show": True}
            }
            st_echarts(option, height="300px")

            # Job Role Treemap
            option = {
                "series": [{
                    "type": "treemap",
                    "data": [{"name": str(k), "value": int(v)} for k, v in role_counts.items()],
                    "label": {"show": True}
                }]
            }
            st_echarts(option, height="300px")

# ------------------------------ JOB MAP ------------------------------
elif mode == "Job Map":
    st.subheader(" Map of Most Popular Job Roles by City in India")
    if 'City' not in df_view.columns or df_view['City'].dropna().empty:
        st.warning("No city data available in the current dataset.")
    else:
        city_top_roles = df_view.groupby('City')['title'].agg(lambda x: x.value_counts().idxmax()).reset_index()
        city_counts = df_view['City'].value_counts().reset_index()
        city_counts.columns = ['City', 'Job Count']
        city_data = pd.merge(city_top_roles, city_counts, on='City')

        city_coordinates = {
            "Bangalore": [12.9716, 77.5946],
            "Hyderabad": [17.3850, 78.4867],
            "Mumbai": [19.0760, 72.8777],
            "Pune": [18.5204, 73.8567],
            "Chennai": [13.0827, 80.2707],
            "Noida": [28.5355, 77.3910],
            "Delhi": [28.6139, 77.2090],
            "Gurgaon": [28.4595, 77.0266],
            "Ahmedabad": [23.0225, 72.5714],
            "Kolkata": [22.5726, 88.3639],
            "Indore": [22.7196, 75.8577],
            "Jaipur": [26.9124, 75.7873],
            "Chandigarh": [30.7333, 76.7794],
            "Lucknow": [26.8467, 80.9462]
        }

        city_data['Latitude'] = city_data['City'].map(lambda x: city_coordinates.get(x, [None, None])[0])
        city_data['Longitude'] = city_data['City'].map(lambda x: city_coordinates.get(x, [None, None])[1])
        city_data.dropna(inplace=True)
        city_data['Job Count'] = city_data['Job Count'].astype(int)

        fig = px.scatter_mapbox(
            city_data,
            lat="Latitude",
            lon="Longitude",
            size="Job Count",
            color="City",
            hover_name="City",
            hover_data={"Job Count": True, "title": True},
            zoom=4,
            height=600,
            size_max=20,
            mapbox_style="carto-positron"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- LIVE SCRAPING ----------------------
elif mode == "Live Scraping":
    st.subheader("🔴 Live Job Scraping (Jooble India) — tuned (user's scraper)")
    st.markdown("This uses a Selenium Chrome browser to fetch job cards from Jooble India (JS-rendered).")
    pages = st.sidebar.slider("Number of Jooble pages to scrape", 1, 10, 3)
    delay = st.sidebar.slider("Delay between pages (seconds)", 0, 4, 1)
    run_headless = st.sidebar.checkbox("Run Chrome headlessly (background)", value=False)
    debug_mode = st.sidebar.checkbox("Show scraper debug logs in console", value=False)

    if st.sidebar.button("Start India Scraping (Jooble)"):
        with st.spinner("Scraping Jooble (India) — Chrome browser will open..."):
            try:
                scraped_df = scrape_jooble_tuned(pages=pages, delay=delay, headless=run_headless, timeout=12, debug=debug_mode)
            except Exception as e:
                st.error("Selenium/Chrome error: " + str(e))
                st.info("Make sure Chrome is installed and webdriver_manager can download the driver. If error persists, try restarting the app.")
                scraped_df = pd.DataFrame(columns=['title','experience','company','City','Date Posted','Skills'])

        if scraped_df.empty:
            st.warning("⚠️ Scraper returned no jobs. If this happens: try increasing `pages` or `delay`, enable `debug` log, or paste a fresh job card outerHTML here so I can re-tune selectors.")
        else:
            st.success(f"✅ Scraped {len(scraped_df)} jobs from Jooble (India).")
            st.dataframe(scraped_df, use_container_width=True)
            st.session_state.live_data = scraped_df.copy()
            csv = scraped_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "jooble_india_scraped.csv", "text/csv")
            scraped_df.to_excel("jooble_india_scraped.xlsx", index=False)
            with open("jooble_india_scraped.xlsx", "rb") as f:
                st.download_button("📥 Download Excel", f, "jooble_india_scraped.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------- End ----------------------
