import streamlit as st
import json
import requests
import pandas as pd
from mistralai import Mistral
from io import StringIO

st.set_page_config(page_title="LinkedIn Lead Generator", layout="wide")

st.title("🔎 AI LinkedIn Lead Generator")

st.markdown("Generate LinkedIn leads using Mistral AI + Apify")

# ==============================
# SIDEBAR - API KEYS
# ==============================

st.sidebar.header("🔐 API Configuration")

mistral_api_key = st.sidebar.text_input(
    "Mistral API Key",
    type="password"
)

apify_api_token = st.sidebar.text_input(
    "Apify API Token",
    type="password"
)

model_choice = st.sidebar.selectbox(
    "Mistral Model",
    ["open-mistral-nemo"]
)

# ==============================
# USER INPUT
# ==============================

user_input = st.text_area(
    "Enter your LinkedIn search request",
    placeholder="Example: Give me 50 SaaS founders in UK"
)

run_button = st.button("🚀 Generate Leads")

# ==============================
# FUNCTIONS
# ==============================

def generate_payload(user_input, api_key, model):
    client = Mistral(api_key=api_key)

    system_prompt = """
You are an API payload generation agent.

Convert user input into JSON formatted for the following API payload:

{
  "search": "string",
  "location": "string",
  "maxResults": number,
  "jobTitles": ["string"],
  "industry": "string"
}

Rules:
- Convert shorthand locations to full names (UK → United Kingdom, US → United States).
- If quantity is missing → default maxResults to 25.
- Infer job titles when not explicit (Founder role implies: Founder, Co-Founder, CEO).
- Output ONLY valid JSON.
"""

    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    return json.loads(content)


def run_linkedin_scraper(payload, apify_token):
    url = f"https://api.apify.com/v2/acts/harvestapi~linkedin-profile-search/run-sync-get-dataset-items?token={apify_token}"

    body = {
        "currentJobTitles": payload["jobTitles"],
        "maxItems": payload["maxResults"],
        "profileScraperMode": "Full + email search",
        "recentlyChangedJobs": False,
        "searchQuery": payload["search"],
        "startPage": 1
    }

    response = requests.post(url, json=body)
    response.raise_for_status()

    return response.json()


def convert_to_dataframe(data):
    rows = []

    for profile in data:
        rows.append({
            "Profile URL": profile.get("linkedinUrl"),
            "First Name": profile.get("firstName"),
            "Last Name": profile.get("lastName"),
            "Email": (profile.get("emails") or [{}])[0].get("email"),
            "Headline": profile.get("headline"),
            "Company Website": (profile.get("companyWebsites") or [{}])[0].get("url"),
            "Location": (profile.get("location") or {}).get("linkedinText"),
            "Followers": profile.get("followerCount"),
            "Connections Count": profile.get("connectionsCount"),
        })

    return pd.DataFrame(rows)


# ==============================
# MAIN EXECUTION
# ==============================

if run_button:

    if not mistral_api_key or not apify_api_token:
        st.error("Please provide both API keys in the sidebar.")
        st.stop()

    if not user_input.strip():
        st.error("Please enter a search request.")
        st.stop()

    try:
        with st.spinner("🧠 Generating structured search using Mistral..."):
            payload = generate_payload(user_input, mistral_api_key, model_choice)

        st.success("Structured payload generated")
        st.json(payload)

        with st.spinner("🔎 Running LinkedIn scraper (this may take time)..."):
            results = run_linkedin_scraper(payload, apify_api_token)

        st.success(f"Retrieved {len(results)} profiles")

        df = convert_to_dataframe(results)

        st.subheader("📊 Results")
        st.dataframe(df, use_container_width=True)

        # Convert dataframe to CSV for download
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name="linkedin_leads.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")