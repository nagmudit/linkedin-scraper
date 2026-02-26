import streamlit as st
import json
import requests
import pandas as pd
from mistralai import Mistral
from apify_client import ApifyClient
from io import StringIO

st.set_page_config(page_title="AI Lead & Instagram Scraper", layout="wide")

st.title("🔎 AI Lead & Instagram Scraper")

st.markdown("Generate LinkedIn leads and scrape Instagram posts using Mistral AI + Apify")

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


def run_instagram_posts_scraper(profile_url, max_items, apify_token):
    client = ApifyClient(apify_token)

    run_input = {
        "startUrls": [profile_url],
        "maxItems": int(max_items),
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
        },
    }

    run = client.actor("parseforge/instagram-posts-scraper").call(run_input=run_input)

    return list(client.dataset(run["defaultDatasetId"]).iterate_items())


def convert_instagram_to_dataframe(data):
    rows = []

    for post in data:
        rows.append({
            "Post URL": post.get("postUrl") or post.get("url") or post.get("permalink"),
            "Caption": post.get("caption") or post.get("text"),
            "Likes": post.get("likesCount") or post.get("likes"),
            "Comments": post.get("commentsCount") or post.get("comments"),
            "Posted At": post.get("timestamp") or post.get("takenAt"),
            "Is Video": post.get("isVideo"),
            "Thumbnail": post.get("displayUrl") or post.get("thumbnailUrl"),
            "Username": post.get("ownerUsername") or post.get("username"),
        })

    return pd.DataFrame(rows)


# ==============================
# MAIN EXECUTION
# ==============================

linkedin_tab, instagram_tab = st.tabs(["LinkedIn Leads", "Instagram Posts"])

with linkedin_tab:
    st.subheader("LinkedIn Lead Generation")

    user_input = st.text_area(
        "Enter your LinkedIn search request",
        placeholder="Example: Give me 50 SaaS founders in UK"
    )

    run_button = st.button("🚀 Generate Leads")

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

with instagram_tab:
    st.subheader("Instagram Post Scraper")

    instagram_profile_url = st.text_input(
        "Instagram profile URL",
        placeholder="https://www.instagram.com/instagram/"
    )

    instagram_max_items = st.number_input(
        "Maximum posts to fetch",
        min_value=1,
        max_value=500,
        value=10,
        step=1
    )

    run_instagram_button = st.button("📸 Scrape Instagram Posts")

    if run_instagram_button:
        if not apify_api_token:
            st.error("Please provide your Apify API token in the sidebar.")
            st.stop()

        if not instagram_profile_url.strip():
            st.error("Please enter an Instagram profile URL.")
            st.stop()

        try:
            with st.spinner("🔎 Running Instagram posts scraper (this may take time)..."):
                instagram_results = run_instagram_posts_scraper(
                    instagram_profile_url.strip(),
                    instagram_max_items,
                    apify_api_token
                )

            st.success(f"Retrieved {len(instagram_results)} posts")

            instagram_df = convert_instagram_to_dataframe(instagram_results)

            st.subheader("📊 Instagram Posts")
            st.dataframe(instagram_df, use_container_width=True)

            instagram_csv_buffer = StringIO()
            instagram_df.to_csv(instagram_csv_buffer, index=False)
            instagram_csv_data = instagram_csv_buffer.getvalue()

            st.download_button(
                label="⬇️ Download Instagram CSV",
                data=instagram_csv_data,
                file_name="instagram_posts.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")