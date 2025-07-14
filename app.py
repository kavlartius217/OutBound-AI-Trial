# streamlit_app.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import requests
import json
import datetime
import uuid
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Neo4j Credentials
os.environ['NEO4J_URI'] = st.secrets["neo4j"]["uri"]
os.environ['NEO4J_USERNAME'] = st.secrets["neo4j"]["username"]
os.environ['NEO4J_PASSWORD'] = st.secrets["neo4j"]["password"]

# OpenAI API Key
os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["api_key"]

# Exa API Key
os.environ['EXA_API_KEY'] = st.secrets["exa"]["api_key"]

# Apollo API Key
APOLLO_API_KEY = st.secrets["apollo"]["api_key"]

# AWS Credentials and S3 Configuration
AWS_ACCESS_KEY_ID = st.secrets["aws"]["access_key_id"]
AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["secret_access_key"]
AWS_REGION = st.secrets["aws"]["region"]
S3_BUCKET_NAME = st.secrets["aws"]["s3_bucket_name"]

# --- END: Credentials & Configuration from Streamlit Secrets ---


# --- START: Import Heavy Libraries ---
# These are imported after the initial credential setup.
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from exa_py import Exa
from pydantic import BaseModel
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_neo4j import Neo4jChatMessageHistory
# --- END: Import Heavy Libraries ---


# --- START: Session State Initialization ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'page_num' not in st.session_state:
    st.session_state.page_num = 1
if 'url_template' not in st.session_state:
    st.session_state.url_template = ""
if 'total_pages' not in st.session_state:
    st.session_state.total_pages = 0
if 'apollo_bot_output' not in st.session_state:
    st.session_state.apollo_bot_output = ""
if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = ""
# No longer needed: if 'aws_configured' not in st.session_state: st.session_state.aws_configured = False

# --- START: Agent and Tool Definitions (Cached for Performance) ---

@st.cache_resource
def get_llms():
    """Initializes and returns the LLMs to be used by the agents."""
    llm_openai_url = LLM(model="openai/gpt-4.1-2025-04-14", temperature=0)
    llm_openai_leads = LLM(model='openai/gpt-4.1-mini-2025-04-14', temperature=0.1)
    llm_openai_apollo = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
    return llm_openai_url, llm_openai_leads, llm_openai_apollo

llm_openai_url, llm_openai_leads, llm_openai_apollo = get_llms()

class Url(BaseModel):
    url: str

@st.cache_resource
def get_url_generator_crew():
    """Defines and returns the Crew for generating Apollo URLs."""
    apollo_url_agent = Agent(
        role="Apollo URL Generator",
        goal="Convert natural language queries into properly formatted Apollo People Search API URLs",
        backstory="""Expert at translating natural language search requests into Apollo People Search API URLs.
        Specializes in parsing queries, mapping criteria to API parameters, and generating properly formatted URLs
        with correct parameter names and values.""",
        llm=llm_openai_url
    )

    apollo_url_task = Task(
        description="""
        Generate Apollo People Search API URL from natural language query: {query}

        CORE PROCESS:
        1. Parse query to identify search criteria
        2. Map criteria to Apollo API parameters (listed below)
        3. Construct properly formatted URL with base endpoint and parameters
        4. Only include explicitly mentioned or clearly implied parameters
        5. Always add pagination: page=PLACEHOLDER_FOR_PAGE_NUMBER&per_page=10

        APOLLO API PARAMETERS:
        • person_titles[]: Job titles ("marketing manager", "sales development representative")
        • person_seniorities[]: Seniority levels (owner, founder, c_suite, partner, vp, head, director, manager, senior, entry, intern)
        • person_locations[]: Personal locations (cities, states, countries)
        • organization_locations[]: Company headquarters locations
        • q_organization_domains_list[]: Company domains (without www. or @)
        • contact_email_status[]: Email statuses (verified, unverified, likely_to_engage, unavailable)
        • organization_ids[]: Apollo company IDs
        • organization_num_employees_ranges[]: Employee count ranges ("min,max" format)
        • revenue_range[min]/revenue_range[max]: Revenue ranges (numbers only)
        • currently_using_all_of_technology_uids[]: Technologies company uses (all specified)
        • currently_using_any_of_technology_uids[]: Technologies company uses (any specified)
        • currently_not_using_any_of_technology_uids[]: Exclude companies using these technologies
        • q_organization_job_titles[]: Job titles in active job postings
        • organization_job_locations[]: Job posting locations
        • organization_num_jobs_range[min/max]: Number of active job postings range
        • organization_job_posted_at_range[min/max]: Job posting date range (YYYY-MM-DD)
        • include_similar_titles: Boolean for similar job title matching
        • q_keywords: General keyword search

        FORMATTING RULES:
        • Base URL: https://api.apollo.io/v1/mixed_people/search
        • Array format: parameter[]=value1¶meter[]=value2
        • Employee ranges: "min,max" (e.g., "1,10", "250,500")
        • Technology filters: replace spaces/periods with underscores
        • Date format: YYYY-MM-DD
        • URL encode special characters
        • Final format: "https://api.apollo.io/v1/mixed_people/search?[parameters]&page=PLACEHOLDER_FOR_PAGE_NUMBER&per_page=10"

        CRITICAL: Only include parameters explicitly requested or clearly implied in the query.
        """,
        expected_output="Properly formatted Apollo People Search API URL with appropriate query parameters and PLACEHOLDER_FOR_PAGE_NUMBER for pagination",
        agent=apollo_url_agent,
        output_pydantic=Url
    )

    return Crew(agents=[apollo_url_agent], tasks=[apollo_url_task])

def apollo_api_func(page_num: int) -> str:
    """Queries the Apollo API for a specific page number using the URL template from session_state."""
    if not st.session_state.get('url_template'):
        return "Error: Apollo URL template not found in session state. Please generate a URL first."

    url_template = st.session_state.url_template
    # The template has "PLACEHOLDER_FOR_PAGE_NUMBER", which we replace.
    url = url_template.replace("PLACEHOLDER_FOR_PAGE_NUMBER", str(page_num))

    headers = {
        "accept": "application/json",
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
        "x-api-key": APOLLO_API_KEY
    }
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error calling Apollo API: {e}"

apollo_tool = Tool.from_function(
    name="apollo_tool",
    func=apollo_api_func,
    description="Use this tool to extract the results from the apollo api for a given page number."
)

def get_apollo_bot_agent_executor(session_id: str, total_pages: int):
    """Creates the Apollo Bot agent executor with a specific history session."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You are a helpful assistant that finds people at companies. "
            "You have access to apollo_tool which returns 10 people per page. "
            f"There are {total_pages if total_pages > 0 else 'an unknown number of'} total pages available in this specific search."
            "\n\nCHAT HISTORY:"
            "\n{history}"
            "\n\nPAGINATION RULES:"
            "\n1. Check chat history for previous apollo_tool calls and their page numbers"
            "\n2. If no previous calls exist, start with page 1"
            "\n3. Otherwise, use the next page number after the last fetched page"
            "\n4. If last page was 500, start over at page 1"
            "\n\nOUTPUT REQUIREMENTS:"
            "\n- Always mention the page number in your response"
            "\n- Show pagination context (e.g., 'Page 15 of 500')"
            "\n- Display total people shown so far"
            "\n- Include LinkedIn and company website links clearly:"
            "\n    * Use one line per link."
            "\n    * Label them like: 'LinkedIn: [Profile](URL)' and 'Company Website: [Website](URL)'."
            "\n    * Never put multiple links on the same line."
            "\n    * Ensure links are valid and formatted as clickable Markdown hyperlinks."
            "\n\nINSTRUCTIONS:"
            "\n- Never ask user for page numbers"
            "\n- Use scratchpad to analyze history and determine next page"
            "\n- Call apollo_tool with integer page number"
            "\n\n{agent_scratchpad}"
        ),
        ("user", "{input}")
    ])
    history = Neo4jChatMessageHistory(session_id=session_id)
    agent = create_openai_functions_agent(llm=llm_openai_apollo, tools=[apollo_tool], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[apollo_tool], verbose=True)
    return agent_executor, history

@tool("Exa search and get contents")
def search_and_get_contents_tool(question: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""
    exa = Exa(os.getenv("EXA_API_KEY"))
    response = exa.search_and_contents(
        question,
        type="neural",
        num_results=3,
        highlights=True
    )
    return ''.join([
        f'<Title id={idx}>{eachResult.title}</Title>'
        f'<URL id={idx}>{eachResult.url}</URL>'
        f'<Highlight id={idx}>{"".join(eachResult.highlights)}</Highlight>'
        for (idx, eachResult) in enumerate(response.results)
    ])

@st.cache_resource
def get_analysis_crew():
    """Defines and returns the Crew for analyzing LinkedIn profiles and generating a report."""
    # --- UPDATED LinkedIn Analysis Agent ---
    LinkedIn_analysis_agent = Agent(
        role="LinkedIn Sales Intelligence Analyst",
        goal="Extract actionable sales insights from LinkedIn profiles by analyzing recent activity, pain points, and technology challenges to identify prospects most likely to need {solution_service}",
        backstory="""Expert sales intelligence researcher with 10+ years analyzing LinkedIn profiles for B2B sales. Excel at identifying purchase intent signals and business pain points through LinkedIn activity analysis.""",
        tools=[search_and_get_contents_tool],
        llm=llm_openai_leads,
        verbose=True,
        memory=True
    )

    # --- UPDATED LinkedIn Analysis Task ---
    LinkedIn_analysis_task = Task(
        description="""
        Analyze LinkedIn profiles {leads} to identify sales opportunities. Research each prospect's:

        1. Recent posts/activity (3 months) - challenges, goals, tech needs
        2. Job changes/promotions - new responsibilities, budget authority
        3. Company updates - growth, funding, acquisitions, strategic shifts
        4. Industry pain points expressed or engaged with
        5. Engagement with {solution_service} related content
        6. Professional network changes and speaking engagements
        7. Analyze ALL prospects provided - no omissions

        Target prospects who:
        - Posted about relevant challenges/tech needs
        - Changed roles (new budgets/solutions needed)
        - Work at growing/transforming companies
        - Engage with solution-domain content
        - Express frustration with current tools

        Input: LinkedIn profile URLs from Apollo
        Solution/Service: {solution_service}
        """,
        expected_output="""
        For each prospect provide:

        1. **Priority Score (1-10)** with justification:
           - Buying intent (30%) | Pain alignment (25%) | Recent activity (20%) | Company signals (15%) | Engagement (10%)

        2. **Pain Points** (3-5 specific challenges):
           - Quote evidence from posts/activities
           - Urgency level (Critical/High/Medium)
           - Business impact connection

        3. **Recent Activity** (90-day summary):
           - Post dates and content snippets
           - Engagement metrics
           - Technology mentions/frustrations

        4. **Solution Fit**:
           - Specific features addressing their needs
           - Competitive landscape awareness
           - Implementation complexity

        5. **Messaging Angles** (2-3 starters):
           - Reference specific recent posts
           - Lead with value, not features
           - Timing considerations

        6. **Buying Signals**:
           - Budget cycle timing
           - Tech refresh indicators
           - Organizational changes
           - Competitive pressures

        7. **Contact Strategy**:
           - Best approach method
           - Optimal timing
           - Risk assessment
           - Alternative contact methods

        8. **LinkedIn Profile URL**

        9. **Analyze ALL provided LinkedIn profiles**

        Format as structured JSON-like data for report processing.
        """,
        agent=LinkedIn_analysis_agent
    )

    report_generator_agent = Agent(
        role="Sales Report Strategist",
        goal="Transform LinkedIn intelligence into prioritized, actionable sales reports that drive immediate outreach results",
        backstory="""You are a sales operations strategist who has helped hundreds of B2B sales teams convert
        raw prospect data into revenue. You understand what busy sales reps need: clear priorities, proven messaging,
        and specific next steps. You create reports that sales teams actually use because they're practical,
        prioritized, and immediately actionable.""",
        llm=llm_openai_leads,
        tools=[search_and_get_contents_tool],
        verbose=True,
        memory=True
    )

    report_generator_task = Task(
        description="""
        Create a comprehensive sales report that turns LinkedIn analysis into immediate sales actions.

        Report Structure:
        1. Executive Dashboard - Key metrics and top opportunities
        2. Comprehensive Priority Prospects Table - All sales intelligence in one table
        3. Success metrics to track

        Include:
        - Clear ROI projections based on prospect quality
        - Risk assessment for each prospect
        - Recommended outreach cadence and channels
        - Success metrics to track
        """,
        expected_output="""
        A comprehensive markdown report with:

        ## Executive Summary
        - Total prospects analyzed
        - High-priority prospects (score 8+)
        - Estimated pipeline value
        - Key insights and trends

        ## Comprehensive Priority Prospects Table

        Use this exact table format with ALL information. Do NOT use HTML tags like <br> in the output. Use bullet points or short sentences for multi-item cells.

        | Rank | Prospect | Company | Score | Primary Pain Points | Recent Activity | Solution Fit | Message Template | Outreach Sequence | Next Action | Timeline | Follow-up Plan |
        |------|----------|---------|-------|-------------------|-----------------|--------------|------------------|-------------------|-------------|----------|----------------|
        | 1 | [Prospect Name](linkedin-url) | Company Name | X/10 | • Pain point 1. • Pain point 2. | Recent activity description (X days ago) | High/Medium/Low - Specific fit details | "Personalized message template referencing their specific situation." | Day 1: LinkedIn DM. Day 3: Follow-up. Day 7: Email. | Specific action with context | Within Xhrs/days | Week 1: Track response. Week 2: Nurture. Week 3: Re-engage. |

        Requirements for the comprehensive table:
        - Prospect names must be hyperlinked to LinkedIn profiles: [Name](linkedin-url)
        - Pain points should use bullet points for clarity.
        - Recent activity must include timing (X days/weeks ago)
        - Solution fit must specify High/Medium/Low with explanation
        - Message template must be personalized and reference specific prospect details
        - Outreach sequence must include multi-touch cadence with channels and timing
        - Next action must be specific and actionable
        - Timeline must be precise (Within 24hrs, Within 1 week, etc.)
        - Follow-up plan must include ongoing nurture strategy

        ## Success Metrics to Track
        - Response Rate: % of prospects engaging with initial outreach
        - Meeting Conversion: # of discovery calls booked
        - Pipeline Value: $ value of qualified opportunities created
        - Deal Velocity: Time from first contact to qualified opportunity
        - ROI Realization: % of prospects achieving projected outcomes

        Ready-to-execute sales intelligence report with everything in one comprehensive table.
        """,
        context=[LinkedIn_analysis_task],
        output_file="actionable_leads.md",
        agent=report_generator_agent
    )

    return Crew(
        agents=[LinkedIn_analysis_agent, report_generator_agent],
        tasks=[LinkedIn_analysis_task, report_generator_task]
    )

# --- END: Agent and Tool Definitions ---


# --- START: Streamlit UI ---
st.set_page_config(layout="wide", page_title="OutboundAI - Apollo Bot MARK III")

st.title("🚀 OutboundAI: Your AI-Powered Sales Intelligence Engine")
st.markdown("Convert natural language to an Apollo URL, fetch leads, and generate an actionable sales intelligence report.")

st.sidebar.title("Configuration")
st.sidebar.info("This app uses credentials stored in Streamlit's secrets manager. Ensure all API keys and AWS credentials are correctly configured in your app settings.")


col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.header("Step 1: Define Your Search")
        query = st.text_area(
            "Enter your search query in natural language:",
            "Find marketing directors and VPs at SaaS companies in California and Texas with 50-500 employees that have revenue between 5 million and 50 million dollars and are currently using HubSpot or Salesforce",
            height=150
        )

        if st.button("Generate Apollo URL", type="primary"):
            with st.spinner("Generating URL..."):
                url_crew = get_url_generator_crew()
                result = url_crew.kickoff({"query": query})
                # Reset state for the new URL
                st.session_state.url_template = result['url']
                st.session_state.session_id = str(uuid.uuid4()) # New URL = New History Session
                st.session_state.page_num = 1
                st.session_state.total_pages = 0
                st.session_state.apollo_bot_output = ""
                st.session_state.analysis_report = ""

        if st.session_state.url_template:
            st.success("URL Generated Successfully!")
            st.code(st.session_state.url_template, language="text")
            st.info(f"A new history session has been created: `{st.session_state.session_id}`. All subsequent fetches for this URL will use this session.")


with col2:
    with st.container(border=True):
        st.header("Step 2: Fetch & Analyze Leads")
        is_disabled = not st.session_state.url_template

        if is_disabled:
            st.warning("Please generate a URL first.")

        solution_service = st.text_input(
            "What solution/service are you selling?",
            "Microsoft Azure Cloud Migration Services"
        )

        if st.button("Fetch & Analyze Leads", disabled=is_disabled, type="primary"):
            # --- BOT EXECUTION ---
            with st.spinner(f"Apollo Bot fetching page {st.session_state.page_num}..."):
                apollo_bot_agent, history = get_apollo_bot_agent_executor(
                    st.session_state.session_id,
                    st.session_state.total_pages
                )

                # The bot needs an input to trigger the tool call with the right page number
                bot_input = f"find me companies using the tool on page {st.session_state.page_num}"

                response = apollo_bot_agent.invoke({
                    "input": bot_input,
                    "history": history.messages
                })

                history.add_user_message(bot_input)
                history.add_ai_message(response['output'])
                st.session_state.apollo_bot_output = response['output']

                # Try to parse total pages from the first API call
                if st.session_state.total_pages == 0:
                    try:
                        # The actual data is in the tool call's output, not the AI's summary
                        raw_json_str = apollo_api_func(st.session_state.page_num)
                        data = json.loads(raw_json_str)
                        st.session_state.total_pages = data.get("pagination", {}).get("total_pages", 0)
                    except (json.JSONDecodeError, KeyError) as e:
                        st.warning(f"Could not parse total pages from API response: {e}")

            # --- ANALYSIS EXECUTION ---
            with st.spinner("Agents are analyzing leads and generating the report..."):
                analysis_crew = get_analysis_crew()
                analysis_crew.kickoff({
                    "solution_service": solution_service,
                    "leads": st.session_state.apollo_bot_output
                })

                # Read the generated markdown file
                try:
                    with open("actionable_leads.md", "r") as f:
                        st.session_state.analysis_report = f.read()
                except FileNotFoundError:
                    st.session_state.analysis_report = "Error: Report file 'actionable_leads.md' not found."

            # --- S3 UPLOAD ---
            with st.spinner("Uploading report to S3..."):
                try:
                    s3_client = boto3.client(
                        's3',
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        region_name=AWS_REGION
                    )
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    filename = f"leads_report_{timestamp}.md"
                    s3_client.upload_file(
                        Filename="actionable_leads.md",
                        Bucket=S3_BUCKET_NAME,
                        Key=filename
                    )
                    st.success(f"Report successfully uploaded to S3 bucket '{S3_BUCKET_NAME}' as `{filename}`.")
                except (NoCredentialsError, PartialCredentialsError):
                    st.error("S3 Upload Failed: AWS credentials are not configured correctly in Streamlit Secrets.")
                except Exception as e:
                    st.error(f"S3 Upload Failed: {e}")

            # Increment page number for the next run
            st.session_state.page_num += 1

# --- Display Results ---
if st.session_state.url_template:
    st.divider()
    st.header("Results")

    if st.session_state.total_pages > 0:
        st.metric(label="Total Pages Found", value=st.session_state.total_pages)

    if st.session_state.apollo_bot_output:
        with st.expander(f"Apollo Bot Output (Page {st.session_state.page_num - 1})", expanded=False):
            st.markdown(st.session_state.apollo_bot_output)

    if st.session_state.analysis_report:
        with st.expander("Sales Intelligence Report", expanded=True):
            st.markdown(st.session_state.analysis_report)
