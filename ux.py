import regex
import streamlit as st
import json
import os
import time
from datetime import datetime
import random
import sys
import io
import re
import markdown


@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model to avoid reloading on every page refresh."""
    from retriever.Embbedingator import Embbedingator
    return Embbedingator()


class PolicyChatbot:
    def __init__(self):
        """
        Initialize the PolicyChat conversational assistant.
        """
        self.state = {
            'query': None,
            'analysis_option': None,
            'papers_folder': None,
            'local_papers': [],
            'current_stage': 'welcome'
        }
        self.conversation_history = []

    def generate_response(self, user_input=None):
        """
        Generate contextual responses based on the current conversation stage.
        """
        responses = {
            'welcome': [
                "Hello! I'm PolicyChat, ready to explore some policy insights?",
                "Hello! I'm PolicyChat — what policy question would you like to explore today?",
                "Hi — PolicyChat here. What's your policy question?"
            ],
            'query_clarification': [
                "Could you tell me about the specific policy situation you're interested in? Please give as much detail as possible.",
                "Please describe the policy question or context you want to explore. The more detail, the better.",
                "What exactly is the policy question you'd like to investigate? Please include relevant scope, geography, time horizon, and stakeholders."
            ],
            'analysis_option': [
                "I can help you analyze papers in three ways:\n\n1) Local papers only — you provide a folder path with your documents and I analyze them.\n2) External papers only — I search online (Semantic Scholar + Genie API) and analyze the results.\n3) Mixed — I analyze your local papers and also search online for complementary resources.\n\nWhich approach would you prefer?",
            ],
            'papers_folder': [
                "Please provide the full path to the folder containing your policy papers.",
                "I'll need the directory where your research papers are stored. Can you share the path?",
                "Which folder contains the policy documents you want to analyze?"
            ],
            'paper_selection': [
                "Great! Here are the papers I found in the local folder you provided. Please, select up to 5 for our analysis by typing the paper's reference numbers separated by comma.",
                "These are the documents in your folder. Which ones would you like to focus on? Please, select up to 5 for our analysis by typing the paper's reference numbers separated by comma",
                "I've discovered these papers in the provided folder. Please, select up to 5 for our analysis by typing the paper's reference numbers separated by comma"
            ],
            'complete': [
                "👍 Perfect! I'll now search for relevant papers online and analyze them for you. Make sure 'Enable external search' is checked in the sidebar.",
                "Great! I'll search online resources to help answer your policy question. Please ensure external search is enabled in the sidebar.",
                "Ready to go! I'll query online databases for relevant research. Check that external search is enabled in the sidebar settings."
            ]
        }

        # Add randomness to responses
        if self.state['current_stage'] in responses:
            return random.choice(responses[self.state['current_stage']])
        return "I'm not sure how to respond right now. Let's start over. Please, refresh this page to do so. "

    def process_user_input(self, user_input):
        """
        Process user input based on the current conversation stage.
        """
        if self.state['current_stage'] == 'welcome':
            self.state['current_stage'] = 'query_clarification'
            return self.generate_response()

        elif self.state['current_stage'] == 'query_clarification':
            self.state['query'] = user_input
            self.state['current_stage'] = 'analysis_option'
            return self.generate_response()

        elif self.state['current_stage'] == 'analysis_option':
            li = user_input.lower()
            if any(option in li for option in ['1', 'one', 'local', 'local papers', 'first']):
                self.state['analysis_option'] = '1'
                # local-only requires a papers folder
                self.state['current_stage'] = 'papers_folder'
                return self.generate_response()
            elif any(option in li for option in ['2', 'two', 'external', 'online', 'only']):
                self.state['analysis_option'] = '2'
                # external-only: skip mandatory local folder
                self.state['current_stage'] = 'complete'
                return self.generate_response()
            elif any(option in li for option in ['3', 'three', 'mixed', 'both']):
                self.state['analysis_option'] = '3'
                # mixed requires local folder as well
                self.state['current_stage'] = 'papers_folder'
                return self.generate_response()
            else:
                return "Please choose 1, 2, or 3. Let me show the options again:\n\n" + self.generate_response()

        elif self.state['current_stage'] == 'papers_folder':
            if os.path.exists(user_input):
                self.state['papers_folder'] = user_input
                files = [f for f in os.listdir(user_input) if f.endswith('.pdf') or f.endswith('.txt')]
                
                if not files:
                    return "No papers found in that folder. Please check the path."

                self.state['available_papers'] = files
                self.state['current_stage'] = 'paper_selection'
                return f"I found {len(files)} papers in the local folder provided. Here they are:\n\n" + \
                       "\n".join(f"{i+1}. {file}" for i, file in enumerate(files)) + \
                       "\n\nWhich papers would you like to analyze? (Please enter the reference numbers separated by commas)"

            else:
                return "That folder doesn't seem to exist. Please provide a valid folder path."

        elif self.state['current_stage'] == 'paper_selection':
            try:
                selected_indices = [int(x.strip())-1 for x in user_input.split(',')]
                selected_papers = [os.path.join(self.state['papers_folder'], 
                                                self.state['available_papers'][i]) 
                                   for i in selected_indices]
                
                self.state['local_papers'] = selected_papers
                self.state['current_stage'] = 'complete'
                
                # Save user inputs
                with open('user_inputs.json', 'w') as f:
                    json.dump({
                        'query': self.state['query'],
                        'analysis_option': self.state['analysis_option'],
                        'papers_folder': self.state['papers_folder'],
                        'local_papers': self.state['local_papers']
                    }, f, indent=4)
                
                return f"👍 Great! I've saved the {len(selected_papers)} papers you indicated for analysis and will now start the research. " 
            
            except (ValueError, IndexError):
                return "Invalid selection. Please enter valid paper numbers."

        return "I'm not sure what to do next. Let's start over."
    
    

def main():
    # OPTIMIZATION: Set page config FIRST (before any other Streamlit commands)
    st.set_page_config(page_title="PolicyChat", page_icon="🤖", layout="wide")
    
    # OPTIMIZATION: Show loading state immediately while session initializes
    if 'initialized' not in st.session_state:
        with st.spinner('🚀 Initializing PolicyChat...'):
            st.session_state.initialized = True
            st.session_state.chatbot = None
            st.session_state.messages = []
    
    def message_output(text):
        # Add the message to the Streamlit session state messages
        if 'messages' in st.session_state:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": text
            })
            # Optional: if you want to display the message immediately
            with st.chat_message("assistant"):
                st.write(text)

    st.title("PolicyChat: Your AI Copilot for Policy Research 🤖 📄 ⚖️")

    # Sidebar: external search and API key
    with st.sidebar:
        st.header("External Search")
        external_search = st.checkbox("Enable external search (Semantic Scholar)", value=True)
        max_results = st.number_input("Max external results", min_value=1, max_value=50, value=10)
        
        st.header("Policy Domain")
        topic = st.selectbox(
            "Select policy topic for evidence scoring",
            [
                "unemployment",
                "female_unemployment",
                "education",
                "poverty",
                "poverty_reduction"
            ],
            index=0,
            help="This helps PolicyChat score and rank papers based on topic-specific quality criteria."
        )
        
        st.header("API & Local Files")
        api_key_input = st.text_input("Semantic Scholar API key (optional)", type="password")
        optional_local_folder = st.text_input("Optional local papers folder (leave blank to skip)")
        st.markdown("---")
        st.caption("If you enable external search, the app will query Semantic Scholar and may download open-access PDFs for highly relevant papers.")

    # OPTIMIZATION: Initialize session state components only once (prevents reconnection issues)
    if 'chatbot' not in st.session_state or st.session_state.chatbot is None:
        st.session_state.chatbot = PolicyChatbot()
    
    # Initialize chat history only on first load
    if 'messages' not in st.session_state or len(st.session_state.messages) == 0:
        st.session_state.messages = []
        
        # Add initial assistant greeting
        try:
            initial_greeting = st.session_state.chatbot.generate_response()
            st.session_state.messages.append({
                "role": "assistant", 
                "content": initial_greeting
            })
        except Exception as e:
            # Fallback greeting if chatbot initialization fails
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hello! I'm PolicyChat, ready to explore some policy insights?"
            })

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What's your policy research goal?", key="policy_research_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process user input and get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(0.5)  # Simulate thinking
                response = st.session_state.chatbot.process_user_input(prompt)
                st.write(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Check if all required inputs are collected
        chatbot_state = st.session_state.chatbot.state
        option = chatbot_state.get('analysis_option')
        # external-only (2) doesn't require local folder; local (1) and mixed (3) do
        if option == '2':
            ready = bool(chatbot_state.get('query') and option)
        else:
            ready = bool(chatbot_state.get('query') and option and chatbot_state.get('papers_folder') and chatbot_state.get('local_papers'))

        if ready:
            # Prepare user inputs for Coordinator
            user_inputs = {
                'query': chatbot_state['query'],
                'option': chatbot_state['analysis_option'],  # Changed from 'analysis_option' to 'option'
                'topic': topic  # Add topic from sidebar
            }
            if chatbot_state.get('papers_folder'):
                user_inputs['papers_folder'] = chatbot_state['papers_folder']
            if chatbot_state.get('local_papers'):
                user_inputs['local_papers'] = chatbot_state['local_papers']
            # Include external search settings
            user_inputs['external_search'] = bool(external_search)
            user_inputs['max_results'] = int(max_results)
            if api_key_input:
                user_inputs['semantic_scholar_api_key'] = api_key_input
            # Optional local folder from sidebar overrides chat-provided folder
            if optional_local_folder and os.path.exists(optional_local_folder):
                user_inputs['optional_local_folder'] = optional_local_folder
            
            # Save to JSON for Coordinator
            with open('user_inputs.json', 'w') as f:
                json.dump(user_inputs, f, indent=4)
            
            try:
                # Call Coordinator (import only when needed to speed up initial load)
                from retriever.coordinator import Coordinator
                coordinator = Coordinator(message_output=message_output)
                results = coordinator.run_pipeline()
                
                # Display top papers with EQR scores
                if results and results.get("top_papers"):
                    st.subheader("📊 Top Evidence (Ranked by Quality)")
                    st.markdown("Papers are scored using Evidence Quality Rating (EQR) based on methodology, data quality, recency, citations, and topic relevance.")
                    
                    for i, paper in enumerate(results["top_papers"][:5], start=1):
                        tier_emoji = {"A": "🥇", "B": "🥈", "C": "🥉", "D": "📄"}.get(paper.get("tier", "D"), "📄")
                        with st.expander(f"{tier_emoji} {i}. {paper.get('title', 'Unknown Title')[:80]}... (Tier {paper.get('tier', 'D')} | EQR {paper.get('eqr', 0)})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Venue:** {paper.get('venue', 'Unknown')}")
                                st.markdown(f"**Year:** {paper.get('year', 'N/A')}")
                                st.markdown(f"**Citations:** {paper.get('citation_count', 0)}")
                            with col2:
                                st.markdown(f"**Relevance Score:** {paper.get('query_similarity', 0):.2f}")
                                st.markdown(f"**Rank Score:** {paper.get('rank_score', 0):.2f}")
                                if paper.get('url'):
                                    st.markdown(f"[View Paper]({paper['url']})")
                            
                            if paper.get("eqr_reasons"):
                                st.markdown("**✅ Quality Indicators:**")
                                for r in paper.get("eqr_reasons", []):
                                    st.markdown(f"- {r}")
                            
                            if paper.get("eqr_warnings"):
                                st.markdown("**⚠️ Limitations:**")
                                for w in paper.get("eqr_warnings", []):
                                    st.markdown(f"- {w}")
                
                # Find the memo from coordinator results (no CWD scan)
                memo_path = results.get("memo_path") if results else None
                
                if memo_path and os.path.exists(memo_path):
                    # Read and display Markdown file
                    with open(memo_path, 'r', encoding='utf-8') as file:
                        markdown_content = file.read()
                    
                    # Convert Markdown to HTML
                    html_content = markdown.markdown(markdown_content, extensions=['fenced_code', 'tables'])
                    
                    # Create an expander for the Markdown content
                    with st.expander("Policy Memo"):
                        st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.warning("No memo file found.")
                
            except Exception as e:
                st.error(f"Error in pipeline: {e}")
                # Log full traceback for debugging
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch any top-level errors and display them gracefully
        st.error(f"⚠️ Application Error: {e}")
        st.info("Try refreshing the page. If the issue persists, check Docker logs.")
        import traceback
        with st.expander("🔍 Debug Information"):
            st.code(traceback.format_exc())