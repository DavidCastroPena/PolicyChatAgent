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
from retriever.coordinator import Coordinator



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
    def message_output(text):
        # Add the message to the Streamlit session state messages
        st.session_state.messages.append({
            "role": "assistant", 
            "content": text
        })
        # Optional: if you want to display the message immediately
        with st.chat_message("assistant"):
            st.write(text)

    st.set_page_config(page_title="PolicyChat", page_icon="🤖", layout="wide")
    st.title("PolicyChat: Your AI Copilot for Policy Research 🤖 📄 ⚖️")

    # Sidebar: external search and API key
    with st.sidebar:
        st.header("External Search")
        external_search = st.checkbox("Enable external search (Semantic Scholar)", value=True)
        max_results = st.number_input("Max external results", min_value=1, max_value=50, value=10)
        api_key_input = st.text_input("Semantic Scholar API key (optional)", type="password")
        optional_local_folder = st.text_input("Optional local papers folder (leave blank to skip)")
        st.markdown("---")
        st.caption("If you enable external search, the app will query Semantic Scholar and may download open-access PDFs for highly relevant papers.")

    # Initialize or retrieve session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PolicyChatbot()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
        # Add initial assistant greeting
        initial_greeting = st.session_state.chatbot.generate_response()
        st.session_state.messages.append({
            "role": "assistant", 
            "content": initial_greeting
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
                'option': chatbot_state['analysis_option']
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
                # Call Coordinator
                coordinator = Coordinator(message_output=message_output)
                coordinator.run_pipeline()
                
                # Find the most recent memo file
                directory_path = os.getcwd()
                memo_files = [f for f in os.listdir(directory_path) if f.startswith('memo_') and f.endswith('.md')]
                
                if memo_files:
                    # Sort files by modification time, most recent first
                    most_recent_memo = max(memo_files, key=lambda f: os.path.getmtime(os.path.join(directory_path, f)))
                    markdown_file_path = os.path.join(directory_path, most_recent_memo)
                    
                    # Read and display Markdown file
                    if os.path.exists(markdown_file_path):
                        with open(markdown_file_path, 'r') as file:
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

if __name__ == "__main__":
    main()