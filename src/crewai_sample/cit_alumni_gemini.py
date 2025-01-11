from dotenv import load_dotenv
import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource

# Load environment variables
load_dotenv()

# Validate API keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("MODEL")
metadata = {"source": "user_profile", "description": "Basic user information"}

# Initialize Excel knowledge source
excel_source = ExcelKnowledgeSource(
    file_path="cit_alumni_master.xlsx",  # Corrected: file_path expects a string
    metadata=metadata
)

# Initialize LLM
gemini_llm = LLM(
    model="gemini/gemini-1.5-pro-002",
    api_key=gemini_api_key,
    temperature=0,
)

# Initialize Agent
cit_alumni_agent = Agent(
    role="CIT Alumni Information Assistant",
    goal="Provide accurate information about CIT alumni.",
    backstory="You specialize in providing alumni information including payments, membership details, and personal data.",
    llm=gemini_llm
)

# Streamlit Chatbot UI
st.title("🎓 CIT Alumni Chatbot")
st.markdown("Ask me anything about CIT Alumni!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I assist you with CIT Alumni information today?"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if user_input := st.chat_input("Type your question here..."):
    # Append user input to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Create a task
    task = Task(
        description=f"Answer this CIT alumni query: {user_input}",
        expected_output="A detailed answer based on the CIT alumni data.",
        agent=cit_alumni_agent
    )

    # Initialize Crew
    crew = Crew(
        agents=[cit_alumni_agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        knowledge_sources=[excel_source],
        embedder={
            "provider": "google",
            "config": {
                "model": "models/text-embedding-004",
                "api_key": gemini_api_key,
            }
        }
    )

    # Execute the task and handle results
    try:
        result = crew.kickoff(inputs={"user_question": user_input})
        response = result["tasks_output"][0]["raw"]  # Extract raw answer
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        st.session_state["messages"].append({"role": "assistant", "content": error_message})
        st.chat_message("assistant").write(error_message)
