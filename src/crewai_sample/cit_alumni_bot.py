from dotenv import load_dotenv
import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.knowledge.knowledge import Knowledge
import chromadb.utils.embedding_functions.google_embedding_function as embedding_functions

# Load environment variables
load_dotenv()

# Validate API keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("MODEL")

# google_ai = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
#     api_key=gemini_api_key
# )

storage = KnowledgeStorage(
    embedder_config={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": gemini_api_key
        }
    }
)
storage.initialize_knowledge_storage()

excel_source = ExcelKnowledgeSource(file_paths=["knowledge/cit_alumni.xls"])
knowledge = Knowledge(collection_name="excel_knowledge", sources=[excel_source])

knowledge.storage = storage
knowledge.add()
gemini_llm = LLM(
    model="gemini/gemini-1.5-pro-002",
    api_key=gemini_api_key,
    temperature=0,
)

# Initialize CIT Alumni Agent
cit_alumni_agent = Agent(
    role="CIT Alumni Information Assistant",
    goal="Provide accurate information about CIT alumni.",
    backstory="You specialize in providing alumni information including payments, membership details, and personal data.",
    llm=gemini_llm
)

# Streamlit Chatbot UI
st.title("🎓 CIT Alumni Chatbot")
st.markdown("Ask me anything about CIT Alumni!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I assist you with CIT Alumni information today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle User Input
if user_input := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Create a task and initialize Crew
    task = Task(
        description=f"Answer this CIT alumni query: {user_input}",
        expected_output="A detailed answer based on the CIT alumni data.",
        agent=cit_alumni_agent
    )

    crew = Crew(
        agents=[cit_alumni_agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
    #     knowledge_sources=[knowledge],
    #     embedder={
    #     "provider": "google",
    #     "config": {
    #         "model": "models/text-embedding-004",
    #         "api_key": gemini_api_key,
    #     }
    # }
    )

    # Execute and handle results
    try:
        result = crew.kickoff(inputs={"user_question": user_input})
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").write(result)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.chat_message("assistant").write(error_message)
