import streamlit as st
from rag_pipeline import RAGPipeline
from safety_classifier import safe_respond

# Set page config
st.set_page_config(page_title="Minds on Fire", page_icon="🧠", layout="wide")

@st.cache_resource
def load_pipeline():
    """
    Loads the RAG Pipeline globally to avoid reloading on every interaction.
    """
    return RAGPipeline()

def main():
    st.title("Minds on Fire — LLM Mental Health Companion")
    
    # Sidebar
    with st.sidebar:
        st.header("How it works")
        st.write("""
        This application uses a Retrieval-Augmented Generation (RAG) pipeline to provide empathetic responses based on a mental health knowledge base. 
        It retrieves relevant past counseling transcripts to contextualize its advice.
        Additionally, a BERT-based Safety Classifier continuously monitors input to detect signs of distress or crisis, automatically providing helpline numbers when needed.
        """)
        st.markdown("---")
        st.warning(
            "**System Disclaimer:**\\n"
            "This AI is for informational purposes only and is not a substitute for professional mental health care, diagnosis, or treatment. "
            "If you are in immediate danger or experiencing a medical emergency, please contact your local emergency services or a crisis hotline."
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load RAG Pipeline. Please ensure data and models are initialized. Error: {e}")
        st.stop()

    # React to user input
    if prompt := st.chat_input("How are you feeling today?"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        
        # Format history for RAG
        chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = safe_respond(prompt, chat_history, pipeline)
            st.markdown(response)
            
        # Add assistant message to state
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
