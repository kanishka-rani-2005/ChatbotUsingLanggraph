from backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)
import streamlit as st
import uuid
import tempfile
import os 

# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


st.set_page_config(
    page_title="Agentic Chatbot",
)

# Display the main application title
st.title("Agentic Chatbot with LangGraph")


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()


# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])


# Display the sidebar title
st.sidebar.title("My Conversations")


# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()



# Display all conversation threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    # Create one sidebar button for every conversation
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else:
                continue

            # Convert the LangChain message into a dictionary
            temp_messages.append({
                "role": role,
                "content": message.content
            })

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()


# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.text(message["content"])



submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"]
)


# Default user input value
user_input = None

if submission:

    # Get the text entered by the user
    user_input = submission.text

    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        # Store the temporary file path
        temporary_file_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(
                    uploaded_pdf.getvalue()
                )

                temporary_file_path = temporary_file.name


            with st.spinner(
                f"Processing {uploaded_pdf.name}..."
            ):

                ingest_rag_document(
                    temporary_file_path
                )


            # Display PDF processing confirmation
            st.toast(
                f"{uploaded_pdf.name} processed successfully.",
                icon="✅"
            )

        except Exception as error:

            # Display PDF processing error
            st.error(
                f"PDF processing failed: {error}"
            )

        finally:

            if (
                temporary_file_path
                and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)

if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })


    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)


    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }


    # Assistant streaming block
    with st.chat_message("assistant"):

        # Holder for tool status
        status_holder = {
            "box": None
        }

        # Holder for thinking status
        thinking_holder = {
            "box": None
        }

        # Track whether AI has started responding
        first_ai_chunk = True


        def extract_text(content):

            # Gemini sometimes returns normal string content
            if isinstance(content, str):
                return content

            # Gemini can also return structured content
            elif isinstance(content, list):

                result = ""

                for item in content:

                    if isinstance(item, dict):

                        text = item.get("text")

                        if text:
                            result += text

                return result

            return ""


        def ai_only_stream():

            nonlocal_first_chunk = None
            first_ai_chunk = True

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                if isinstance(message_chunk, ToolMessage):

                    # Remove Thinking indicator
                    if thinking_holder["box"] is not None:

                        thinking_holder["box"].empty()

                        thinking_holder["box"] = None


                    # Get tool name
                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )


                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f" Using `{tool_name}` …",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f" Using `{tool_name}` …",
                            state="running",
                            expanded=True
                        )


                if isinstance(message_chunk, AIMessage):

                    content = extract_text(
                        message_chunk.content
                    )


                    if content and first_ai_chunk:

                        if thinking_holder["box"] is not None:

                            thinking_holder["box"].empty()

                            thinking_holder["box"] = None

                        first_ai_chunk = False


                    if content:

                        yield content

        thinking_holder["box"] = st.empty()

        thinking_holder["box"].markdown(
            " **Thinking...**"
        )

        ai_message = st.write_stream(
            ai_only_stream()
        )


        if status_holder["box"] is not None:

            status_holder["box"].update(
                label="Tool finished",
                state="complete",
                expanded=False
            )


    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })