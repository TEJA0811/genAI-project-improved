import streamlit as st
import requests

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000"  # Update this if needed

st.title("Unlock the wisdom in your data.")

# File upload handling
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file:
    # Upload the file to FastAPI backend
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{FASTAPI_URL}/upload/", files=files)
    if response.status_code == 200:
        st.success("File uploaded and processed successfully!")
    else:
        st.error(f"Failed to upload file: {response.text}")

# Query input and button
query = st.text_input("Ask a question")

# Add an 'Ask' button
ask_button = st.button("Ask")

if ask_button:
    # Validate if a query has been entered
    if query:
        # Send the query to FastAPI backend
        url= f"{FASTAPI_URL}/query/?question={query}"
        payload = ""
        headers = {}
        response = requests.request("POST", url, headers=headers, data=payload)
        #response = requests.get(f"{FASTAPI_URL}/query/?question={query}")
        if response.status_code == 200:
            result = response.json()
            st.header("Answer")
            st.write(result.get("answer", "No answer found."))

            # Display sources
            if "sources" in result:
                st.subheader("Sources:")
                for source in result["sources"]:
                    st.write(source)
        else:
            st.error("Error while querying the document.")
    else:
        st.error("Please enter a query before clicking 'Ask'.")
