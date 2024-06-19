import streamlit as st

st.title("Writing Tools Prototypes")

st.markdown("Click one of the links below to see a prototype in action.")

st.page_link("pages/1_Rewrite.py", label="Rewrite with predictions", icon="📝")
st.page_link("pages/2_Highlights.py", label="Highlight locations for possible edits", icon="🖍️")

st.markdown("*Note*: These services send data to a remote server for processing. The server logs requests. Don't use sensitive or identifiable information on this page.")
