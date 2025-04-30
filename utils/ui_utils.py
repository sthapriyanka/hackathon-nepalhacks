import streamlit as st
import pandas as pd
from utils.file_utils import get_download_link, get_json_download_link

def display_qa_pairs(qa_pairs, filter_option="All", sort_option="Confidence (High to Low)"):
    """
    Display QA pairs with filtering and sorting options.
    
    Args:
        qa_pairs (list): List of QA pair dictionaries
        filter_option (str): Option for filtering pairs
        sort_option (str): Option for sorting pairs
    """
    if not qa_pairs:
        st.info("No QA pairs were generated. Try adjusting the settings or using a different document.")
        return
    
    # Convert to DataFrame for easier filtering and sorting
    df = pd.DataFrame(qa_pairs)
    
    # Add filtering options
    col1, col2 = st.columns(2)
    with col1:
        filter_option = st.selectbox(
            "Filter QA Pairs",
            ["All", "High Confidence (>0.8)", "Flagged Items", "No Flags"],
            index=["All", "High Confidence (>0.8)", "Flagged Items", "No Flags"].index(filter_option)
        )
    with col2:
        sort_option = st.selectbox(
            "Sort QA Pairs",
            ["Confidence (High to Low)", "Confidence (Low to High)"],
            index=["Confidence (High to Low)", "Confidence (Low to High)"].index(sort_option)
        )
    
    # Apply filtering
    display_df = df.copy()
    if filter_option == "High Confidence (>0.8)":
        display_df = display_df[display_df["confidence"] > 0.8]
    elif filter_option == "Flagged Items":
        display_df = display_df[display_df["flags"].apply(lambda x: len(x) > 0)]
    elif filter_option == "No Flags":
        display_df = display_df[display_df["flags"].apply(lambda x: len(x) == 0)]
    
    # Apply sorting
    if sort_option == "Confidence (High to Low)":
        display_df = display_df.sort_values("confidence", ascending=False)
    else:
        display_df = display_df.sort_values("confidence", ascending=True)
    
    # Display QA pairs
    for i, row in display_df.iterrows():
        with st.container():
            st.markdown(f"""
            <div style="padding: 1rem; margin-bottom: 1rem; border-radius: 0.5rem; border: 1px solid #E5E7EB;">
                <p class="qa-question">Q: {row['question']}</p>
                <p class="qa-answer">A: {row['answer']}</p>
                <p class="qa-confidence">Confidence: {row['confidence']:.2f}</p>
                {f'<p class="qa-flag">Flags: {", ".join(row["flags"])}</p>' if row["flags"] else ''}
            </div>
            """, unsafe_allow_html=True)
    
    return display_df

def show_download_options(qa_pairs):
    """
    Display download options for QA pairs.
    
    Args:
        qa_pairs (list): List of QA pair dictionaries
    """
    st.markdown('<h3>Export Results</h3>', unsafe_allow_html=True)
    
    if not qa_pairs:
        st.info("No data available to download.")
        return
    
    df = pd.DataFrame(qa_pairs)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(get_download_link(df, "qa_pairs.csv", "Download CSV"), unsafe_allow_html=True)
    with col2:
        st.markdown(get_json_download_link(qa_pairs, "qa_pairs.json", "Download JSON"), unsafe_allow_html=True)