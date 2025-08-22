import streamlit as st

def setup_page_config():
    """Setup page configuration and styling"""
    st.set_page_config(
        page_title="Martingale Optimizer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .optimization-button {
        background-color: #28a745;
        color: white;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .optimization-button:hover {
        background-color: #218838;
    }
    
    .parameter-input {
        margin-bottom: 1rem;
    }
    
    .progress-section {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def get_icon_html(icon_name: str) -> str:
    """Get HTML for icons"""
    icons = {
        "target": "🎯",
        "chart": "📊",
        "gear": "⚙️",
        "wrench": "🔧",
        "rocket": "🚀",
        "clock": "⏰",
        "document": "📄"
    }
    return icons.get(icon_name, "📌")
