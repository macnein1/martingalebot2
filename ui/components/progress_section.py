import streamlit as st
from utils.config import get_icon_html

def render_progress_section():
    """Render the progress monitoring section"""
    st.markdown(f"""
    <div class="section-header">
        {get_icon_html("clock")} İlerleme
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    if 'optimization_progress' not in st.session_state:
        st.session_state.optimization_progress = 0
    
    progress_bar = st.progress(st.session_state.optimization_progress)
    
    # Status information
    if st.session_state.optimization_progress == 0:
        st.info("Optimizasyon henüz başlatılmadı.")
    elif st.session_state.optimization_progress < 100:
        st.info(f"Optimizasyon devam ediyor... %{st.session_state.optimization_progress:.1f}")
    else:
        st.success("Optimizasyon tamamlandı!")
    
    # Detailed progress information
    with st.expander("Detaylı İlerleme Bilgileri"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tamamlanan İterasyon", "0/1000")
            st.metric("Geçen Süre", "00:00:00")
        
        with col2:
            st.metric("Kalan Süre", "00:00:00")
            st.metric("Hız", "0 iterasyon/sn")
    
    return progress_bar

def update_progress(progress_value: float):
    """Update the progress bar"""
    st.session_state.optimization_progress = progress_value
