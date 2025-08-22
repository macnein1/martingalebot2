import streamlit as st
import sys
import os

# Add the ui directory to the path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_icon_html, setup_page_config

def render_results_page():
    """Render the results page"""
    # Main title
    st.markdown(f"""
    <div class="main-header">
        {get_icon_html("chart")} Sonuçlar Sayfası
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Bu sayfa henüz geliştirme aşamasındadır. Gelecekte detaylı sonuç analizleri burada görüntülenecek.")
    
    # Placeholder content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Geçmiş Optimizasyonlar")
        st.markdown("""
        - Optimizasyon #1 (2024-01-15)
        - Optimizasyon #2 (2024-01-14)
        - Optimizasyon #3 (2024-01-13)
        """)
    
    with col2:
        st.markdown("### İstatistikler")
        st.metric("Toplam Optimizasyon", "3")
        st.metric("Başarı Oranı", "85%")
        st.metric("Ortalama Süre", "2.5 saat")
    
    # Placeholder for future features
    st.markdown("### Gelecek Özellikler")
    st.markdown("""
    - Detaylı grafik analizleri
    - Karşılaştırmalı sonuçlar
    - Performans trendleri
    - Otomatik raporlama
    - Veri dışa aktarma
    """)
    
    # Navigation back to main page
    st.markdown("---")
    if st.button(f"{get_icon_html('target')} Ana Sayfaya Dön", type="secondary"):
        st.switch_page("main.py")

def main():
    """Results page as a standalone application"""
    setup_page_config()
    render_results_page()

if __name__ == "__main__":
    main()
