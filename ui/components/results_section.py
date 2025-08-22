import streamlit as st
from ui.utils.config import get_icon_html

def render_results_section():
    """Render the final results section"""
    st.markdown(f"""
    <div class="section-header">
        {get_icon_html("document")} Son Sonuçlar
    </div>
    """, unsafe_allow_html=True)
    
    # Check if optimization has been completed
    if 'optimization_results' not in st.session_state:
        st.info("Henüz optimizasyon sonucu bulunmuyor.")
        return
    
    results = st.session_state.optimization_results
    statistics = st.session_state.get('optimization_statistics', {})
    
    # Display results in a structured way
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### En İyi Parametreler")
        if 'results' in results and results['results']:
            best_result = results['results'][0]  # First result is typically the best
            best_params = best_result.get('params', {})
            st.json({
                "Min Overlap": f"{best_params.get('min_overlap', 0):.2f}%",
                "Max Overlap": f"{best_params.get('max_overlap', 0):.2f}%",
                "Min Order": best_params.get('min_order', 0),
                "Max Order": best_params.get('max_order', 0),
                "Risk Factor": f"{best_params.get('risk_factor', 1.0):.2f}",
                "Smoothing Factor": f"{best_params.get('smoothing_factor', 0.1):.3f}",
                "Tail Weight": f"{best_params.get('tail_weight', 0.2):.2f}"
            })
        else:
            st.info("Parametre bilgisi mevcut değil.")
    
    with col2:
        st.markdown("### Performans Metrikleri")
        if 'results' in results and results['results']:
            best_result = results['results'][0]
            best_score = best_result.get('score', 0)
            breakdown = best_result.get('breakdown', {})
            
            st.metric("En İyi Skor", f"{best_score:.4f}")
            st.metric("Max Score", f"{breakdown.get('max_score', 0):.4f}")
            st.metric("Variance Score", f"{breakdown.get('variance_score', 0):.4f}")
            st.metric("Tail Score", f"{breakdown.get('tail_score', 0):.4f}")
        else:
            st.info("Performans metrikleri mevcut değil.")
    
    # Statistics section
    if statistics:
        st.markdown("### Optimizasyon İstatistikleri")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Toplam Değerlendirme", statistics.get('total_evaluations', 0))
            st.metric("Toplam Süre", f"{statistics.get('total_time', 0):.2f}s")
        
        with col2:
            st.metric("Saniyede Değerlendirme", f"{statistics.get('evaluations_per_second', 0):.1f}")
            st.metric("Ortalama Süre", f"{statistics.get('average_time_per_evaluation', 0):.4f}s")
        
        with col3:
            st.metric("En İyi Skor", f"{statistics.get('best_score', 0):.4f}")
            st.metric("Geçen Süre", f"{statistics.get('elapsed_time', 0):.2f}s")
    
    # Detailed results table
    st.markdown("### Detaylı Sonuçlar")
    if 'results' in results and results['results']:
        # Create a dataframe from the results
        import pandas as pd
        
        data = []
        for i, result in enumerate(results['results'][:10]):  # Show top 10 results
            params = result.get('params', {})
            breakdown = result.get('breakdown', {})
            
            data.append({
                'Rank': i + 1,
                'Score': f"{result.get('score', 0):.4f}",
                'Min Overlap': f"{params.get('min_overlap', 0):.2f}%",
                'Max Overlap': f"{params.get('max_overlap', 0):.2f}%",
                'Min Order': params.get('min_order', 0),
                'Max Order': params.get('max_order', 0),
                'Max Score': f"{breakdown.get('max_score', 0):.4f}",
                'Variance Score': f"{breakdown.get('variance_score', 0):.4f}",
                'Tail Score': f"{breakdown.get('tail_score', 0):.4f}",
                'Penalty Total': f"{breakdown.get('penalty_total', 0):.4f}"
            })
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Detaylı sonuç verisi mevcut değil.")
    else:
        st.info("Detaylı sonuçlar henüz mevcut değil.")
    
    # Export options
    st.markdown("### Sonuçları Dışa Aktar")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("CSV Olarak İndir"):
            st.info("CSV indirme özelliği eklenecek.")
    
    with col2:
        if st.button("JSON Olarak İndir"):
            st.info("JSON indirme özelliği eklenecek.")
    
    with col3:
        if st.button("Rapor Oluştur"):
            st.info("Rapor oluşturma özelliği eklenecek.")
