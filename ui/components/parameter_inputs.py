import streamlit as st
from ui.utils.config import get_icon_html

def render_parameter_input(label: str, key: str, min_val: float, max_val: float, 
                          default_val: float, step: float = 0.01, format_str: str = "%.2f"):
    """Render a parameter input"""
    st.markdown(f"**{label}**")
    value = st.number_input(
        label=label,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=step,
        format=format_str,
        key=f"input_{key}",
        label_visibility="collapsed"
    )
    
    return value

def render_optimization_parameters():
    """Render the optimization parameters section"""
    st.markdown(f"""
    <div class="section-header">
        {get_icon_html("gear")} Optimizasyon Parametreleri
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize default values in session state
    if 'min_overlap' not in st.session_state:
        st.session_state.min_overlap = 1.00
    if 'max_overlap' not in st.session_state:
        st.session_state.max_overlap = 30.00
    if 'min_order' not in st.session_state:
        st.session_state.min_order = 3
    if 'max_order' not in st.session_state:
        st.session_state.max_order = 20
    
    # Parameter inputs
    min_overlap = render_parameter_input(
        "Min Overlap (%)", "min_overlap", 0.0, 100.0, st.session_state.min_overlap, 0.01
    )
    
    max_overlap = render_parameter_input(
        "Max Overlap (%)", "max_overlap", 0.0, 100.0, st.session_state.max_overlap, 0.01
    )
    
    min_order = render_parameter_input(
        "Min Sipariş", "min_order", 1, 50, st.session_state.min_order, 1, "%d"
    )
    
    max_order = render_parameter_input(
        "Max Sipariş", "max_order", 1, 50, st.session_state.max_order, 1, "%d"
    )
    
    return {
        'min_overlap': min_overlap,
        'max_overlap': max_overlap,
        'min_order': min_order,
        'max_order': max_order
    }
