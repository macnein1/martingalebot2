import streamlit as st
import time
from ui.components.parameter_inputs import render_optimization_parameters
from ui.components.system_performance import render_system_performance
from ui.components.progress_section import render_progress_section, update_progress
from ui.components.results_section import render_results_section
from ui.utils.config import get_icon_html, setup_page_config
from ui.utils.optimization_bridge import optimization_bridge

# Streamlit page configuration
st.set_page_config(
    page_title="Ana Sayfa - Martingale Optimizer",
    page_icon="🎯",
    layout="wide"
)

# Setup page config
setup_page_config()

# Main title
st.markdown(f"""
<div class="main-header">
    {get_icon_html("target")} Martingale Optimizer
</div>
""", unsafe_allow_html=True)

# Create two main columns
left_col, right_col = st.columns([1, 1])

with left_col:
    # Optimization Parameters Section
    parameters = render_optimization_parameters()
    
    # Start Optimization Button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button(
        f"{get_icon_html('rocket')} Optimizasyonu Başlat",
        type="primary",
        use_container_width=True,
        key="start_optimization"
    ):
        # Validate parameters first
        validation_result = optimization_bridge.validate_parameters(parameters)
        if not validation_result['success']:
            st.error(f"Parameter validation failed: {validation_result['error']}")
            st.rerun()
        
        # Create optimization session
        session_result = optimization_bridge.create_optimization_session(
            parameters=parameters,
            max_iterations=1000,
            time_limit=300.0
        )
        
        if session_result['success']:
            # Set up progress callback
            def progress_callback(progress_data):
                st.session_state.optimization_progress = progress_data['progress_percentage']
                st.session_state.current_score = progress_data['current_score']
                st.session_state.best_score = progress_data['best_score']
            
            optimization_bridge.set_progress_callback(progress_callback)
            
            # Start optimization
            start_result = optimization_bridge.start_optimization()
            if start_result['success']:
                st.session_state.optimization_running = True
                st.session_state.optimization_progress = 0
                st.session_state.session_id = session_result['session_id']
                st.success("Optimization started successfully!")
            else:
                st.error(f"Failed to start optimization: {start_result['error']}")
        else:
            st.error(f"Failed to create session: {session_result['error']}")
        
        st.rerun()
    
    # Stop Optimization Button (if running)
    if st.session_state.get('optimization_running', False):
        if st.button(
            "Optimizasyonu Durdur",
            type="secondary",
            use_container_width=True,
            key="stop_optimization"
        ):
            stop_result = optimization_bridge.stop_optimization()
            if stop_result['success']:
                st.session_state.optimization_running = False
                st.success("Optimization stopped successfully!")
            else:
                st.error(f"Failed to stop optimization: {stop_result['error']}")
            st.rerun()

with right_col:
    # System Performance Section
    metrics = render_system_performance()
    
    # Progress Section
    st.markdown("<br>", unsafe_allow_html=True)
    progress_bar = render_progress_section()
    
    # Results Section (at the bottom)
    st.markdown("<br>", unsafe_allow_html=True)
    render_results_section()

# Handle optimization logic
if st.session_state.get('optimization_running', False):
    # Check optimization status
    status_result = optimization_bridge.get_optimization_status()
    
    if status_result['success']:
        status_data = status_result['data']
        current_status = status_data.get('status', 'unknown')
        
        if current_status == 'completed':
            # Optimization completed, get results
            results_result = optimization_bridge.get_results()
            if results_result['success']:
                st.session_state.optimization_running = False
                st.session_state.optimization_results = results_result['results']
                st.session_state.optimization_statistics = results_result['statistics']
                st.success("Optimization completed successfully!")
                st.rerun()
            else:
                st.error(f"Failed to get results: {results_result['error']}")
                st.session_state.optimization_running = False
                st.rerun()
        
        elif current_status == 'error':
            # Optimization failed
            st.error(f"Optimization failed: {status_data.get('error', {}).get('error_message', 'Unknown error')}")
            st.session_state.optimization_running = False
            st.rerun()
        
        elif current_status == 'running':
            # Optimization is still running, update progress
            if 'progress' in status_data:
                progress_data = status_data['progress']
                st.session_state.optimization_progress = progress_data.get('progress_percentage', 0)
                st.session_state.current_score = progress_data.get('current_score', 0)
                st.session_state.best_score = progress_data.get('best_score', 0)
            
            # Add a small delay to prevent too frequent updates
            time.sleep(0.5)
            st.rerun()
    else:
        st.error(f"Failed to get optimization status: {status_result['error']}")
        st.session_state.optimization_running = False
        st.rerun()


