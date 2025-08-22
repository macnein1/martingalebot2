import streamlit as st
import time
import logging, sys
from ui.components.parameter_inputs import render_optimization_parameters
from ui.components.system_performance import render_system_performance
from ui.components.progress_section import render_progress_section, update_progress
from ui.utils.config import get_icon_html, setup_page_config
from ui.utils.optimization_bridge import optimization_bridge
from ui.utils.logging_buffer import tail_logs, ensure_ring_handler
from ui.utils.constants import DB_PATH

# Root logger setup (once)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mlab")
ensure_ring_handler("mlab")

# Streamlit page configuration
st.set_page_config(
    page_title="Ana Sayfa - Martingale Optimizer",
    page_icon="🎯",
    layout="wide"
)

# Setup page config
setup_page_config()

# Path/import diagnostics
try:
    import martingale_lab as _mlab
    logger.info("PYTHONPATH entries=%d", len(sys.path))
    logger.info("martingale_lab.__file__=%s", getattr(_mlab, "__file__", "<unknown>"))
except Exception:
    logger.exception("Import martingale_lab failed")

# Main title
st.markdown(f"""
<div class="main-header">
    {get_icon_html("target")} Martingale Optimizer
</div>
""", unsafe_allow_html=True)

# Optimization Parameters Section
parameters = render_optimization_parameters()

# System Performance Section
st.markdown("<br>", unsafe_allow_html=True)
metrics = render_system_performance()

# Start Optimization Button
st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    f"{get_icon_html('rocket')} Optimizasyonu Başlat",
    type="primary",
    use_container_width=True,
    key="start_optimization"
):
    try:
        validation_result = optimization_bridge.validate_parameters(parameters)
        if not validation_result['success']:
            st.error(f"Parameter validation failed: {validation_result['error']}")
            st.stop()
        start_result = optimization_bridge.start_optimization(parameters, db_path=DB_PATH)
        if not start_result.get('success'):
            raise RuntimeError(start_result.get('error', 'unknown'))
        st.session_state["run_id"] = start_result.get('run_id')
        st.session_state["job_running"] = True
        st.success("Optimization started!")
        st.rerun()
    except Exception as e:
        st.exception(e)

# Stop Optimization Button (if running)
if st.session_state.get('job_running', False):
    if st.button(
        "Optimizasyonu Durdur",
        type="secondary",
        use_container_width=True,
        key="stop_optimization"
    ):
        try:
            stop_result = optimization_bridge.stop_optimization()
            if not stop_result.get('success'):
                raise RuntimeError(stop_result.get('error', 'unknown'))
            st.session_state["job_running"] = False
            st.success("Optimization stopped successfully!")
        except Exception as e:
            st.exception(e)
        st.rerun()

# Progress Section (only show when optimization is running)
if st.session_state.get('job_running', False):
    st.markdown("<br>", unsafe_allow_html=True)
    render_progress_section()

# Live status/logs
if st.session_state.get("job_running"):
    try:
        status_result = optimization_bridge.get_optimization_status()
        if not status_result.get('success'):
            raise RuntimeError(status_result.get('error', 'unknown'))
        with st.status("Running…", expanded=True) as status:
            for line in tail_logs("mlab", last_n=200):
                st.write(line)
            if status_result.get('data', {}).get('status') == 'completed':
                st.session_state["job_running"] = False
                status.update(label="Done", state="complete")
                st.success("Optimization completed successfully!")
            else:
                time.sleep(0.5)
                st.rerun()
    except Exception as e:
        st.exception(e)


