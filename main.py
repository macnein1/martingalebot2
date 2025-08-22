"""
Main Streamlit Application for DCA/Martingale "İşlemden En Hızlı Çıkış" Optimization
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ui.components.sidebar import render_sidebar
from ui.components.parameter_inputs import render_parameter_inputs
from ui.components.progress_section import render_progress_section
from ui.components.results_section import render_results_section
from ui.components.system_performance import render_system_performance
from ui.utils.config import load_page_config

# Import DCA components
from martingale_lab.orchestrator.dca_orchestrator import create_dca_orchestrator, DCAConfig
from martingale_lab.storage.experiments_store import ExperimentsStore

def main():
    """Main application entry point."""
    # Load page configuration
    load_page_config()
    
    # Initialize session state
    if 'current_experiment_id' not in st.session_state:
        st.session_state.current_experiment_id = None
    if 'optimization_running' not in st.session_state:
        st.session_state.optimization_running = False
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    
    # Main header
    st.title("🎯 DCA/Martingale Optimizer")
    st.markdown("**İşlemden En Hızlı Çıkış** - Optimized for fastest exit strategies")
    
    # Sidebar navigation
    page = render_sidebar()
    
    # Main content based on page selection
    if page == "Configuration":
        render_configuration_page()
    elif page == "Optimization":
        render_optimization_page()
    elif page == "Results":
        render_results_page()
    elif page == "System":
        render_system_performance()
    else:
        render_configuration_page()  # Default


def render_configuration_page():
    """Render the configuration page."""
    st.header("⚙️ Configuration")
    
    # Parameter inputs with new DCA contract parameters
    st.subheader("Search Space")
    col1, col2 = st.columns(2)
    
    with col1:
        overlap_min = st.number_input("Min Overlap %", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        overlap_max = st.number_input("Max Overlap %", min_value=1.0, max_value=50.0, value=30.0, step=0.5)
        
    with col2:
        orders_min = st.number_input("Min Orders", min_value=2, max_value=30, value=5, step=1)
        orders_max = st.number_input("Max Orders", min_value=2, max_value=30, value=15, step=1)
    
    # Scoring weights
    st.subheader("Scoring Weights")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        alpha = st.slider("α (Max Need)", 0.0, 1.0, 0.5, 0.1, help="Weight for maximum need percentage")
    with col2:
        beta = st.slider("β (Var Need)", 0.0, 1.0, 0.3, 0.1, help="Weight for need variance")
    with col3:
        gamma = st.slider("γ (Tail)", 0.0, 1.0, 0.2, 0.1, help="Weight for tail concentration")
    with col4:
        lambda_penalty = st.slider("λ (Penalty)", 0.0, 0.5, 0.1, 0.05, help="Weight for penalties")
    
    # Wave pattern settings
    st.subheader("Wave Pattern Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wave_pattern = st.checkbox("Enable Wave Pattern", value=False, 
                                 help="Encourage alternating strong-weak martingale patterns")
    with col2:
        wave_strong_threshold = st.number_input("Strong Threshold %", 30.0, 80.0, 50.0, 5.0,
                                              help="Martingale % considered 'strong'")
    with col3:
        wave_weak_threshold = st.number_input("Weak Threshold %", 1.0, 30.0, 10.0, 1.0,
                                            help="Martingale % considered 'very weak'")
    
    # Constraints
    st.subheader("Constraints")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tail_cap = st.slider("Tail Cap", 0.2, 0.8, 0.4, 0.05, 
                           help="Maximum volume % for last order")
    with col2:
        min_indent_step = st.number_input("Min Indent Step %", 0.01, 1.0, 0.05, 0.01,
                                        help="Minimum indent step percentage")
    with col3:
        softmax_temp = st.slider("Softmax Temperature", 0.1, 3.0, 1.0, 0.1,
                                help="Temperature for volume distribution")
    
    # Optimization settings
    st.subheader("Optimization Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_candidates = st.number_input("Candidates per Batch", 100, 10000, 1000, 100)
        max_batches = st.number_input("Max Batches", 10, 1000, 100, 10)
    
    with col2:
        n_workers = st.number_input("Parallel Workers", 1, 16, 4, 1)
        early_stop_patience = st.number_input("Early Stop Patience", 5, 50, 10, 1)
    
    with col3:
        random_seed = st.number_input("Random Seed", 0, 999999, 42, 1)
        top_k_keep = st.number_input("Top K Keep", 1000, 50000, 10000, 1000)
    
    # Save configuration
    if st.button("Save Configuration", type="primary"):
        config = DCAConfig(
            overlap_min=overlap_min,
            overlap_max=overlap_max,
            orders_min=orders_min,
            orders_max=orders_max,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_penalty=lambda_penalty,
            wave_pattern=wave_pattern,
            wave_strong_threshold=wave_strong_threshold,
            wave_weak_threshold=wave_weak_threshold,
            tail_cap=tail_cap,
            min_indent_step=min_indent_step,
            softmax_temp=softmax_temp,
            n_candidates_per_batch=n_candidates,
            max_batches=max_batches,
            n_workers=n_workers,
            early_stop_patience=early_stop_patience,
            random_seed=random_seed,
            top_k_keep=top_k_keep
        )
        
        st.session_state.dca_config = config
        st.success("Configuration saved successfully!")
        
        # Show configuration summary
        with st.expander("Configuration Summary"):
            st.json({
                "search_space": {
                    "overlap_range": f"{overlap_min}-{overlap_max}%",
                    "orders_range": f"{orders_min}-{orders_max}",
                },
                "scoring": {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "lambda": lambda_penalty
                },
                "wave_pattern": {
                    "enabled": wave_pattern,
                    "strong_threshold": wave_strong_threshold,
                    "weak_threshold": wave_weak_threshold
                },
                "constraints": {
                    "tail_cap": tail_cap,
                    "min_indent_step": min_indent_step,
                    "softmax_temp": softmax_temp
                },
                "optimization": {
                    "candidates_per_batch": n_candidates,
                    "max_batches": max_batches,
                    "workers": n_workers,
                    "early_stop_patience": early_stop_patience
                }
            })


def render_optimization_page():
    """Render the optimization page."""
    st.header("🚀 Optimization")
    
    # Check if configuration exists
    if 'dca_config' not in st.session_state:
        st.warning("Please configure the optimization parameters first.")
        if st.button("Go to Configuration"):
            st.rerun()
        return
    
    config = st.session_state.dca_config
    
    # Show current configuration summary
    with st.expander("Current Configuration"):
        st.write(f"**Search Space:** {config.overlap_min}-{config.overlap_max}% overlap, {config.orders_min}-{config.orders_max} orders")
        st.write(f"**Scoring:** α={config.alpha}, β={config.beta}, γ={config.gamma}, λ={config.lambda_penalty}")
        st.write(f"**Wave Pattern:** {'Enabled' if config.wave_pattern else 'Disabled'}")
        st.write(f"**Optimization:** {config.n_candidates_per_batch} candidates/batch, {config.max_batches} max batches")
    
    # Optimization controls
    notes = st.text_area("Experiment Notes", placeholder="Optional notes for this optimization run...")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_optimization = st.button("Start Optimization", type="primary", 
                                     disabled=st.session_state.optimization_running)
    
    with col2:
        stop_optimization = st.button("Stop Optimization", 
                                    disabled=not st.session_state.optimization_running)
    
    with col3:
        clear_results = st.button("Clear Results")
    
    # Handle optimization control
    if start_optimization and not st.session_state.optimization_running:
        run_optimization(config, notes)
    
    if stop_optimization:
        st.session_state.optimization_running = False
        st.warning("Optimization stopped by user.")
    
    if clear_results:
        st.session_state.optimization_results = None
        st.session_state.current_experiment_id = None
        st.success("Results cleared.")
    
    # Show progress if optimization is running
    if st.session_state.optimization_running:
        render_progress_section()


def run_optimization(config: DCAConfig, notes: str = ""):
    """Run the DCA optimization."""
    st.session_state.optimization_running = True
    
    # Create progress placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.container()
    
    try:
        # Create orchestrator
        orchestrator = create_dca_orchestrator(
            overlap_range=(config.overlap_min, config.overlap_max),
            orders_range=(config.orders_min, config.orders_max),
            wave_pattern=config.wave_pattern,
            n_candidates=config.n_candidates_per_batch,
            max_batches=config.max_batches,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            lambda_penalty=config.lambda_penalty,
            wave_strong_threshold=config.wave_strong_threshold,
            wave_weak_threshold=config.wave_weak_threshold,
            tail_cap=config.tail_cap,
            min_indent_step=config.min_indent_step,
            softmax_temp=config.softmax_temp,
            n_workers=config.n_workers,
            early_stop_patience=config.early_stop_patience,
            random_seed=config.random_seed,
            top_k_keep=config.top_k_keep
        )
        
        # Progress callback
        def progress_callback(info):
            progress = info["batch"] / info["total_batches"]
            progress_bar.progress(progress)
            
            status_text.text(
                f"Batch {info['batch']}/{info['total_batches']} - "
                f"Best Score: {info['best_score']:.6f} - "
                f"Speed: {info['evaluations_per_second']:.1f} eval/s"
            )
            
            # Update metrics
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Score", f"{info['best_score']:.6f}")
                with col2:
                    st.metric("Total Evaluations", f"{info['total_evaluations']:,}")
                with col3:
                    st.metric("Eval/Second", f"{info['evaluations_per_second']:.1f}")
                with col4:
                    st.metric("Candidates", f"{info['candidates_kept']:,}")
        
        # Run optimization
        results = orchestrator.run_optimization(progress_callback, notes)
        
        # Store results
        st.session_state.optimization_results = results
        st.session_state.current_experiment_id = results["experiment_id"]
        st.session_state.optimization_running = False
        
        # Success message
        st.success(f"Optimization completed! Found {len(results['best_candidates'])} candidates.")
        
        # Show summary
        stats = results["statistics"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Time", f"{stats['total_time']:.1f}s")
            st.metric("Total Evaluations", f"{stats['total_evaluations']:,}")
        
        with col2:
            st.metric("Best Score", f"{stats['best_score']:.6f}")
            st.metric("Batches Completed", stats['batches_completed'])
        
        with col3:
            st.metric("Early Stopped", "Yes" if stats['early_stopped'] else "No")
            st.metric("Sanity Violations", f"{stats['sanity_violations']:,}")
        
    except Exception as e:
        st.session_state.optimization_running = False
        st.error(f"Optimization failed: {str(e)}")


def render_results_page():
    """Render the results page."""
    experiment_id = st.session_state.get('current_experiment_id')
    render_results_section(experiment_id)


if __name__ == "__main__":
    main()


