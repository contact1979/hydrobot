# /dashboard/app.py

import logging
import sys
import os
import dash
import dash_bootstrap_components as dbc
from dash import html

# Add the root directory to the Python module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config # To potentially set Dash debug mode based on LOG_LEVEL
# Use explicit absolute imports for modules within the dashboard package
import dashboard.layouts as layouts
import dashboard.callbacks as callbacks
# from .. import config # Use this if config was one level up from dashboard

log = logging.getLogger(__name__)

# --- Initialize Dash App ---
# Load external stylesheets (Bootstrap theme)
external_stylesheets = [dbc.themes.BOOTSTRAP] # Or choose another theme like CYBORG, DARKLY

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True, # Set to True if callbacks are in different files
    title="Cyclone v2 Dashboard"
)
server = app.server # Expose server for potential WSGI deployment (e.g., Gunicorn)

# --- Define App Layout ---
app.layout = layouts.create_main_layout()

# --- Register Callbacks ---
# Call the function from callbacks.py to register all callbacks
callbacks.register_callbacks(app)

# --- Run the App ---
if __name__ == '__main__':
    # Setup basic logging for dashboard process if run directly
    # Note: If run via main.py, logging might be configured there already
    # Check if logging is already configured by root logger to avoid duplicate handlers
    if not logging.getLogger().hasHandlers():
        log_level_dashboard = getattr(logging, config.LOG_LEVEL, logging.INFO)
        logging.basicConfig(level=log_level_dashboard, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    log.info("Starting Dash development server...")
    # Set debug=True for development (enables hot-reloading, detailed error pages)
    # Set debug=False for production deployment
    debug_mode = config.LOG_LEVEL == "DEBUG"
    # Use app.run (corrected from run_server)
    app.run(debug=debug_mode, host='0.0.0.0', port=8051) # Using port 8051 as decided earlier
    # Use host='0.0.0.0' to make it accessible on your network
