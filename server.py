#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import argparse

def install_dependencies():
    print("Installing dependencies...")
    server_packages = [
        "flask",
        "PyMuPDF",
        "pdf2image",
        "torch",
        "transformers",
        "langchain",
        "opencv-python",
        "matplotlib",
        "numpy",
        "requests",
        "scikit-learn",
        "faiss-cpu",
        "python-doctr",
        "sentence-transformers"
    ]
    
    client_packages = [
        "streamlit",
        "PyPDF2",
        "requests"
    ]
    
    # Install all packages
    all_packages = list(set(server_packages + client_packages))
    
    for package in all_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Please install it manually.")
            continue
    
    print("Dependencies installed successfully!")

def start_server(port=5000):
    """Start the Flask server."""
    print(f"Starting Flask server on port {port}...")
    server_log = open("server.log", "w")
    server_proc = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=server_log,
        stderr=server_log
    )
    print(f"Flask server started (PID {server_proc.pid}).")
    return server_proc

def start_streamlit(port=8501):
    """Start the Streamlit app."""
    print(f"Starting Streamlit app on port {port}...")
    streamlit_log = open("streamlit.log", "w")
    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", f"--server.port={port}"],
        stdout=streamlit_log,
        stderr=streamlit_log
    )
    print(f"Streamlit app started (PID {streamlit_proc.pid}).")
    return streamlit_proc

def main():
    """Main function to run the entire system."""
    parser = argparse.ArgumentParser(description="Run PDF Processing Server and Streamlit Client")
    parser.add_argument("--server-port", type=int, default=5000, help="Port for Flask server")
    parser.add_argument("--streamlit-port", type=int, default=8501, help="Port for Streamlit app")
    parser.add_argument("--skip-deps", action="store_true", help="Skip installing dependencies")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("extracted_data", exist_ok=True)
    
    # Install dependencies if not skipped
    if not args.skip_deps:
        install_dependencies()
    
    # Start server and client
    try:
        server_proc = start_server(args.server_port)
        time.sleep(5)  # Give the server time to start
        
        streamlit_proc = start_streamlit(args.streamlit_port)
        
        print("\n---- Server Log (recent) ----")
        time.sleep(2)
        with open("server.log") as f:
            for i, line in enumerate(f.readlines()[-10:]):
                print(line.strip())
        
        print("\n---- Streamlit Log (recent) ----")
        with open("streamlit.log") as f:
            for i, line in enumerate(f.readlines()[-10:]):
                print(line.strip())
        
        print(f"\n🚀 System is running!")
        print(f"Flask server: http://localhost:{args.server_port}")
        print(f"Streamlit app: http://localhost:{args.streamlit_port}")
        print("\nPress Ctrl+C to stop all services...")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nShutting down services...")
        try:
            server_proc.terminate()
            streamlit_proc.terminate()
        except:
            pass
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
