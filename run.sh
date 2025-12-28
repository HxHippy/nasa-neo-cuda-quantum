#!/bin/bash
# NEO Quantum - Interactive setup and run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

clear
echo -e "${CYAN}"
echo "  _   _ _____ ___    ___                  _                  "
echo " | \ | | ____/ _ \  / _ \ _   _  __ _ _ __ | |_ _   _ _ __ ___  "
echo " |  \| |  _|| | | || | | | | | |/ _\` | '_ \| __| | | | '_ \` _ \ "
echo " | |\  | |__| |_| || |_| | |_| | (_| | | | | |_| |_| | | | | | |"
echo " |_| \_|_____\___/  \__\_\\__,_|\__,_|_| |_|\__|\__,_|_| |_| |_|"
echo -e "${NC}"
echo -e "${GREEN}  Asteroid Threat Visualization with Quantum Simulation${NC}"
echo ""

# Check for .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}First time setup detected!${NC}"
    echo ""
    cp .env.example .env

    echo -e "Do you have a NASA API key? (get one free at https://api.nasa.gov)"
    read -p "(y/n, or press Enter to use demo key): " has_key

    if [[ "$has_key" == "y" || "$has_key" == "Y" ]]; then
        read -p "Enter your NASA API key: " nasa_key
        sed -i "s/NASA_API_KEY=DEMO_KEY/NASA_API_KEY=$nasa_key/" .env
        echo -e "${GREEN}API key saved!${NC}"
    else
        echo -e "${YELLOW}Using DEMO_KEY (rate limited to 30 req/hour)${NC}"
    fi
    echo ""
fi

# Check/create Python venv
if [ ! -d "venv" ]; then
    echo -e "${GREEN}Setting up Python environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}Installing Python dependencies...${NC}"
    pip install -q -r requirements.txt
    echo -e "${GREEN}Done!${NC}"
    echo ""
else
    source venv/bin/activate
fi

# Check for CUDA-Q
if ! python -c "import cudaq" 2>/dev/null; then
    echo -e "${YELLOW}CUDA-Q not found. Quantum features require NVIDIA GPU + CUDA-Q.${NC}"
    read -p "Try to install CUDA-Q? (y/n): " install_cudaq
    if [[ "$install_cudaq" == "y" || "$install_cudaq" == "Y" ]]; then
        echo "Installing CUDA-Q..."
        pip install cudaq
    else
        echo -e "${YELLOW}Skipping - quantum features will be disabled.${NC}"
    fi
    echo ""
fi

# Install npm deps if needed
if [ ! -d "ui/node_modules" ]; then
    echo -e "${GREEN}Installing frontend dependencies...${NC}"
    cd ui && npm install --silent && cd ..
    echo -e "${GREEN}Done!${NC}"
    echo ""
fi

# Ready to launch
echo -e "${CYAN}================================${NC}"
echo -e "${GREEN}Starting NEO Quantum...${NC}"
echo -e "${CYAN}================================${NC}"
echo ""

# Start backend in background
echo -e "${GREEN}[1/2]${NC} Backend on http://localhost:8000"
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload --log-level warning &
BACKEND_PID=$!

sleep 2

# Start frontend
echo -e "${GREEN}[2/2]${NC} Frontend on http://localhost:3000"
cd "$SCRIPT_DIR/ui"
npm run dev -- --open &
FRONTEND_PID=$!

cd "$SCRIPT_DIR"

echo ""
echo -e "${CYAN}================================${NC}"
echo -e "${GREEN}NEO Quantum is running!${NC}"
echo ""
echo -e "  Frontend: ${CYAN}http://localhost:3000${NC}"
echo -e "  Backend:  ${CYAN}http://localhost:8000${NC}"
echo ""
echo -e "${CYAN}================================${NC}"
echo ""
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop"

# Wait for interrupt
trap "echo ''; echo -e '${GREEN}Shutting down...${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
