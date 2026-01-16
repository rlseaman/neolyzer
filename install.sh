#!/bin/bash
# NEOlyzer - Cross-Platform Installation Script
# Supports: macOS (Intel/Apple Silicon), RHEL/Rocky, Debian/Ubuntu, Raspberry Pi
# 
# Usage:
#   ./install.sh                    # Auto-detect Python
#   ./install.sh --python /path/to/python3
#   ./install.sh --skip-deps        # Skip system dependency installation
#   ./install.sh --help

set -e  # Exit on error

VERSION="2.03"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Default settings
CUSTOM_PYTHON=""
SKIP_DEPS=false
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10  # Minimum Python 3.10

#######################################
# Print colored messages
#######################################
print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

#######################################
# Show help
#######################################
show_help() {
    cat << EOF
NEOlyzer Installation Script v${VERSION}

Usage: ./install.sh [OPTIONS]

Options:
  --python PATH    Use specific Python interpreter
                   Example: --python /usr/local/bin/python3.12
                   Example: --python ~/python312/bin/python3
  
  --skip-deps      Skip system dependency installation
                   (use if you've already installed them)
  
  --help           Show this help message

Environment Variables:
  NEO_PYTHON       Alternative way to specify Python path
                   Example: NEO_PYTHON=/opt/python3.12/bin/python3 ./install.sh

Supported Platforms:
  - macOS (Intel and Apple Silicon)
  - RHEL / Rocky Linux / CentOS / Fedora
  - Debian / Ubuntu
  - Raspberry Pi OS

Examples:
  # Standard installation (auto-detect Python)
  ./install.sh
  
  # Use locally installed Python (RHEL with frozen system Python)
  ./install.sh --python /home/user/python312/bin/python3
  
  # Use pyenv Python
  ./install.sh --python ~/.pyenv/versions/3.12.0/bin/python3
  
  # Skip system deps (already installed)
  ./install.sh --skip-deps

Building Python Locally (for systems with frozen Python):

  IMPORTANT: You must have OpenSSL development libraries installed
  BEFORE building Python, or pip won't work!

  # Step 1: Install OpenSSL dev libraries (requires sudo)
  RHEL/Rocky:    sudo dnf install openssl-devel bzip2-devel libffi-devel
  Debian/Ubuntu: sudo apt install libssl-dev libbz2-dev libffi-dev

  # Step 2: Download and build Python
  wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
  tar xzf Python-3.12.0.tgz && cd Python-3.12.0
  ./configure --prefix=\$HOME/python312 --enable-optimizations
  make -j\$(nproc) && make install

  # Step 3: Run installer
  cd .. && ./install.sh --python \$HOME/python312/bin/python3

EOF
    exit 0
}

#######################################
# Parse command line arguments
#######################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --python)
                CUSTOM_PYTHON="$2"
                shift 2
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check environment variable
    if [ -z "$CUSTOM_PYTHON" ] && [ -n "$NEO_PYTHON" ]; then
        CUSTOM_PYTHON="$NEO_PYTHON"
    fi
}

#######################################
# Detect operating system and distribution
#######################################
detect_os() {
    OS_TYPE=""
    OS_DISTRO=""
    OS_VERSION=""
    ARCH=$(uname -m)
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        OS_VERSION=$(sw_vers -productVersion)
        if [[ "$ARCH" == "arm64" ]]; then
            OS_DISTRO="apple_silicon"
        else
            OS_DISTRO="intel"
        fi
    elif [[ -f /etc/os-release ]]; then
        source /etc/os-release
        OS_TYPE="linux"
        
        case "$ID" in
            rhel|rocky|centos|almalinux)
                OS_DISTRO="rhel"
                OS_VERSION="$VERSION_ID"
                ;;
            fedora)
                OS_DISTRO="fedora"
                OS_VERSION="$VERSION_ID"
                ;;
            debian)
                OS_DISTRO="debian"
                OS_VERSION="$VERSION_ID"
                ;;
            ubuntu)
                OS_DISTRO="ubuntu"
                OS_VERSION="$VERSION_ID"
                ;;
            raspbian)
                OS_DISTRO="raspbian"
                OS_VERSION="$VERSION_ID"
                ;;
            *)
                # Check for Raspberry Pi specifically
                if [[ -f /proc/device-tree/model ]] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
                    OS_DISTRO="raspbian"
                else
                    OS_DISTRO="$ID"
                fi
                OS_VERSION="$VERSION_ID"
                ;;
        esac
    else
        OS_TYPE="unknown"
        OS_DISTRO="unknown"
    fi
    
    # Detect if running on Raspberry Pi hardware
    IS_RASPBERRY_PI=false
    if [[ -f /proc/device-tree/model ]] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
        IS_RASPBERRY_PI=true
    fi
}

#######################################
# Check Python version meets requirements
#######################################
check_python_version() {
    local python_bin="$1"
    
    if ! PYTHON_VERSION=$("$python_bin" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null); then
        return 1
    fi
    
    local major=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    local minor=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    
    if [ "$major" -gt "$MIN_PYTHON_MAJOR" ]; then
        return 0
    elif [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
        return 0
    fi
    
    return 1
}

#######################################
# Check Python has SSL support
#######################################
check_python_ssl() {
    local python_bin="$1"
    
    if "$python_bin" -c "import ssl" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

#######################################
# Find suitable Python interpreter
#######################################
find_python() {
    PYTHON_CMD=""
    PYTHON_VERSION=""
    
    # If custom Python specified, verify it
    if [ -n "$CUSTOM_PYTHON" ]; then
        if [ -x "$CUSTOM_PYTHON" ]; then
            if check_python_version "$CUSTOM_PYTHON"; then
                # Check SSL support
                if ! check_python_ssl "$CUSTOM_PYTHON"; then
                    echo ""
                    print_error "Python at $CUSTOM_PYTHON was built WITHOUT SSL support!"
                    echo ""
                    echo "pip requires SSL to download packages from PyPI."
                    echo ""
                    echo "To fix this, you need to rebuild Python with SSL support:"
                    echo ""
                    echo "  1. First, install OpenSSL development libraries:"
                    echo "     RHEL/Rocky/CentOS: sudo dnf install openssl-devel"
                    echo "     Debian/Ubuntu:     sudo apt install libssl-dev"
                    echo ""
                    echo "  2. Then rebuild Python:"
                    echo "     cd Python-3.12.0  # (your Python source directory)"
                    echo "     make clean"
                    echo "     ./configure --prefix=\$HOME/python312 --enable-optimizations"
                    echo "     make -j\$(nproc)"
                    echo "     make install"
                    echo ""
                    echo "  3. Then re-run this installer:"
                    echo "     ./install.sh --python \$HOME/python312/bin/python3"
                    echo ""
                    exit 1
                fi
                PYTHON_CMD="$CUSTOM_PYTHON"
                print_status "Using specified Python: $CUSTOM_PYTHON (version $PYTHON_VERSION)"
                return 0
            else
                print_error "Specified Python ($CUSTOM_PYTHON) version $PYTHON_VERSION does not meet minimum requirement ($MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR)"
                exit 1
            fi
        else
            print_error "Specified Python not found or not executable: $CUSTOM_PYTHON"
            exit 1
        fi
    fi
    
    # Search for Python in common locations
    local PYTHON_SEARCH=(
        # Explicit version commands (in PATH)
        "python3.14"
        "python3.13"
        "python3.12"
        "python3.11"
        "python3.10"
        "python3"
        # Homebrew on macOS (Apple Silicon)
        "/opt/homebrew/bin/python3.14"
        "/opt/homebrew/bin/python3.13"
        "/opt/homebrew/bin/python3.12"
        "/opt/homebrew/bin/python3.11"
        "/opt/homebrew/bin/python3.10"
        "/opt/homebrew/bin/python3"
        # Homebrew on macOS (Intel)
        "/usr/local/bin/python3.14"
        "/usr/local/bin/python3.13"
        "/usr/local/bin/python3.12"
        "/usr/local/bin/python3.11"
        "/usr/local/bin/python3.10"
        "/usr/local/bin/python3"
        # pyenv
        "$HOME/.pyenv/shims/python3"
        # Common local install locations
        "$HOME/python3/bin/python3"
        "$HOME/python312/bin/python3"
        "$HOME/python311/bin/python3"
        "$HOME/.local/bin/python3"
        "/usr/local/python3/bin/python3"
        # Deadsnakes PPA (Ubuntu)
        "/usr/bin/python3.14"
        "/usr/bin/python3.13"
        "/usr/bin/python3.12"
        "/usr/bin/python3.11"
        "/usr/bin/python3.10"
    )
    
    for python_path in "${PYTHON_SEARCH[@]}"; do
        # Check if command exists (for bare commands in PATH)
        if command -v "$python_path" &>/dev/null; then
            local resolved=$(command -v "$python_path")
            if check_python_version "$resolved"; then
                if check_python_ssl "$resolved"; then
                    PYTHON_CMD="$resolved"
                    print_status "Found Python: $PYTHON_CMD (version $PYTHON_VERSION)"
                    return 0
                fi
            fi
        # Check if path exists directly
        elif [ -x "$python_path" ] 2>/dev/null; then
            if check_python_version "$python_path"; then
                if check_python_ssl "$python_path"; then
                    PYTHON_CMD="$python_path"
                    print_status "Found Python: $PYTHON_CMD (version $PYTHON_VERSION)"
                    return 0
                fi
            fi
        fi
    done
    
    # Also check pyenv versions directory
    if [ -d "$HOME/.pyenv/versions" ]; then
        for pyenv_python in $HOME/.pyenv/versions/*/bin/python3; do
            if [ -x "$pyenv_python" ] && check_python_version "$pyenv_python"; then
                if check_python_ssl "$pyenv_python"; then
                    PYTHON_CMD="$pyenv_python"
                    print_status "Found Python (pyenv): $PYTHON_CMD (version $PYTHON_VERSION)"
                    return 0
                fi
            fi
        done
    fi
    
    # Python not found
    return 1
}

#######################################
# Install system dependencies
#######################################
install_system_deps() {
    if [ "$SKIP_DEPS" = true ]; then
        print_info "Skipping system dependency installation (--skip-deps)"
        return 0
    fi
    
    echo ""
    echo "Installing system dependencies..."
    
    case "$OS_TYPE-$OS_DISTRO" in
        macos-*)
            install_deps_macos
            ;;
        linux-rhel|linux-fedora)
            install_deps_rhel
            ;;
        linux-debian|linux-ubuntu)
            install_deps_debian
            ;;
        linux-raspbian)
            install_deps_raspbian
            ;;
        *)
            print_warning "Unknown OS ($OS_TYPE-$OS_DISTRO). You may need to install dependencies manually:"
            echo "  - HDF5 development libraries"
            echo "  - Qt6 libraries"
            echo "  - OpenGL libraries"
            ;;
    esac
}

#######################################
# macOS dependencies (Homebrew)
#######################################
install_deps_macos() {
    if ! command -v brew &> /dev/null; then
        print_warning "Homebrew not found. Some dependencies may fail."
        print_info "Install Homebrew: https://brew.sh"
        return 0
    fi
    
    # Check/install HDF5
    if [ ! -d "/opt/homebrew/opt/hdf5" ] && [ ! -d "/usr/local/opt/hdf5" ]; then
        print_info "Installing HDF5..."
        brew install hdf5 || print_warning "HDF5 install failed"
    fi
    
    # Set HDF5 path for h5py build
    if [ -d "/opt/homebrew/opt/hdf5" ]; then
        export HDF5_DIR=/opt/homebrew/opt/hdf5
    elif [ -d "/usr/local/opt/hdf5" ]; then
        export HDF5_DIR=/usr/local/opt/hdf5
    fi
    
    print_status "macOS dependencies ready"
}

#######################################
# RHEL/Rocky/CentOS/Fedora dependencies
#######################################
install_deps_rhel() {
    local PKG_MGR="dnf"
    if ! command -v dnf &> /dev/null; then
        PKG_MGR="yum"
    fi
    
    print_info "Checking dependencies via $PKG_MGR..."
    
    # Check if we have sudo access
    if sudo -n true 2>/dev/null; then
        print_info "Installing system packages (this may take a moment)..."
        
        # Build dependencies (needed for compiling Python and packages)
        sudo $PKG_MGR install -y \
            openssl-devel \
            bzip2-devel \
            libffi-devel \
            readline-devel \
            sqlite-devel \
            zlib-devel \
            xz-devel \
            2>/dev/null || true
        
        # Runtime dependencies
        sudo $PKG_MGR install -y \
            hdf5-devel \
            mesa-libGL \
            mesa-libGL-devel \
            libxcb \
            xcb-util-wm \
            xcb-util-image \
            xcb-util-keysyms \
            xcb-util-renderutil \
            xcb-util-cursor \
            libxkbcommon-x11 \
            gcc \
            gcc-c++ \
            2>/dev/null || print_warning "Some packages may not be available"
        
        # Qt6 packages vary by version
        sudo $PKG_MGR install -y qt6-qtbase qt6-qtbase-gui 2>/dev/null || \
        sudo $PKG_MGR install -y qt5-qtbase qt5-qtbase-gui 2>/dev/null || \
        print_warning "Qt packages not found - PyQt6 will install its own"
    else
        print_warning "No sudo access. Please ask your administrator to install:"
        echo ""
        echo "  # Build dependencies (for compiling Python with SSL):"
        echo "  $PKG_MGR install openssl-devel bzip2-devel libffi-devel"
        echo ""
        echo "  # Runtime dependencies (REQUIRED for Qt/GUI):"
        echo "  $PKG_MGR install xcb-util-cursor libxkbcommon-x11 mesa-libGL"
        echo ""
        echo "  # Optional (for HDF5 cache):"
        echo "  $PKG_MGR install hdf5-devel"
        echo ""
        echo "Or install without system packages using:"
        echo "  ./install.sh --skip-deps"
    fi
    
    print_status "RHEL/Rocky dependencies checked"
}

#######################################
# Debian/Ubuntu dependencies
#######################################
install_deps_debian() {
    print_info "Checking dependencies via apt..."
    
    if sudo -n true 2>/dev/null; then
        print_info "Installing system packages..."
        sudo apt-get update -qq 2>/dev/null || true
        sudo apt-get install -y \
            libhdf5-dev \
            libgl1-mesa-glx \
            libgl1-mesa-dev \
            libxcb-xinerama0 \
            libxcb-cursor0 \
            libxkbcommon-x11-0 \
            build-essential \
            2>/dev/null || print_warning "Some packages may not be available"
        
        # Qt6 on Ubuntu
        sudo apt-get install -y qt6-base-dev 2>/dev/null || \
        print_info "Qt6 packages not found - PyQt6 will install its own"
    else
        print_warning "No sudo access. Please ask your administrator to install:"
        echo "  apt install libhdf5-dev libgl1-mesa-glx build-essential"
    fi
    
    print_status "Debian/Ubuntu dependencies checked"
}

#######################################
# Raspberry Pi dependencies
#######################################
install_deps_raspbian() {
    print_info "Installing Raspberry Pi dependencies..."
    
    if sudo -n true 2>/dev/null; then
        sudo apt-get update -qq 2>/dev/null || true
        sudo apt-get install -y \
            libhdf5-dev \
            libgl1-mesa-glx \
            libxcb-xinerama0 \
            libxcb-cursor0 \
            python3-pyqt6 \
            python3-numpy \
            build-essential \
            2>/dev/null || print_warning "Some packages may not be available"
    else
        print_warning "No sudo access. Please install packages manually."
    fi
    
    print_status "Raspberry Pi dependencies checked"
}

#######################################
# Create virtual environment
#######################################
create_venv() {
    echo ""
    echo "Creating virtual environment..."
    
    # Remove old venv if exists
    if [ -d "venv" ]; then
        print_info "Removing existing virtual environment..."
        rm -rf venv
    fi
    
    # Create new venv
    # On Raspberry Pi, we may want to include system packages
    if [ "$IS_RASPBERRY_PI" = true ]; then
        "$PYTHON_CMD" -m venv --system-site-packages venv
        print_info "Created venv with system site-packages (Raspberry Pi)"
    else
        "$PYTHON_CMD" -m venv venv
    fi
    
    # Activate
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip --quiet
    
    print_status "Virtual environment created and activated"
}

#######################################
# Install Python packages
#######################################
install_python_packages() {
    echo ""
    echo "Installing Python packages..."
    echo "(This may take 5-10 minutes)"
    echo ""
    
    # Track if PyQt installed
    PYQT_INSTALLED=""
    
    # Core packages
    pip install numpy --quiet && print_status "numpy"
    pip install cython --quiet && print_status "cython"
    pip install pandas --quiet && print_status "pandas"
    pip install sqlalchemy --quiet && print_status "sqlalchemy"
    
    # HDF5 - may need special handling
    if [ -n "$HDF5_DIR" ]; then
        HDF5_DIR="$HDF5_DIR" pip install h5py --quiet && print_status "h5py"
    else
        pip install h5py --quiet 2>/dev/null && print_status "h5py" || {
            print_warning "h5py install failed - cache features may not work"
            print_info "Try: sudo apt/dnf install libhdf5-dev, then reinstall"
        }
    fi
    
    # Astronomy packages
    pip install skyfield --quiet && print_status "skyfield"
    pip install jplephem --quiet && print_status "jplephem"
    
    # Visualization
    pip install matplotlib --quiet && print_status "matplotlib"
    
    # Qt6 - Try binary wheel first, then PyQt5 as fallback
    # PyQt6 from source requires Qt6 dev tools (qmake6) which aren't on RHEL 8
    PYQT_INSTALLED=""
    if pip install PyQt6 --only-binary :all: --quiet 2>/dev/null; then
        print_status "PyQt6 (binary wheel)"
        PYQT_INSTALLED="6"
    elif pip install PyQt6 --quiet 2>/dev/null; then
        print_status "PyQt6 (built from source)"
        PYQT_INSTALLED="6"
    elif pip install PyQt5 --quiet 2>/dev/null; then
        print_warning "PyQt6 unavailable, installed PyQt5 as fallback"
        PYQT_INSTALLED="5"
    else
        print_error "Neither PyQt6 nor PyQt5 could be installed!"
        echo ""
        echo "Possible solutions:"
        echo "  RHEL 8: Try using system Python with PyQt5:"
        echo "    sudo dnf install python3-pyqt5"
        echo "    ./install.sh --python /usr/bin/python3"
        echo ""
        echo "  Or install Qt6 development tools to build PyQt6:"
        echo "    (Qt6 is not in RHEL 8 repos - requires EPEL or manual install)"
        echo ""
        PYQT_INSTALLED=""
    fi
    
    # Utilities
    pip install requests tqdm certifi --quiet && print_status "requests, tqdm, certifi"
    
    # Optional: astropy for galactic coordinates
    pip install astropy --quiet 2>/dev/null && print_status "astropy (optional)" || print_info "astropy skipped (optional)"
    
    # Optional: scipy for smoother density contours
    pip install scipy --quiet 2>/dev/null && print_status "scipy (optional)" || print_info "scipy skipped (optional)"
    
    echo ""
    print_status "All Python packages installed"
}

#######################################
# Create launcher scripts
#######################################
create_launchers() {
    echo ""
    echo "Creating launcher scripts..."
    
    # Main launcher script
    cat > run_neolyzer.sh << 'LAUNCHER_EOF'
#!/bin/bash
# NEOlyzer Launcher
# Automatically finds and uses the correct Python environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use virtual environment Python
if [ -f "./venv/bin/python" ]; then
    exec ./venv/bin/python src/neolyzer.py "$@"
else
    echo "Error: Virtual environment not found. Run ./install.sh first."
    exit 1
fi
LAUNCHER_EOF
    chmod +x run_neolyzer.sh
    
    # Setup launcher
    cat > run_setup.sh << 'SETUP_EOF'
#!/bin/bash
# NEOlyzer Setup Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "./venv/bin/python" ]; then
    exec ./venv/bin/python scripts/setup_database.py "$@"
else
    echo "Error: Virtual environment not found. Run ./install.sh first."
    exit 1
fi
SETUP_EOF
    chmod +x run_setup.sh
    
    print_status "Launcher scripts created (run_neolyzer.sh, run_setup.sh)"
}

#######################################
# Save configuration
#######################################
save_config() {
    cat > .neo_config << CONFIG_EOF
# NEOlyzer Configuration
# Generated by install.sh on $(date)
# This file is used by the application to track installation details

OS_TYPE="$OS_TYPE"
OS_DISTRO="$OS_DISTRO"
OS_VERSION="$OS_VERSION"
ARCH="$ARCH"
IS_RASPBERRY_PI="$IS_RASPBERRY_PI"
PYTHON_CMD="$PYTHON_CMD"
PYTHON_VERSION="$PYTHON_VERSION"
INSTALL_DATE="$(date -Iseconds 2>/dev/null || date)"
NEO_VERSION="$VERSION"
CONFIG_EOF
    
    print_status "Configuration saved to .neo_config"
}

#######################################
# Main installation flow
#######################################
main() {
    echo "======================================================================="
    echo "NEOlyzer v${VERSION} - Cross-Platform Installation"
    echo "======================================================================="
    echo ""
    
    # Acquire lock to prevent concurrent installs (Linux only - flock not on macOS)
    LOCK_FILE="$SCRIPT_DIR/.install.lock"
    HAVE_LOCK=false
    
    if command -v flock &>/dev/null; then
        # Linux: use flock
        exec 200>"$LOCK_FILE"
        if ! flock -n 200; then
            print_error "Another installation is already running!"
            echo ""
            echo "If you're sure no other install is running, remove the lock file:"
            echo "  rm $LOCK_FILE"
            echo ""
            exit 1
        fi
        echo "PID: $$" >&200
        echo "Started: $(date)" >&200
        HAVE_LOCK=true
    else
        # macOS: simple lock file check (less robust but works)
        if [ -f "$LOCK_FILE" ]; then
            # Check if the PID in the lock file is still running
            if [ -f "$LOCK_FILE" ]; then
                OLD_PID=$(head -1 "$LOCK_FILE" 2>/dev/null | grep -o '[0-9]*')
                if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
                    print_error "Another installation is already running (PID: $OLD_PID)!"
                    echo ""
                    echo "If you're sure no other install is running, remove the lock file:"
                    echo "  rm $LOCK_FILE"
                    echo ""
                    exit 1
                else
                    # Stale lock file - remove it
                    rm -f "$LOCK_FILE"
                fi
            fi
        fi
        echo "$$" > "$LOCK_FILE"
        echo "$(date)" >> "$LOCK_FILE"
        HAVE_LOCK=true
    fi
    
    # Cleanup function
    cleanup() {
        if [ "$HAVE_LOCK" = true ]; then
            rm -f "$LOCK_FILE" 2>/dev/null
        fi
    }
    trap cleanup EXIT
    
    # Parse arguments
    parse_args "$@"
    
    # Detect OS
    echo "Detecting system..."
    detect_os
    print_status "OS: $OS_TYPE ($OS_DISTRO ${OS_VERSION:-unknown})"
    print_status "Architecture: $ARCH"
    if [ "$IS_RASPBERRY_PI" = true ]; then
        print_status "Hardware: Raspberry Pi"
    fi
    
    # Find Python
    echo ""
    echo "Locating Python interpreter..."
    if ! find_python; then
        echo ""
        print_error "Python $MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR+ (with SSL support) not found!"
        echo ""
        echo "Options:"
        echo ""
        echo "  1. Specify a local Python installation:"
        echo "     ./install.sh --python /path/to/python3"
        echo ""
        echo "  2. Use pyenv to install Python:"
        echo "     curl https://pyenv.run | bash"
        echo "     pyenv install 3.12.0"
        echo "     ./install.sh --python ~/.pyenv/versions/3.12.0/bin/python3"
        echo ""
        echo "  3. Build Python locally (for frozen system Python):"
        echo ""
        echo "     # FIRST: Install build dependencies (requires sudo once)"
        echo "     sudo dnf install openssl-devel bzip2-devel libffi-devel  # RHEL/Rocky"
        echo "     # OR"
        echo "     sudo apt install libssl-dev libbz2-dev libffi-dev        # Debian/Ubuntu"
        echo ""
        echo "     # THEN: Build Python"
        echo "     wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz"
        echo "     tar xzf Python-3.12.0.tgz && cd Python-3.12.0"
        echo "     ./configure --prefix=\$HOME/python312 --enable-optimizations"
        echo "     make -j\$(nproc) && make install"
        echo ""
        echo "     # Verify SSL works:"
        echo "     \$HOME/python312/bin/python3 -c 'import ssl; print(\"SSL OK\")'"
        echo ""
        echo "     # Install NEO Visualizer:"
        echo "     cd /path/to/neo_visualizer"
        echo "     ./install.sh --python \$HOME/python312/bin/python3"
        echo ""
        exit 1
    fi
    
    # Install system dependencies
    install_system_deps
    
    # Create virtual environment
    create_venv
    
    # Install Python packages
    install_python_packages
    
    # Create launcher scripts
    create_launchers
    
    # Save configuration
    save_config
    
    # Check for critical Qt runtime dependencies on Linux
    if [ "$OS_TYPE" = "linux" ] && [ "$PYQT_INSTALLED" = "6" ]; then
        echo ""
        echo "Checking Qt6 runtime dependencies..."
        
        # Check for xcb-cursor (required by Qt 6.5+)
        if ! ldconfig -p 2>/dev/null | grep -q "libxcb-cursor"; then
            if ! [ -f /usr/lib64/libxcb-cursor.so.0 ] && ! [ -f /usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0 ]; then
                echo ""
                print_warning "xcb-cursor library not found!"
                echo ""
                echo "Qt 6.5+ requires xcb-cursor for the GUI. Please install it:"
                echo ""
                echo "  RHEL/Rocky/Fedora:  sudo dnf install xcb-util-cursor"
                echo "  Debian/Ubuntu:      sudo apt install libxcb-cursor0"
                echo ""
                echo "Without this library, the visualizer will crash on startup."
                echo ""
            fi
        fi
        
        # Check for libxkbcommon-x11
        if ! ldconfig -p 2>/dev/null | grep -q "libxkbcommon-x11"; then
            print_warning "libxkbcommon-x11 may be missing (needed for keyboard input)"
            echo "  RHEL/Rocky/Fedora:  sudo dnf install libxkbcommon-x11"
            echo "  Debian/Ubuntu:      sudo apt install libxkbcommon-x11-0"
        fi
    fi
    
    # Run setup
    echo ""
    echo "======================================================================="
    echo "Running Database Setup"
    echo "======================================================================="
    echo ""
    
    ./venv/bin/python scripts/setup_database.py
    
    # Done
    echo ""
    echo "======================================================================="
    echo "Installation Complete!"
    echo "======================================================================="
    echo ""
    echo "System Information:"
    echo "  OS:           $OS_TYPE ($OS_DISTRO ${OS_VERSION:-unknown})"
    echo "  Architecture: $ARCH"
    echo "  Python:       $PYTHON_CMD ($PYTHON_VERSION)"
    if [ "$IS_RASPBERRY_PI" = true ]; then
        echo "  Hardware:     Raspberry Pi"
    fi
    echo ""
    echo "To launch NEOlyzer:"
    echo "  ./run_neolyzer.sh"
    echo "  # or"
    echo "  ./venv/bin/python src/neolyzer.py"
    echo ""
    echo "To re-run setup (download data, rebuild cache):"
    echo "  ./run_setup.sh"
    echo ""
}

# Run main
main "$@"
