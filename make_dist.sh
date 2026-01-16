#!/bin/bash
# Create distribution tar.gz for NEOlyzer
# Places the archive in the parent directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Extract version from neolyzer.py
VERSION=$(grep -m1 "setWindowTitle.*NEOlyzer" src/neolyzer.py | sed 's/.*v\([0-9.]*\).*/\1/')
if [ -z "$VERSION" ]; then
    VERSION="unknown"
fi

DIST_NAME="neolyzer-v${VERSION}"
DIST_FILE="../${DIST_NAME}.tar.gz"
TEMP_DIR=$(mktemp -d)

echo "Creating distribution: ${DIST_NAME}"
echo "Output: ${DIST_FILE}"

# Copy files to temp directory with correct name
mkdir -p "${TEMP_DIR}/${DIST_NAME}"

# Copy source files
cp -r src "${TEMP_DIR}/${DIST_NAME}/"
cp -r scripts "${TEMP_DIR}/${DIST_NAME}/"
cp -r assets "${TEMP_DIR}/${DIST_NAME}/"

# Copy top-level files
cp install.sh "${TEMP_DIR}/${DIST_NAME}/"
cp run_neolyzer.sh "${TEMP_DIR}/${DIST_NAME}/"
cp run_setup.sh "${TEMP_DIR}/${DIST_NAME}/"
cp make_dist.sh "${TEMP_DIR}/${DIST_NAME}/"
cp requirements.txt "${TEMP_DIR}/${DIST_NAME}/"
cp README.txt "${TEMP_DIR}/${DIST_NAME}/"
cp PLATFORM_NOTES.txt "${TEMP_DIR}/${DIST_NAME}/"
cp CLAUDE.md "${TEMP_DIR}/${DIST_NAME}/"

# Copy diagnostic scripts
cp diagnose_*.py "${TEMP_DIR}/${DIST_NAME}/" 2>/dev/null || true

# Remove any backup files
find "${TEMP_DIR}/${DIST_NAME}" -name "*.backup" -delete 2>/dev/null || true
find "${TEMP_DIR}/${DIST_NAME}" -name "*.pyc" -delete 2>/dev/null || true
find "${TEMP_DIR}/${DIST_NAME}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "${TEMP_DIR}/${DIST_NAME}" -name ".DS_Store" -delete 2>/dev/null || true

# Create tar.gz
cd "${TEMP_DIR}"
tar -czf "$SCRIPT_DIR/../${DIST_NAME}.tar.gz" "${DIST_NAME}"

# Cleanup
rm -rf "${TEMP_DIR}"

# Show result
cd "$SCRIPT_DIR"
ls -lh "$DIST_FILE"
echo ""
echo "Distribution created successfully!"
echo "Contents:"
tar -tzf "$DIST_FILE" | head -25
echo "... ($(tar -tzf "$DIST_FILE" | wc -l | tr -d ' ') files total)"
