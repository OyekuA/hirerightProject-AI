#!/bin/bash
set -e

DOMAIN="docs.hirerightapp.com"
COMPOSE_DIR="$HOME/project-HR"
CERT_DIR="/etc/letsencrypt/live/$DOMAIN"
DEST_DIR="$COMPOSE_DIR/nginx/certs"
LOG_FILE="/var/log/certbot-renew.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting cert renewal check ==="

cd "$COMPOSE_DIR"

log "Stopping nginx..."
docker compose stop nginx

log "Running certbot renew..."
if sudo certbot renew --quiet --standalone; then
    log "Renewal check complete."
else
    log "certbot renew exited with code $? — continuing to copy certs and restart nginx"
fi

if [ -f "$CERT_DIR/fullchain.pem" ] && [ -f "$CERT_DIR/privkey.pem" ]; then
    log "Copying certs to nginx volume..."
    mkdir -p "$DEST_DIR"
    sudo cp "$CERT_DIR/fullchain.pem" "$DEST_DIR/fullchain.pem"
    sudo cp "$CERT_DIR/privkey.pem"  "$DEST_DIR/privkey.pem"
else
    log "WARNING: cert files not found at $CERT_DIR"
fi

log "Starting nginx..."
docker compose start nginx

log "=== Done ==="
