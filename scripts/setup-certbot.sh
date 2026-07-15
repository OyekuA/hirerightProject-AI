#!/bin/bash
set -e

DOMAIN="docs.hirerightapp.com"
COMPOSE_DIR="$HOME/project-HR"
CERT_DIR="/etc/letsencrypt/live/$DOMAIN"
DEST_DIR="$COMPOSE_DIR/nginx/certs"

echo "=== Installing certbot ==="
sudo apt update && sudo apt install certbot -y

echo "=== Stopping nginx to free port 80 ==="
cd "$COMPOSE_DIR"
docker compose stop nginx

echo "=== Obtaining certificate for $DOMAIN ==="
sudo certbot certonly --standalone --agree-tos --non-interactive \
    -d "$DOMAIN" \
    --email "admin@$DOMAIN"

echo "=== Copying certs to nginx volume ==="
mkdir -p "$DEST_DIR"
sudo cp "$CERT_DIR/fullchain.pem" "$DEST_DIR/fullchain.pem"
sudo cp "$CERT_DIR/privkey.pem"  "$DEST_DIR/privkey.pem"

echo "=== Starting nginx ==="
docker compose start nginx

echo "=== Installing auto-renewal cron job ==="
RENEW_SCRIPT="$COMPOSE_DIR/scripts/certbot-renew.sh"

echo "0 3 * * * root bash $RENEW_SCRIPT" | sudo tee /etc/cron.d/certbot-renewal

echo "=== Done! Certificate will auto-renew. ==="
echo "Test renewal with: sudo certbot renew --dry-run"
