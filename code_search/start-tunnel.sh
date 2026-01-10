#!/bin/bash
# Start Cloudflare tunnel to PostgreSQL
# This creates a local listener on port 5433 that tunnels to your VPS PostgreSQL

cloudflared access tcp --hostname postgres-codetalk.el-jefe.me --url localhost:5433
