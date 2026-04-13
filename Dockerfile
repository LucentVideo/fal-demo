# Frontend-only image for Railway. Two stages: node builds the Vite app,
# then nginx serves the static dist. Railway sets $PORT at runtime and
# nginx-alpine's template entrypoint substitutes it into the server block.

# ── Build stage ──────────────────────────────────────────────────────
FROM node:20-alpine AS build

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Vite inlines VITE_* env vars at build time. On Railway, set
# VITE_LUCENT_CONTROLLER_URL in the service's variables and expose it to
# this Dockerfile as a build arg via railway.toml (see deploy notes below).
ARG VITE_LUCENT_CONTROLLER_URL=""
ENV VITE_LUCENT_CONTROLLER_URL=${VITE_LUCENT_CONTROLLER_URL}

RUN npm run build


# ── Runtime: nginx static server ─────────────────────────────────────
FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html

# nginx-alpine's entrypoint renders /etc/nginx/templates/*.template via
# envsubst at container start. Filter to only $PORT so nginx's own
# variables ($uri, $host, …) pass through unchanged.
COPY frontend/nginx.conf /etc/nginx/templates/default.conf.template
ENV NGINX_ENVSUBST_FILTER='^PORT$'
ENV PORT=8080
EXPOSE 8080
