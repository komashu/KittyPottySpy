# KittyPottySpy Django Web
Minimal Django app for reviewing detections & identifications.

## Quick start (Docker)
```bash
cp kittyweb/.env.example kittyweb/.env
# edit .env (SECRET_KEY etc.)

docker build -t kittyweb:latest kittyweb
docker run --rm -p 127.0.0.1:8000:8000   --env-file kittyweb/.env   -v "$(pwd)":/app   kittyweb:latest
```
Then open http://localhost:8000 (use SSH port-forward on remote servers).

## Compose snippet
```yaml
  django:
    image: kittyweb:latest
    build:
      context: ./kittyweb
    container_name: kittyweb
    restart: unless-stopped
    env_file: ./kittyweb/.env
    working_dir: /app
    volumes:
      - ./:/app
    ports:
      - "127.0.0.1:8000:8000"
    depends_on:
      - catvision
```
