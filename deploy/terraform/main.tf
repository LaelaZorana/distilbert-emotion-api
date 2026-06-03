# Minimal Terraform stub for deploying the container image to Fly.io.
#
# This is a starting point, not a turnkey module: it shows the shape of an IaC
# deployment (provider, app, machine running the published image, env, health).
# Apply with:
#   export FLY_API_TOKEN=...                # `fly auth token`
#   terraform init
#   terraform apply -var="image=ghcr.io/laelazorana/distilbert-emotion-api:latest"
#
# For the real model, set the `offline` var to "0" and use an image built from
# requirements-ml.txt with a larger machine (memory >= 2048).

terraform {
  required_version = ">= 1.5"
  required_providers {
    fly = {
      source  = "fly-apps/fly"
      version = "~> 0.0.23"
    }
  }
}

provider "fly" {
  # Reads FLY_API_TOKEN from the environment.
}

resource "fly_app" "emotion_api" {
  name = var.app_name
  org  = var.fly_org
}

resource "fly_ip" "emotion_api_v4" {
  app  = fly_app.emotion_api.name
  type = "v4"
}

resource "fly_ip" "emotion_api_v6" {
  app  = fly_app.emotion_api.name
  type = "v6"
}

resource "fly_machine" "emotion_api" {
  app    = fly_app.emotion_api.name
  region = var.region
  name   = "${var.app_name}-machine"
  image  = var.image

  env = {
    OFFLINE            = var.offline
    PORT               = "8000"
    LOG_LEVEL          = "INFO"
    MAX_BATCH_SIZE     = "64"
    BATCH_MAX_DELAY_MS = "5"
  }

  services = [
    {
      ports = [
        { port = 443, handlers = ["tls", "http"] },
        { port = 80, handlers = ["http"] },
      ]
      protocol      = "tcp"
      internal_port = 8000
    },
  ]

  cpus     = var.cpus
  memorymb = var.memory_mb
}
