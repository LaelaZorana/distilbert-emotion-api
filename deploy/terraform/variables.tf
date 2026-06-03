variable "app_name" {
  description = "Fly app name (must be globally unique)."
  type        = string
  default     = "distilbert-emotion-api"
}

variable "fly_org" {
  description = "Fly.io organization slug."
  type        = string
  default     = "personal"
}

variable "region" {
  description = "Fly.io region to deploy the machine in."
  type        = string
  default     = "iad"
}

variable "image" {
  description = "Container image to run (e.g. ghcr.io/<owner>/distilbert-emotion-api:latest)."
  type        = string
}

variable "offline" {
  description = "Run with the offline stub ('1') or load the real model ('0')."
  type        = string
  default     = "1"
}

variable "cpus" {
  description = "vCPUs for the machine."
  type        = number
  default     = 1
}

variable "memory_mb" {
  description = "Memory in MB. Use >= 2048 when offline = '0' (torch + weights)."
  type        = number
  default     = 512
}
