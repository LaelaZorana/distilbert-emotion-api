output "app_name" {
  description = "Deployed Fly app name."
  value       = fly_app.emotion_api.name
}

output "app_url" {
  description = "Public URL of the deployed service."
  value       = "https://${fly_app.emotion_api.name}.fly.dev"
}

output "healthcheck_url" {
  description = "Readiness/liveness endpoint."
  value       = "https://${fly_app.emotion_api.name}.fly.dev/healthz"
}
