# justpay-api

![Version: 1.4.0](https://img.shields.io/badge/Version-1.4.0-informational?style=flat-square)
![AppVersion: 41](https://img.shields.io/badge/AppVersion-41-informational?style=flat-square)

JustPay API backend service for payment processing and tenancy management.

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| replicaCount | int | `2` | Number of pod replicas |
| image.repository | string | `"registry.k8s.local/justpay-api"` | Container image repository |
| image.tag | string | `"41"` | Container image tag |
| image.pullPolicy | string | `"IfNotPresent"` | Image pull policy |
| service.type | string | `"ClusterIP"` | Kubernetes service type |
| service.port | int | `8080` | Service port |
| ingress.enabled | bool | `true` | Enable ingress resource |
| ingress.host | string | `"api.justpay.ng"` | Ingress hostname |
| resources.limits.cpu | string | `"500m"` | CPU resource limit |
| resources.limits.memory | string | `"512Mi"` | Memory resource limit |
| resources.requests.cpu | string | `"100m"` | CPU resource request |
| resources.requests.memory | string | `"128Mi"` | Memory resource request |
| postgresql.enabled | bool | `true` | Deploy PostgreSQL subchart |
| redis.enabled | bool | `true` | Deploy Redis subchart |
| env.DATABASE_URL | string | `""` | PostgreSQL connection string |
| env.REDIS_URL | string | `""` | Redis connection string |
